"""Train a custom spaCy NER component while retaining its English POS tagger."""

import argparse
import random
from pathlib import Path

import spacy
from spacy.scorer import Scorer
from spacy.training import Example
from spacy.util import fix_random_seed, minibatch

from .data import load_jsonl, make_examples
from .labels import ENTITY_LABELS


def split_examples(
    examples: list[Example], test_size: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Create a deterministic holdout when there are enough examples."""
    if len(examples) < 5 or test_size <= 0:
        return examples, []
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    test_count = max(1, round(len(examples) * test_size))
    test_count = min(test_count, len(examples) - 2)
    test_indices = set(indices[:test_count])
    train = [example for index, example in enumerate(examples) if index not in test_indices]
    test = [example for index, example in enumerate(examples) if index in test_indices]
    return train, test


def evaluate(nlp: spacy.Language, examples: list[Example]) -> dict[str, float]:
    """Score NER spans in the holdout set."""
    if not examples:
        return {}
    predictions = [Example(nlp(example.reference.text), example.reference) for example in examples]
    return Scorer().score(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to annotated JSONL")
    parser.add_argument("--test-data", help="Optional separate held-out JSONL (overrides random split)")
    parser.add_argument("--output", default="models/pc-hardware-ner", help="Model output directory")
    parser.add_argument("--base-model", default="en_core_web_sm", help="spaCy English POS base model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not 0 <= args.test_size < 1:
        parser.error("--test-size must be in [0, 1)")

    fix_random_seed(args.seed)
    records = load_jsonl(args.data)
    nlp = spacy.load(args.base_model)
    if "ner" in nlp.pipe_names:
        nlp.remove_pipe("ner")
    ner = nlp.add_pipe("ner", last=True)
    for label in ENTITY_LABELS:
        ner.add_label(label)

    all_examples = make_examples(nlp, records)
    if args.test_data:
        train_examples = all_examples
        test_examples = make_examples(nlp, load_jsonl(args.test_data))
    else:
        train_examples, test_examples = split_examples(all_examples, args.test_size, args.seed)

    # Initialize only the newly added NER component so the pretrained POS model stays intact.
    ner.initialize(get_examples=lambda: train_examples, nlp=nlp)
    optimizer = nlp.resume_training()
    random_generator = random.Random(args.seed)

    with nlp.select_pipes(enable=["ner"]):
        for epoch in range(args.epochs):
            random_generator.shuffle(train_examples)
            losses: dict[str, float] = {}
            for batch in minibatch(train_examples, size=8):
                nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
            print(f"epoch {epoch + 1}/{args.epochs} - ner loss: {losses.get('ner', 0.0):.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)
    print(f"Saved model to {output_path}")

    if test_examples:
        metrics = evaluate(nlp, test_examples)
        print(
            "Holdout NER scores: "
            f"precision={metrics.get('ents_p', 0.0):.3f}, "
            f"recall={metrics.get('ents_r', 0.0):.3f}, "
            f"F1={metrics.get('ents_f', 0.0):.3f}"
        )
    else:
        print("Holdout evaluation skipped (need at least 5 examples and a non-zero test size, or pass --test-data).")


if __name__ == "__main__":
    main()
