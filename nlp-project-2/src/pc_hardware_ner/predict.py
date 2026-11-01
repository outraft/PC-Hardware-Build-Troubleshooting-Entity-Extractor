"""Run NER and POS predictions with a saved spaCy pipeline."""

import argparse
import json
from pathlib import Path
from typing import Any

import spacy


def predict(nlp: spacy.Language, text: str) -> dict[str, Any]:
    doc = nlp(text)
    return {
        "text": text,
        "entities": [
            {"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_}
            for ent in doc.ents
        ],
        "tokens": [
            {
                "text": token.text,
                "pos": token.pos_,
                "tag": token.tag_,
                "lemma": token.lemma_,
            }
            for token in doc
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a trained spaCy model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to analyze")
    source.add_argument("--input", help="Input JSONL; each record must contain a text field")
    parser.add_argument("--output", help="Output JSONL path (batch mode; defaults to stdout)")
    parser.add_argument("--pretty", action="store_true", help="Indent single-text JSON output")
    args = parser.parse_args()

    nlp = spacy.load(args.model)
    if "ner" not in nlp.pipe_names:
        parser.error("the selected model has no NER component; train a model first")

    if args.text is not None:
        if args.output:
            parser.error("--output is only available with --input")
        print(json.dumps(predict(nlp, args.text), ensure_ascii=False, indent=2 if args.pretty else None))
        return

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stream = output_path.open("w", encoding="utf-8") if output_path else None
    try:
        with input_path.open("r", encoding="utf-8") as source_stream:
            for line_number, line in enumerate(source_stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text") if isinstance(record, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"{input_path}:{line_number}: expected a JSON object with string 'text'")
                result = json.dumps(predict(nlp, text), ensure_ascii=False)
                if output_stream:
                    output_stream.write(result + "\n")
                else:
                    print(result)
    finally:
        if output_stream:
            output_stream.close()


if __name__ == "__main__":
    main()
