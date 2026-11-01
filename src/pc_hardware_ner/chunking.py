"""Extract base noun-phrase chunks and token-level IOB chunk tags."""

import argparse
import json
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language


def chunk_text(nlp: Language, text: str) -> dict[str, Any]:
    """Use spaCy's dependency parser to identify base noun phrases (NPs)."""
    doc = nlp(text)
    token_chunks = ["O"] * len(doc)
    chunks = []

    for span in doc.noun_chunks:
        chunks.append(
            {"text": span.text, "start": span.start_char, "end": span.end_char, "label": "NP"}
        )
        for position in range(span.start, span.end):
            token_chunks[position] = "B-NP" if position == span.start else "I-NP"

    return {
        "text": text,
        "chunks": chunks,
        "tokens": [
            {
                "text": token.text,
                "pos": token.pos_,
                "dep": token.dep_,
                "chunk_tag": token_chunks[token.i],
            }
            for token in doc
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model with a dependency parser")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to chunk")
    source.add_argument("--input", help="JSONL input; each line must contain a text field")
    parser.add_argument("--output", help="Output JSONL file in batch mode; defaults to stdout")
    parser.add_argument("--pretty", action="store_true", help="Indent single-text JSON output")
    args = parser.parse_args()

    nlp = spacy.load(args.model)
    if "parser" not in nlp.pipe_names:
        parser.error("the selected spaCy model does not contain a dependency parser")

    if args.text is not None:
        if args.output:
            parser.error("--output is only available with --input")
        print(json.dumps(chunk_text(nlp, args.text), ensure_ascii=False, indent=2 if args.pretty else None))
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
                result = json.dumps(chunk_text(nlp, text), ensure_ascii=False)
                if output_stream:
                    output_stream.write(result + "\n")
                else:
                    print(result)
    finally:
        if output_stream:
            output_stream.close()


if __name__ == "__main__":
    main()
