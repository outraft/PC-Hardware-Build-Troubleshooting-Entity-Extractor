"""Detect and redact common PII in text supplied by the upstream data pipeline."""

import argparse
import ipaddress
import json
import re
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "EMAIL",
        re.compile(
            r"(?i)(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
            r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+"
        ),
    ),
    ("IP_ADDRESS", re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d)")),
    ("PHONE", re.compile(r"(?<!\w)\+?\d[\d\s()./-]{5,}\d(?!\w)")),
    ("USERNAME", re.compile(r"(?<![\w@])@[A-Za-z0-9_]{2,}")),
)
_MODEL_PII_LABELS = {"PERSON", "GPE", "LOC"}


def detect_pii(nlp: Language, text: str) -> dict[str, Any]:
    """Return common PII spans and a redacted copy of ``text``.

    This baseline is not exhaustive. Names/locations can be missed or over-redacted;
    review results before sharing any user-generated text.
    """
    doc = nlp(text)
    priority = {"EMAIL": 0, "IP_ADDRESS": 1, "PHONE": 2, "USERNAME": 3, "PERSON": 4, "GPE": 5, "LOC": 6}
    candidates: list[tuple[int, int, str, int]] = []

    for label, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group()
            value = raw.strip(" \t\r\n.,;:")
            leading = len(raw) - len(raw.lstrip(" \t\r\n.,;:"))
            start = match.start() + leading
            end = start + len(value)
            if label == "IP_ADDRESS":
                try:
                    ipaddress.ip_address(value)
                except ValueError:
                    continue
            if label == "PHONE" and not 7 <= sum(char.isdigit() for char in value) <= 15:
                continue
            candidates.append((start, end, label, priority[label]))

    for entity in doc.ents:
        if entity.label_ in _MODEL_PII_LABELS:
            candidates.append((entity.start_char, entity.end_char, entity.label_, priority[entity.label_]))

    selected: list[tuple[int, int, str, int]] = []
    for candidate in sorted(candidates, key=lambda item: (item[3], item[0], -(item[1] - item[0]))):
        start, end, _, _ = candidate
        if not any(start < chosen_end and end > chosen_start for chosen_start, chosen_end, _, _ in selected):
            selected.append(candidate)
    selected.sort(key=lambda item: item[0])

    redacted = text
    for start, end, label, _ in reversed(selected):
        redacted = redacted[:start] + f"<PII:{label}>" + redacted[end:]

    return {
        "text": text,
        "redacted_text": redacted,
        "pii": [
            {"text": text[start:end], "start": start, "end": end, "label": label}
            for start, end, label, _ in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model for PERSON/LOCATION recognition")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to inspect and redact")
    source.add_argument("--input", help="JSONL input; each line must contain a text field")
    parser.add_argument("--output", help="Output JSONL file in batch mode; defaults to stdout")
    parser.add_argument("--pretty", action="store_true", help="Indent single-text JSON output")
    args = parser.parse_args()

    nlp = spacy.load(args.model)
    if "ner" not in nlp.pipe_names:
        parser.error("the selected spaCy model needs an NER component for person/location recognition")

    if args.text is not None:
        if args.output:
            parser.error("--output is only available with --input")
        print(json.dumps(detect_pii(nlp, args.text), ensure_ascii=False, indent=2 if args.pretty else None))
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
                result = json.dumps(detect_pii(nlp, text), ensure_ascii=False)
                if output_stream:
                    output_stream.write(result + "\n")
                else:
                    print(result)
    finally:
        if output_stream:
            output_stream.close()


if __name__ == "__main__":
    main()
