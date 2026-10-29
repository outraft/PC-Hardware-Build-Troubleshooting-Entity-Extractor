"""POS-tag lexical values in structured PC-part catalog records."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import spacy
from spacy.language import Language


def _value_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    return None


def _field_values(field: str, value: Any, index: str = "") -> Iterator[tuple[str, str, str]]:
    """Flatten nested values while retaining their source field and array index."""
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_field = f"{field}.{key}" if field else str(key)
            yield from _field_values(nested_field, nested_value, index)
    elif isinstance(value, list):
        for item_index, nested_value in enumerate(value):
            yield from _field_values(field, nested_value, f"{index}[{item_index}]")
    else:
        text = _value_text(value)
        if text:
            yield field, index, text


def tag_product_record(
    nlp: Language, record: dict[str, Any], category: str | None = None
) -> dict[str, Any]:
    """Return POS tags for every lexical/numeric field value in one catalog row.

    The original field name remains attached to each token. This matters for values
    such as ``3200``: POS says it is a number, while ``field="speed"`` supplies its
    dataset meaning. Field names themselves are schema metadata, not sentence tokens.
    """
    if not isinstance(record, dict):
        raise ValueError("each product record must be a JSON object")

    values: list[tuple[str, str, str]] = []
    if category:
        values.append(("category", "", category))
    for field, value in record.items():
        if field in {"id", "url"}:
            continue
        values.extend(_field_values(str(field), value))

    token_rows: list[dict[str, Any]] = []
    for (field, value_index, _), doc in zip(values, nlp.pipe(item[2] for item in values)):
        for token in doc:
            token_rows.append(
                {
                    "text": token.text,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "lemma": token.lemma_,
                    "field": field,
                    "value_index": value_index,
                }
            )

    return {
        "name": record.get("name"),
        "category": category,
        "tokens": token_rows,
    }


def _records(path: Path) -> Iterable[dict[str, Any]]:
    """Read the source repository's JSON arrays, JSONL, or CSV exports."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield record
        return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)
        return
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError(f"{path}: expected a JSON array of product objects")
        yield from data
        return
    raise ValueError("input must be a .json, .jsonl, or .csv file")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="PC-part JSON array, JSONL, or CSV file")
    parser.add_argument("--output", help="Output JSONL path; defaults to stdout")
    parser.add_argument("--category", help="Part category for this file, e.g. cpu or motherboard")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy POS model")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    nlp = spacy.load(args.model, disable=["parser", "ner"])
    if not any(name in nlp.pipe_names for name in ("tagger", "morphologizer")):
        parser.error("the selected spaCy model does not contain a POS tagger")

    output_stream = output_path.open("w", encoding="utf-8") if output_path else None
    try:
        for record in _records(input_path):
            result = tag_product_record(nlp, record, args.category)
            line = json.dumps(result, ensure_ascii=False)
            if output_stream:
                output_stream.write(line + "\n")
            else:
                print(line)
    finally:
        if output_stream:
            output_stream.close()


if __name__ == "__main__":
    main()
