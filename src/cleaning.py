"""Clean and validate Stack Exchange records before annotation."""

import argparse
import json
from pathlib import Path


def clean_text(text):
    """Normalize line endings and preserve technical content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    return text.strip()


def clean_records(input_path):
    """Read JSONL records and separate invalid or duplicate entries."""
    cleaned_records = []
    rejected_records = []
    seen_ids = set()
    seen_texts = set()

    with input_path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rejected_records.append({
                    "line_number": line_number,
                    "reason": "invalid_json",
                    "raw_line": line.rstrip("\r\n"),
                })
                continue

            reason = None

            if not isinstance(record, dict):
                reason = "expected_json_object"
            elif "entities" in record:
                reason = "annotated_record_requires_separate_handling"
            elif type(record.get("question_id")) is not int:
                reason = "invalid_question_id"
            elif record["question_id"] <= 0:
                reason = "invalid_question_id"
            elif not isinstance(record.get("text"), str):
                reason = "invalid_text"
            else:
                text = clean_text(record["text"])

                if not text:
                    reason = "empty_text"
                elif record["question_id"] in seen_ids:
                    reason = "duplicate_question_id"
                elif text in seen_texts:
                    reason = "duplicate_text"

            if reason:
                rejected_records.append({
                    "line_number": line_number,
                    "reason": reason,
                    "record": record,
                })
                continue

            cleaned_record = record.copy()
            cleaned_record["text"] = text
            cleaned_records.append(cleaned_record)

            seen_ids.add(record["question_id"])
            seen_texts.add(text)

    return cleaned_records, rejected_records


def write_jsonl(records, output_path):
    """Save records as UTF-8 JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    """Read command-line arguments and run the cleaning process."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/stackexchange_raw.jsonl"),
        help="Path to the raw JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stackexchange_clean.jsonl"),
        help="Path to the cleaned JSONL file",
    )
    parser.add_argument(
        "--rejected",
        type=Path,
        default=Path("data/stackexchange_rejected.jsonl"),
        help="Path to rejected records and their reasons",
    )

    args = parser.parse_args()

    paths = [
        args.input.resolve(),
        args.output.resolve(),
        args.rejected.resolve(),
    ]

    if len(set(paths)) != 3:
        parser.error("Input, output, and rejected paths must be different.")

    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")

    cleaned_records, rejected_records = clean_records(args.input)

    write_jsonl(cleaned_records, args.output)
    write_jsonl(rejected_records, args.rejected)

    processed_count = len(cleaned_records) + len(rejected_records)

    print(f"Records processed: {processed_count}")
    print(f"Records kept: {len(cleaned_records)}")
    print(f"Records rejected: {len(rejected_records)}")
    print(f"Cleaned data saved to: {args.output}")
    print(f"Rejected records saved to: {args.rejected}")


if __name__ == "__main__":
    main()