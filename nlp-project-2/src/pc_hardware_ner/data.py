"""Loading and validating character-offset NER examples."""

import json
from pathlib import Path
from typing import Any

from spacy.language import Language
from spacy.training import Example

from .labels import ENTITY_LABELS


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records containing text and character-offset entities."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def make_examples(nlp: Language, records: list[dict[str, Any]]) -> list[Example]:
    """Convert records to spaCy examples, rejecting invalid/unaligned spans."""
    examples: list[Example] = []
    for record_number, record in enumerate(records, start=1):
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"record {record_number}: 'text' must be a non-empty string")
        entities = record.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError(f"record {record_number}: 'entities' must be a list")

        doc = nlp.make_doc(text)
        spans: list[tuple[int, int, str]] = []
        for entity_number, entity in enumerate(entities, start=1):
            where = f"record {record_number}, entity {entity_number}"
            if not isinstance(entity, dict):
                raise ValueError(f"{where}: entity must be an object")
            start, end, label = entity.get("start"), entity.get("end"), entity.get("label")
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
                raise ValueError(f"{where}: expected valid character offsets within the text")
            if label not in ENTITY_LABELS:
                raise ValueError(
                    f"{where}: unsupported label {label!r}; choose one of {', '.join(ENTITY_LABELS)}"
                )
            span = doc.char_span(start, end, label=label, alignment_mode="strict")
            if span is None:
                raise ValueError(
                    f"{where}: [{start}, {end}) is not aligned to token boundaries; "
                    "check the text and character offsets"
                )
            spans.append((start, end, label))

        spans.sort(key=lambda item: (item[0], item[1]))
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"record {record_number}: overlapping entity spans "
                    f"[{previous[0]}, {previous[1]}) and [{current[0]}, {current[1]})"
                )
        examples.append(Example.from_dict(doc, {"entities": spans}))
    return examples
