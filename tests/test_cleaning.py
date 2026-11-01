"""Test Stack Exchange data cleaning without network access."""

import json
import tempfile
import unittest
from pathlib import Path

from src.pc_hardware_ner.cleaning import (
    clean_records,
    clean_text,
    write_jsonl,
)


class TestCleaning(unittest.TestCase):

    def run_cleaning(self, records):
        """Write temporary input and run the cleaning function."""
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"

            with input_path.open("w", encoding="utf-8") as source:
                for record in records:
                    source.write(json.dumps(record) + "\n")

            return clean_records(input_path)

    def test_normalize_whitespace(self):
        text = "  GPU\r\nNo\u00a0display\rCheck cable  "
        expected = "GPU\nNo display\nCheck cable"
        self.assertEqual(clean_text(text), expected)

    def test_preserve_technical_content(self):
        text = (
            "RTX 4070 / DDR5 / 850 W / 0x00000124\n"
            "    diagnostic line"
        )
        self.assertEqual(clean_text(text), text)

    def test_keep_record_without_answer_and_preserve_metadata(self):
        record = {
            "question_id": 1,
            "text": "No display from my graphics card.",
            "accepted_answer": None,
            "tags": ["graphics-card"],
            "link": "https://superuser.com/questions/1",
        }

        cleaned, rejected = self.run_cleaning([record])

        self.assertEqual(cleaned, [record])
        self.assertEqual(rejected, [])

    def test_reject_duplicate_id(self):
        records = [
            {"question_id": 1, "text": "No display."},
            {"question_id": 1, "text": "Updated question."},
        ]

        cleaned, rejected = self.run_cleaning(records)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(rejected[0]["reason"], "duplicate_question_id")
        self.assertEqual(rejected[0]["record"], records[1])

    def test_reject_duplicate_cleaned_text(self):
        records = [
            {"question_id": 1, "text": "No display."},
            {"question_id": 2, "text": "  No display.  "},
        ]

        cleaned, rejected = self.run_cleaning(records)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(rejected[0]["reason"], "duplicate_text")

    def test_reject_invalid_records(self):
        cases = [
            ([], "expected_json_object"),
            ({"text": "No display."}, "invalid_question_id"),
            ({"question_id": True, "text": "Fault."}, "invalid_question_id"),
            ({"question_id": 0, "text": "Fault."}, "invalid_question_id"),
            ({"question_id": 1, "text": None}, "invalid_text"),
            ({"question_id": 1, "text": " \n "}, "empty_text"),
            (
                {"question_id": 1, "text": "RAM", "entities": []},
                "annotated_record_requires_separate_handling",
            ),
        ]

        for record, expected_reason in cases:
            with self.subTest(reason=expected_reason, record=record):
                cleaned, rejected = self.run_cleaning([record])

                self.assertEqual(cleaned, [])
                self.assertEqual(rejected[0]["reason"], expected_reason)

    def test_invalid_record_does_not_block_valid_record(self):
        records = [
            {"question_id": 1, "text": ""},
            {"question_id": 1, "text": "RAM test failed."},
        ]

        cleaned, rejected = self.run_cleaning(records)

        self.assertEqual(cleaned, [records[1]])
        self.assertEqual(len(rejected), 1)

    def test_handle_invalid_json_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            input_path.write_text(
                '\nnot valid json\n'
                '{"question_id": 1, "text": "RAM failed."}\n',
                encoding="utf-8-sig",
            )

            cleaned, rejected = clean_records(input_path)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(rejected[0]["reason"], "invalid_json")
        self.assertEqual(rejected[0]["line_number"], 2)
        self.assertEqual(rejected[0]["raw_line"], "not valid json")

    def test_write_and_reload_jsonl(self):
        records = [
            {"question_id": 1, "text": "CPU temperature: 90°C"}
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "results" / "clean.jsonl"
            write_jsonl(records, output_path)

            with output_path.open("r", encoding="utf-8") as source:
                loaded = [json.loads(line) for line in source]

        self.assertEqual(loaded, records)


if __name__ == "__main__":
    unittest.main()