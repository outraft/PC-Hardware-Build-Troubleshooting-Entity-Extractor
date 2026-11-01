"""Tests for POS tagging on the PC-part dataset's structured records."""

import unittest

import spacy

from src.pc_hardware_ner.dataset_pos import tag_product_record


class DatasetPOSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def test_tags_product_names_and_keeps_spec_field_context(self) -> None:
        # Shape is taken from the dataset's case-accessory JSON records.
        record = {
            "name": "NZXT HUE 2 RGB Lighting Kit",
            "price": None,
            "type": "LED Controller",
            "form_factor": 2.5,
        }

        result = tag_product_record(self.nlp, record, category="case-accessory")

        self.assertEqual(result["name"], record["name"])
        self.assertEqual(result["category"], "case-accessory")
        self.assertTrue(any(token["field"] == "name" and token["pos"] for token in result["tokens"]))
        self.assertTrue(
            any(
                token["field"] == "form_factor" and token["text"] == "2.5" and token["pos"] == "NUM"
                for token in result["tokens"]
            )
        )
        self.assertTrue(any(token["field"] == "type" for token in result["tokens"]))

    def test_array_features_keep_their_source_index(self) -> None:
        result = tag_product_record(self.nlp, {"speed": [4, 3200]})

        indexed_values = {(token["field"], token["value_index"]) for token in result["tokens"]}
        self.assertIn(("speed", "[0]"), indexed_values)
        self.assertIn(("speed", "[1]"), indexed_values)


if __name__ == "__main__":
    unittest.main()
