"""Tests for the final prediction/serialization stage."""

import unittest

import spacy

from src.pc_hardware_ner.predict import predict


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp = spacy.load("en_core_web_sm")
        ruler = cls.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(
            [
                {"label": "MODEL", "pattern": "RTX 4070 Super"},
                {"label": "MODEL", "pattern": "Ryzen 7 7800X3D"},
            ]
        )

    def test_simple_request_extracts_both_hardware_counterparts(self) -> None:
        text = "Can I pair the RTX 4070 Super with a Ryzen 7 7800X3D?"

        result = predict(self.nlp, text)

        self.assertEqual(
            [(entity["text"], entity["label"]) for entity in result["entities"]],
            [("RTX 4070 Super", "MODEL"), ("Ryzen 7 7800X3D", "MODEL")],
        )
        for entity in result["entities"]:
            start = text.index(entity["text"])
            self.assertEqual((entity["start"], entity["end"]), (start, start + len(entity["text"])))

    def test_prediction_includes_pos_for_each_token(self) -> None:
        result = predict(self.nlp, "Can I pair the RTX 4070 Super?")

        self.assertTrue(result["tokens"])
        self.assertTrue(all(token["text"] and token["pos"] and token["tag"] for token in result["tokens"]))


if __name__ == "__main__":
    unittest.main()
