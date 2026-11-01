"""Tests for noun-phrase chunking."""

import unittest

import spacy

from src.pc_hardware_ner.chunking import chunk_text


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp = spacy.load("en_core_web_sm")

    def test_emits_noun_chunks_and_iob_tags_with_matching_offsets(self) -> None:
        text = "The Ryzen 7 7800X3D pairs with an AM5 motherboard."

        result = chunk_text(self.nlp, text)

        self.assertTrue(result["chunks"])
        self.assertTrue(any(token["chunk_tag"] == "B-NP" for token in result["tokens"]))
        for chunk in result["chunks"]:
            self.assertEqual(text[chunk["start"]:chunk["end"]], chunk["text"])


if __name__ == "__main__":
    unittest.main()
