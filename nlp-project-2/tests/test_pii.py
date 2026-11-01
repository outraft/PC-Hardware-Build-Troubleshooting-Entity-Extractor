"""Tests for common PII detection and redaction."""

import unittest

import spacy

from src.pc_hardware_ner.pii import detect_pii


class PIITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp = spacy.load("en_core_web_sm")

    def test_detects_and_redacts_email_phone_and_ipv4(self) -> None:
        text = "Email alice@example.com or call +1 (415) 555-2671; device 192.168.1.10."

        result = detect_pii(self.nlp, text)

        labels = {item["label"] for item in result["pii"]}
        self.assertTrue({"EMAIL", "PHONE", "IP_ADDRESS"}.issubset(labels))
        self.assertNotIn("alice@example.com", result["redacted_text"])
        self.assertNotIn("415) 555-2671", result["redacted_text"])
        self.assertNotIn("192.168.1.10", result["redacted_text"])


if __name__ == "__main__":
    unittest.main()
