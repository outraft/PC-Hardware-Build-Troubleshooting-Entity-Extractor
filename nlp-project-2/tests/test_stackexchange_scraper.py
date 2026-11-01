"""Offline tests for the Stack Exchange scraper; all HTTP calls are mocked."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import requests

from src.pc_hardware_ner.stackexchange_scraper import (
    StackExchangeError,
    StackExchangeScraper,
    build_record,
    clean_html,
    combine_text,
    write_jsonl,
)


def _response(status: int = 200, payload: Any = None, headers: dict[str, str] | None = None) -> mock.Mock:
    response = mock.Mock(spec=requests.Response)
    response.status_code = status
    response.headers = headers or {}
    response.text = json.dumps(payload) if payload is not None else ""
    if payload is None:
        response.json.side_effect = ValueError("no JSON")
    else:
        response.json.return_value = payload
    return response


def _question(question_id: int, **fields: Any) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "title": f"Question {question_id}",
        "body": f"<p>Body {question_id}</p>",
        "tags": ["cpu"],
        "score": 1,
        "link": f"https://superuser.com/q/{question_id}",
        "creation_date": 1700000000,
        **fields,
    }


class FakeClock:
    """Deterministic clock whose time only advances when ``sleep`` is called."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ScraperTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        patcher = mock.patch.object(requests.Session, "get", autospec=True)
        self.get = patcher.start()
        self.addCleanup(patcher.stop)
        self.scraper = StackExchangeScraper(
            pagesize=2, base_delay=1.0, min_interval=0.0, max_retries=3,
            sleep=self.clock.sleep, clock=self.clock,
        )

    def called_params(self, call_index: int) -> dict[str, Any]:
        return self.get.call_args_list[call_index].kwargs["params"]

    def called_url(self, call_index: int) -> str:
        return self.get.call_args_list[call_index].args[1]


class TextFormattingTests(unittest.TestCase):
    def test_clean_html_strips_tags_and_unescapes_entities(self) -> None:
        raw = "<p>My <strong>RTX&nbsp;4070</strong> &amp; Ryzen 7 &quot;crash&quot;</p>\n<p>Second  para</p>"
        self.assertEqual(clean_html(raw), 'My RTX 4070 & Ryzen 7 "crash"\n\nSecond para')

    def test_clean_html_keeps_code_lines_and_list_items(self) -> None:
        raw = "<p>Error:</p><pre><code>STOP 0x00000124\nWHEA_UNCORRECTABLE</code></pre><ul><li>Reseat RAM</li><li>Update BIOS</li></ul>"
        self.assertEqual(
            clean_html(raw),
            "Error:\n\nSTOP 0x00000124\nWHEA_UNCORRECTABLE\n\n- Reseat RAM\n\n- Update BIOS",
        )

    def test_clean_html_handles_empty_values(self) -> None:
        self.assertEqual(clean_html(None), "")
        self.assertEqual(clean_html(""), "")

    def test_combine_text_skips_missing_parts(self) -> None:
        self.assertEqual(combine_text("Title", "Body", "Answer"), "Title\n\nBody\n\nAnswer")
        self.assertEqual(combine_text("Title", "", None), "Title")


class ResponseParsingTests(ScraperTestCase):
    def test_build_record_includes_accepted_answer(self) -> None:
        question = _question(1, title="PC won&#39;t POST", accepted_answer_id=10, tags=["motherboard", "cpu"])
        record = build_record(question, {10: {"answer_id": 10, "body": "<p>Clear the <b>CMOS</b>.</p>"}})

        self.assertEqual(record["question_id"], 1)
        self.assertEqual(record["title"], "PC won't POST")
        self.assertEqual(record["tags"], ["motherboard", "cpu"])
        self.assertEqual(record["accepted_answer"], "Clear the CMOS.")
        self.assertEqual(record["text"], "PC won't POST\n\nBody 1\n\nClear the CMOS.")

    def test_build_record_without_accepted_answer(self) -> None:
        record = build_record(_question(2), {})
        self.assertIsNone(record["accepted_answer_id"])
        self.assertIsNone(record["accepted_answer"])
        self.assertEqual(record["text"], "Question 2\n\nBody 2")

    def test_scrape_paginates_fetches_answers_and_deduplicates_across_tags(self) -> None:
        self.get.side_effect = [
            _response(payload={"items": [_question(1, accepted_answer_id=10), _question(2)], "has_more": True, "quota_remaining": 50}),
            _response(payload={"items": [{"answer_id": 10, "body": "<p>Fixed</p>"}], "quota_remaining": 49}),
            _response(payload={"items": [_question(3)], "has_more": False, "quota_remaining": 48}),
            _response(payload={"items": [_question(2), _question(4)], "has_more": False, "quota_remaining": 47}),
        ]

        records = list(self.scraper.scrape(["cpu", "memory"]))

        self.assertEqual([record["question_id"] for record in records], [1, 2, 3, 4])
        self.assertEqual(records[0]["accepted_answer"], "Fixed")
        self.assertEqual(self.called_params(0)["tagged"], "cpu")
        self.assertEqual(self.called_params(0)["page"], 1)
        self.assertEqual(self.called_params(0)["pagesize"], 2)
        self.assertEqual(self.called_params(0)["site"], "superuser")
        self.assertTrue(self.called_url(1).endswith("/answers/10"))
        self.assertEqual(self.called_params(2)["page"], 2)
        self.assertEqual(self.called_params(3)["tagged"], "memory")
        self.assertEqual(self.scraper.quota_remaining, 47)

    def test_max_pages_and_min_score_limit_results(self) -> None:
        self.get.return_value = _response(
            payload={"items": [_question(1, score=5), _question(2, score=-1)], "has_more": True, "quota_remaining": 10}
        )

        records = list(self.scraper.scrape(["cpu"], max_pages=1, min_score=0))

        self.assertEqual([record["question_id"] for record in records], [1])
        self.assertEqual(self.get.call_count, 1)

    def test_stops_when_quota_is_exhausted(self) -> None:
        self.get.return_value = _response(payload={"items": [_question(1)], "has_more": True, "quota_remaining": 0})

        records = list(self.scraper.scrape(["cpu", "memory"]))

        self.assertEqual(len(records), 1)
        self.assertEqual(self.get.call_count, 1)

    def test_api_key_is_sent_when_configured(self) -> None:
        self.get.return_value = _response(payload={"items": [], "has_more": False})
        scraper = StackExchangeScraper(key="abc", sleep=self.clock.sleep, clock=self.clock)

        list(scraper.scrape(["cpu"]))

        self.assertEqual(self.called_params(0)["key"], "abc")

    def test_non_retryable_error_raises(self) -> None:
        self.get.return_value = _response(
            400, {"error_id": 400, "error_name": "bad_parameter", "error_message": "pagesize"}
        )
        with self.assertRaisesRegex(StackExchangeError, "bad_parameter"):
            list(self.scraper.scrape(["cpu"]))
        self.assertEqual(self.get.call_count, 1)

    def test_invalid_pagesize_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StackExchangeScraper(pagesize=101)


class RateLimitTests(ScraperTestCase):
    def test_http_429_retries_with_exponential_backoff(self) -> None:
        self.get.side_effect = [
            _response(429),
            _response(429),
            _response(payload={"items": [], "has_more": False}),
        ]

        payload = self.scraper.request("/questions", {})

        self.assertEqual(payload["items"], [])
        self.assertEqual(self.get.call_count, 3)
        self.assertEqual(self.clock.sleeps, [1.0, 2.0])

    def test_http_429_honors_retry_after_header(self) -> None:
        self.get.side_effect = [
            _response(429, headers={"Retry-After": "7"}),
            _response(payload={"items": []}),
        ]
        self.scraper.request("/questions", {})
        self.assertEqual(self.clock.sleeps, [7.0])

    def test_throttle_violation_waits_for_advertised_seconds(self) -> None:
        self.get.side_effect = [
            _response(400, {"error_id": 502, "error_name": "throttle_violation",
                            "error_message": "too many requests from this IP, more requests available in 12 seconds"}),
            _response(payload={"items": []}),
        ]
        self.scraper.request("/questions", {})
        self.assertEqual(self.clock.sleeps, [12.0])

    def test_gives_up_after_max_retries(self) -> None:
        self.get.return_value = _response(429)
        with self.assertRaisesRegex(StackExchangeError, "gave up after 3 retries"):
            self.scraper.request("/questions", {})
        self.assertEqual(self.get.call_count, 4)

    def test_network_errors_are_retried(self) -> None:
        self.get.side_effect = [requests.ConnectionError("reset"), _response(payload={"items": []})]
        self.scraper.request("/questions", {})
        self.assertEqual(self.get.call_count, 2)
        self.assertEqual(self.clock.sleeps, [1.0])

    def test_backoff_field_delays_next_request(self) -> None:
        self.get.side_effect = [
            _response(payload={"items": [], "backoff": 10}),
            _response(payload={"items": []}),
        ]
        self.scraper.request("/questions", {})
        self.assertEqual(self.clock.sleeps, [])
        self.scraper.request("/questions", {})
        self.assertEqual(self.clock.sleeps, [10.0])

    def test_min_interval_spaces_requests(self) -> None:
        self.scraper.min_interval = 0.5
        self.get.return_value = _response(payload={"items": []})
        self.scraper.request("/questions", {})
        self.scraper.request("/questions", {})
        self.assertEqual(self.clock.sleeps, [0.5])


class OutputTests(ScraperTestCase):
    def test_write_jsonl_produces_one_valid_object_per_line(self) -> None:
        self.get.side_effect = [
            _response(payload={"items": [_question(1, title="Ünïcode GPU", accepted_answer_id=10), _question(2)], "has_more": False}),
            _response(payload={"items": [{"answer_id": 10, "body": "<p>Reseat it</p>"}]}),
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "stackexchange_raw.jsonl"
            count = write_jsonl(self.scraper.scrape(["graphics-card"]), path)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, 2)
        self.assertEqual(len(lines), 2)
        records = [json.loads(line) for line in lines]
        expected_keys = {
            "question_id", "title", "body", "tags", "score", "accepted_answer_id",
            "accepted_answer", "text", "link", "creation_date",
        }
        for record in records:
            self.assertEqual(set(record), expected_keys)
            self.assertIsInstance(record["text"], str)
        self.assertEqual(records[0]["title"], "Ünïcode GPU")
        self.assertIn("Reseat it", records[0]["text"])


if __name__ == "__main__":
    unittest.main()
