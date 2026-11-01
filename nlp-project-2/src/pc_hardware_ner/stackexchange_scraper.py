"""Collect Super User hardware/troubleshooting questions from the Stack Exchange API."""

import argparse
import html
import json
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterator

import requests


API_ROOT = "https://api.stackexchange.com/2.3"
DEFAULT_SITE = "superuser"
DEFAULT_OUTPUT = Path("data/stackexchange_raw.jsonl")
# Canonical Super User tag names. ``ram`` and ``blue-screen`` are synonyms of
# ``memory`` and ``bsod``; ``hardware`` is not a Super User tag at all.
DEFAULT_TAGS = (
    "hardware-failure",
    "motherboard",
    "graphics-card",
    "bsod",
    "power-supply",
    "cpu",
    "memory",
)
MAX_PAGESIZE = 100
RETRY_STATUSES = {429, 500, 502, 503, 504}
# Stack Exchange reports per-IP throttling as HTTP 400 with error_id 502.
THROTTLE_ERROR_ID = 502

_BLOCK_TAGS = {
    "p", "div", "br", "pre", "blockquote", "li", "ul", "ol", "tr", "table", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
}
_SECONDS_PATTERN = re.compile(r"(\d+)\s*seconds?")


class StackExchangeError(RuntimeError):
    """Raised when the API returns an unrecoverable error."""


class _TextExtractor(HTMLParser):
    """Convert Stack Exchange HTML bodies to plain text, keeping code and block breaks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._pre_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")
        if tag == "pre":
            self._pre_depth += 1
        if tag == "img":
            alt = dict(attrs).get("alt")
            if alt:
                self.parts.append(f" {alt} ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        # Preserve line structure inside code blocks; elsewhere newlines are layout only.
        self.parts.append(data if self._pre_depth else data.replace("\n", " "))


def clean_html(value: str | None) -> str:
    """Strip HTML tags/entities and normalize whitespace to a stable plain-text form."""
    if not value:
        return ""
    extractor = _TextExtractor()
    extractor.feed(value)
    extractor.close()
    text = "".join(extractor.parts).replace(" ", " ").replace("\r", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def combine_text(title: str, body: str, accepted_answer: str | None) -> str:
    """Join cleaned parts into the single ``text`` field consumed by annotation/NER."""
    return "\n\n".join(part for part in (title, body, accepted_answer) if part)


def build_record(question: dict[str, Any], answers: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Turn one API question (plus fetched accepted answers) into an output record."""
    title = clean_html(question.get("title"))
    body = clean_html(question.get("body"))
    accepted_id = question.get("accepted_answer_id")
    answer = answers.get(accepted_id) if accepted_id is not None else None
    accepted_text = clean_html(answer.get("body")) if answer else None
    return {
        "question_id": question["question_id"],
        "title": title,
        "body": body,
        "tags": list(question.get("tags", [])),
        "score": question.get("score", 0),
        "accepted_answer_id": accepted_id,
        "accepted_answer": accepted_text,
        "text": combine_text(title, body, accepted_text),
        "link": question.get("link"),
        "creation_date": question.get("creation_date"),
    }


class StackExchangeScraper:
    """Paginated, rate-limit-aware Stack Exchange client.

    ``session``, ``sleep`` and ``clock`` are injectable so tests can run offline
    without real waiting.
    """

    def __init__(
        self,
        site: str = DEFAULT_SITE,
        key: str | None = None,
        pagesize: int = MAX_PAGESIZE,
        max_retries: int = 5,
        base_delay: float = 1.0,
        min_interval: float = 0.1,
        timeout: float = 30.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= pagesize <= MAX_PAGESIZE:
            raise ValueError(f"pagesize must be between 1 and {MAX_PAGESIZE}")
        self.site = site
        self.key = key
        self.pagesize = pagesize
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.min_interval = min_interval
        self.timeout = timeout
        self.session = session or requests.Session()
        self.sleep = sleep
        self.clock = clock
        self.quota_remaining: int | None = None
        self._next_request_at = 0.0

    def _wait_turn(self) -> None:
        delay = self._next_request_at - self.clock()
        if delay > 0:
            self.sleep(delay)

    def _retry_delay(self, response: requests.Response, payload: dict[str, Any], attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        match = _SECONDS_PATTERN.search(str(payload.get("error_message", "")))
        if match:
            return float(match.group(1))
        return self.base_delay * (2 ** attempt)

    def request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET one API page, retrying on HTTP 429/5xx/throttle errors and honoring ``backoff``."""
        query = {"site": self.site, **params}
        if self.key:
            query["key"] = self.key

        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            try:
                response = self.session.get(f"{API_ROOT}{path}", params=query, timeout=self.timeout)
            except requests.RequestException as error:
                if attempt == self.max_retries:
                    raise StackExchangeError(f"request to {path} failed: {error}") from error
                self.sleep(self.base_delay * (2 ** attempt))
                continue
            self._next_request_at = self.clock() + self.min_interval

            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}

            throttled = response.status_code == 400 and payload.get("error_id") == THROTTLE_ERROR_ID
            if response.status_code in RETRY_STATUSES or throttled:
                if attempt == self.max_retries:
                    raise StackExchangeError(
                        f"{path}: gave up after {self.max_retries} retries (HTTP {response.status_code})"
                    )
                self.sleep(self._retry_delay(response, payload, attempt))
                continue

            if response.status_code != 200 or "error_id" in payload:
                raise StackExchangeError(
                    f"{path}: HTTP {response.status_code} "
                    f"{payload.get('error_name', '')}: {payload.get('error_message', response.text[:200])}"
                )

            self.quota_remaining = payload.get("quota_remaining", self.quota_remaining)
            backoff = payload.get("backoff")
            if backoff:
                # The API requires waiting ``backoff`` seconds before hitting this method again.
                self._next_request_at = max(self._next_request_at, self.clock() + float(backoff))
            return payload

        raise StackExchangeError(f"{path}: retries exhausted")  # pragma: no cover

    def iter_questions(self, tag: str, max_pages: int | None = None) -> Iterator[list[dict[str, Any]]]:
        """Yield pages of questions for one tag until ``has_more`` is false or ``max_pages``."""
        page = 1
        while max_pages is None or page <= max_pages:
            payload = self.request(
                "/questions",
                {
                    "tagged": tag,
                    "page": page,
                    "pagesize": self.pagesize,
                    "order": "desc",
                    "sort": "votes",
                    "filter": "withbody",
                },
            )
            yield payload.get("items", [])
            if not payload.get("has_more") or self.quota_remaining == 0:
                return
            page += 1

    def fetch_answers(self, answer_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch answer bodies by id, in batches of up to 100 ids per request."""
        answers: dict[int, dict[str, Any]] = {}
        for start in range(0, len(answer_ids), MAX_PAGESIZE):
            batch = answer_ids[start:start + MAX_PAGESIZE]
            ids = ";".join(str(answer_id) for answer_id in batch)
            payload = self.request(f"/answers/{ids}", {"pagesize": MAX_PAGESIZE, "filter": "withbody"})
            for item in payload.get("items", []):
                answers[item["answer_id"]] = item
        return answers

    def scrape(
        self,
        tags: tuple[str, ...] | list[str] = DEFAULT_TAGS,
        max_pages: int | None = None,
        min_score: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield cleaned, de-duplicated records for questions carrying any of ``tags``.

        The API's ``tagged=a;b`` means *all* tags, so each tag is queried separately.
        """
        seen: set[int] = set()
        for tag in tags:
            for questions in self.iter_questions(tag, max_pages):
                fresh = [
                    question for question in questions
                    if question["question_id"] not in seen
                    and (min_score is None or question.get("score", 0) >= min_score)
                ]
                seen.update(question["question_id"] for question in fresh)
                accepted_ids = [q["accepted_answer_id"] for q in fresh if q.get("accepted_answer_id")]
                answers = self.fetch_answers(accepted_ids) if accepted_ids else {}
                for question in fresh:
                    yield build_record(question, answers)
                if self.quota_remaining == 0:
                    return


def write_jsonl(records: Iterator[dict[str, Any]], path: Path) -> int:
    """Write records as UTF-8 JSONL and return how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL path")
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS), help="Tags to collect (any match)")
    parser.add_argument("--site", default=DEFAULT_SITE, help="Stack Exchange site parameter")
    parser.add_argument("--max-pages", type=int, default=5, help="Pages per tag; 0 for no limit")
    parser.add_argument("--pagesize", type=int, default=MAX_PAGESIZE, help="Questions per page (1-100)")
    parser.add_argument("--min-score", type=int, help="Skip questions scoring below this value")
    parser.add_argument(
        "--key",
        default=os.environ.get("STACKEXCHANGE_KEY"),
        help="Optional API key for a higher daily quota (default: $STACKEXCHANGE_KEY)",
    )
    args = parser.parse_args()

    scraper = StackExchangeScraper(site=args.site, key=args.key, pagesize=args.pagesize)
    records = scraper.scrape(args.tags, max_pages=args.max_pages or None, min_score=args.min_score)
    count = write_jsonl(records, Path(args.output))
    print(f"Wrote {count} records to {args.output} (quota remaining: {scraper.quota_remaining})")


if __name__ == "__main__":
    main()
