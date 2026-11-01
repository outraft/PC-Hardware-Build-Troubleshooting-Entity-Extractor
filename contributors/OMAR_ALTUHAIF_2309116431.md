# Contribution Record — Omar Altuhaif

**Student ID:** 2309116431  
**Project:** PC Hardware Build & Troubleshooting Entity Extractor

## Contribution summary

Implemented the live data extraction pipeline using the StackExchange API (`superuser` site), created responsive error and rate-limit handling routines (HTTP 429 exponential backoff), integrated unit test mocking for CI stability, and updated project dependencies.

## Project implementation

- **StackExchange Scraper (`src/pc_hardware_ner/stackexchange_scraper.py`):** Built automated data collection logic fetching questions/answers tagged with hardware troubleshooting categories (`hardware-failure`, `memory`, `bsod`, etc.), stripped HTML formatting, and constructed structured JSONL outputs.
- **API Resilience & Rate Limiting:** Implemented pagination handling (`page`, `pagesize`), quota detection, exponential backoff retries for 429/5xx status codes, and StackExchange error 502 handling.
- **CI Unit Tests (`tests/test_stackexchange_scraper.py`):** Added 20 offline unit tests utilizing `unittest.mock` for network call simulation, assuring 100% execution pass rate across CI environments without live API requests.
- **Dependencies (`requirements.txt`):** Added `requests==2.34.2` to ensure consistent dependency resolution in GitHub Actions.
