# Contribution Record — Doruk Ünsal

**Student ID:** 2309011046  
**Project:** PC Hardware Build & Troubleshooting Entity Extractor
**Project Handle:** Hadessd4

## Contribution summary

Built the external data collection, preprocessing, and annotation pipeline using the Stack Exchange REST API.Because the core repository consumes pre-cleaned and pre-feature-engineered text while leaving data acquisition outside the codebase, this contribution bridges the gap between raw community troubleshooting posts and validated, character-aligned JSONL training records across all 17 target labels.

## Data collection and API ingestion

- Implemented an automated ingestion client targeting the Stack Exchange REST API v2.3 (site=superuser, filter=withbody) to retrieve real-world PC hardware build and troubleshooting threads across domain tags (hardware, gpu, cpu, motherboards, ram, bsod, thermal, power-supply).
- Added pagination, rate-limit backoff, and deduplication logic to collect a raw corpus of user-reported hardware failures, system specifications, and diagnostic steps.
- AFiltered retrieved posts by minimum token length and technical keyword density to isolate sentences containing both hardware configuration details and troubleshooting context.


## Text sanitization and feature engineering

- Built an HTML and markup sanitization pipeline to clean raw body payloads returned by the Stack Exchange API, stripping <pre><code> crash logs, memory dumps, blockquotes, URLs, and HTML entities while preserving sentence structure.
- Implemented domain-specific normalization rules to standardize inconsistent user shorthand for clock speeds, capacities, power ratings, and temperatures (such as unifying 88C, 88 °C, and 88deg, or separating fused numeric-unit tokens) so character spans align cleanly with spaCy's tokenizer.
- Segmented multi-paragraph forum posts into individual, pre-cleaned sentence records ready for span annotation and downstream POS/NER ingestion.

## Annotation workflow and JSONL export

- Annotated and curated the collected SuperUser troubleshooting corpus across the full 17-label taxonomy (COMPONENT_TYPE, BRAND, MODEL, SOCKET, CHIPSET, FORM_FACTOR, MEMORY_TYPE, CAPACITY, INTERFACE, SPEED, POWER, DIMENSION, PRICE, SYMPTOM, ERROR_CODE, ACTION, and OS), directly providing the real-world SYMPTOM, ERROR_CODE, and ACTION spans that static part catalogs lack.
- Built a pre-annotation and export script that combines regex/heuristic pattern matching (for hex stop codes like 0x00000116, capacities, interfaces, and OS versions) with human verification in Label Studio.
- Added a span-alignment and serialization step that resolves overlapping entities, snaps character start/end offsets to exact token boundaries, and exports deterministic train and holdout .jsonl files matching the repository's validation schema.

## Documentation and data-pipeline validation

- Documented the Stack Exchange API query parameters, HTML sanitization steps, annotation guidelines for resolving ambiguous hardware/troubleshooting spans, and inter-annotator agreement checks.
- Added validation checks on the exported JSONL splits to verify zero overlapping spans, valid character indices, and balanced representation across both hardware-specification and troubleshooting labels before model training.