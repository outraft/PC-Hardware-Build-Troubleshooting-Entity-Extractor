# Contribution Record — Ersin Ege Tezgüler

**Student ID:** 2409011008  
**Project:** PC Hardware Build & Troubleshooting Entity Extractor

## Contribution summary

Created the initial NLP project structure and documented the end-to-end workflow for extracting PC-hardware entities and returning part-of-speech (POS) tags. The implementation uses spaCy, consumes pre-cleaned and pre-feature-engineered text, and leaves scraping and data cleaning outside this repository.

## Project implementation

- Added a spaCy-based pipeline that preserves the English POS model (`en_core_web_sm`) and trains a custom named-entity recognition (NER) component from annotated examples.
- Added JSONL annotation loading and validation, including checks for supported labels, valid character offsets, token-aligned spans, and overlapping entities.
- Added a training command that supports a deterministic random holdout or an optional separate held-out JSONL file, configurable epochs, seed, base model, and output path. It reports holdout NER precision, recall, and F1 when evaluation data is available.
- Added a prediction command for single text and JSONL batches. Results include extracted entities with character offsets and per-token POS, detailed tag, and lemma.
- Added a dataset-aware POS command for the PC-part JSON, JSONL, and CSV records. It POS-tags product names and field values while retaining the source feature key and array index for each token, so numeric tags remain interpretable in context (for example, `NUM` attached to `speed`).
- Added parser-based noun-phrase chunking with NP spans, offsets, and token IOB tags.
- Added a PII detection/redaction CLI for common identifiers and model-recognized people/locations in upstream-prepared text.
- Added a small synthetic annotation example file to demonstrate the input format. It is a workflow/demo fixture, not a sufficient training corpus.

## Entity labels

Defined the following entity taxonomy:

`COMPONENT_TYPE`, `BRAND`, `MODEL`, `SOCKET`, `CHIPSET`, `FORM_FACTOR`, `MEMORY_TYPE`, `CAPACITY`, `INTERFACE`, `SPEED`, `POWER`, `DIMENSION`, `PRICE`, `SYMPTOM`, `ERROR_CODE`, `ACTION`, and `OS`.

POS tags are predicted separately by the pretrained English pipeline. Troubleshooting labels (`SYMPTOM`, `ERROR_CODE`, and `ACTION`) require separately sourced and manually or otherwise reliably annotated troubleshooting text; the PC parts catalog alone cannot train them.

## Dependencies and reproducibility

- Added `requirements.txt` with pinned spaCy, English model, Typer, and Click versions.
- Created a local `.venv` for dependency installation and validation. The environment is ignored by Git and is not part of the committed project.
- Added ignore rules for local datasets and generated models while keeping the synthetic annotation example tracked.

## Documentation and tests

- Replaced the placeholder README with setup instructions, project layout, tag definitions, annotation JSONL format and offsets, training and inference commands, suggested data workflow, dataset attribution, and limitations.
- Documented [docyx/pc-part-dataset](https://github.com/docyx/pc-part-dataset) as a catalog source rather than an annotated NLP corpus. No scraping or dataset download is included.
- Added prediction-stage unit tests that verify a simple hardware request extracts both referenced model counterparts, returns correct character offsets, and includes POS tags.
- Added catalog-record POS tests using the source dataset's product shape; they check POS output and preservation of specification-field/index context.
- Added chunking and PII tests covering phrase offsets/IOB output and common identifier masking.

## Validation performed

- Installed the pinned dependencies into the ignored local virtual environment.
- Ran the unit test suite: **6 tests passed**.
- Compiled the Python source and test files successfully.
- Ran a one-epoch training smoke test using the tiny synthetic examples and loaded the resulting model for a prediction smoke test.
- Confirmed the synthetic annotation spans map to the expected text and ran `git diff --check`.

The smoke-test corpus is intentionally too small to establish model quality. Real NER performance requires a representative, consistently annotated dataset and an appropriately separated evaluation set. Chunking and PII are parser/rule-based baselines and need evaluation on representative troubleshooting text. Punctuation restoration is not included because the planned Stack Exchange prose is already punctuated. The extractor identifies text entities; it does not validate PC compatibility or diagnose hardware faults.
