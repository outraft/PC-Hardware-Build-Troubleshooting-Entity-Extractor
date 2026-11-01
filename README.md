# PC Hardware Build & Troubleshooting Entity Extractor

A Python NLP starter project for extracting PC-hardware entities and returning POS tags for every token. It uses **spaCy**: the English pipeline supplies POS tagging, while a trainable NER component is fine-tuned on your annotated, pre-cleaned examples.

> **Text source:** [Stack Exchange Super User hardware questions](https://api.stackexchange.com/2.3/questions?site=superuser&tagged=cpu%3Bmotherboard%3Bmemory%3Bpower-supply%3Bboot&pagesize=100&order=desc&sort=creation&filter=withbody). This is a focused query for questions tagged with CPU, motherboard, memory, power supply, and boot. The data team handles API collection, cleaning, and feature engineering; this project consumes their prepared text and annotations. No scraper is included.

## What it does

- Predicts token-level POS tags for prepared hardware troubleshooting text.
- Predicts custom, span-level NER tags for hardware/build and troubleshooting text.
- Extracts base noun-phrase chunks (NP) with token-level IOB tags.
- Detects and redacts common PII from upstream-provided text.
- Trains from JSONL annotations, evaluates on a held-out split when the dataset is large enough, saves a reloadable spaCy model, and predicts from text or JSONL.
- Accepts already-cleaned text. The model does not fetch, clean, or engineer data.

## Entity tag set

Use these exact labels in the annotation JSONL. Apply the most specific applicable label to each entity span; avoid overlapping spans.

| Label | Use for | Example |
| --- | --- | --- |
| `COMPONENT_TYPE` | Hardware part category | `graphics card`, `CPU cooler`, `power supply` |
| `BRAND` | Manufacturer/brand | `AMD`, `Noctua`, `Corsair` |
| `MODEL` | Product/model name or number | `Ryzen 7 7800X3D`, `RTX 4070 Super`, `NH-D15` |
| `SOCKET` | CPU/motherboard socket | `AM5`, `LGA1700` |
| `CHIPSET` | Board/chipset family | `B650`, `Z790` |
| `FORM_FACTOR` | Board/case/PSU form factor | `ATX`, `Mini-ITX`, `SFX` |
| `MEMORY_TYPE` | RAM generation/type | `DDR5`, `DDR4` |
| `CAPACITY` | Storage or memory capacity | `32 GB`, `2 TB` |
| `INTERFACE` | Bus, storage, or connector interface | `PCIe 4.0`, `NVMe`, `SATA` |
| `SPEED` | Clock, transfer, or fan speed | `6000 MT/s`, `5.0 GHz`, `1800 RPM` |
| `POWER` | Power rating/requirement | `850 W`, `65W` |
| `DIMENSION` | Physical dimension | `140 mm`, `30 cm` |
| `PRICE` | Price or budget | `$499`, `under $1,000` |
| `SYMPTOM` | Reported fault/behavior | `random shutdowns`, `no display` |
| `ERROR_CODE` | Error code or diagnostic identifier | `DRAM LED`, `0x00000124` |
| `ACTION` | Troubleshooting/build action | `reseat the RAM`, `update BIOS` |
| `OS` | Operating system | `Windows 11`, `Ubuntu` |

Use `SYMPTOM`, `ERROR_CODE`, and `ACTION` only when training on labeled troubleshooting text. The parts catalog by itself cannot teach these labels. If a phrase could be tagged multiple ways, settle a consistent annotation policy before scaling the dataset.

POS tags are produced by spaCy's English model and are **not** custom NER labels. They do not require POS annotations. The included model is English-language; other languages need a matching POS base model.

## Project layout

```text
data/
  annotations.example.jsonl # Tiny synthetic format example
  train.jsonl             # Your annotated examples (local)
models/                   # Generated model output (not included)
.github/workflows/
  ci-cd.yml               # Test matrix and version-tag release bundle
src/pc_hardware_ner/
  __init__.py
  data.py                 # Read and validate JSONL annotations
  labels.py               # Supported entity labels
  train.py                # Train and evaluate
  predict.py              # CLI prediction
  dataset_pos.py          # POS-tag product/catalog JSON, JSONL, or CSV fields
  chunking.py             # Parser-based NP chunks and IOB output
  pii.py                  # Common PII detection and redaction
requirements.txt
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins spaCy and the compatible English pipeline wheel. Use the same Python version and dependency file across environments; generated training models and local datasets are not committed.

## Prepare training data

Create `data/train.jsonl`, one JSON object per line. `data/annotations.example.jsonl` is a tiny synthetic format/demo file (not a usable training corpus). Entity offsets are **character offsets into the exact `text` string**, with `start` inclusive and `end` exclusive. Text must already be in its final cleaned form; changing whitespace or punctuation after labeling invalidates offsets.

For a smoke test only, copy `data/annotations.example.jsonl` to `data/train.jsonl` (PowerShell: `Copy-Item data/annotations.example.jsonl data/train.jsonl`; macOS/Linux: `cp data/annotations.example.jsonl data/train.jsonl`). Its tiny size is not enough to assess model quality.

```json
{"text":"The Ryzen 7 7800X3D uses the AM5 socket.","entities":[{"start":4,"end":19,"label":"MODEL"},{"start":29,"end":32,"label":"SOCKET"}]}
{"text":"No display after I reseated the graphics card.","entities":[{"start":0,"end":10,"label":"SYMPTOM"},{"start":19,"end":27,"label":"ACTION"},{"start":32,"end":45,"label":"COMPONENT_TYPE"}]}
```

The text can also include an optional `id` or other metadata; training only reads `text` and `entities`. Empty `entities` is valid for a negative example.

The project does not automatically create gold labels from product catalog fields. If your upstream feature engineering produces product names/categories, use those to help annotate sentences, then review spans and offsets. Include troubleshooting examples from a separately sourced/annotated corpus if those labels are required.

## Optional POS-tag helper for PC-part catalog records

`dataset_pos.py` is an optional helper for the earlier structured PC-part catalog format. It is separate from the Stack Exchange text pipeline and is not part of feature engineering or data collection.

```bash
python -m src.pc_hardware_ner.dataset_pos --input data/json/motherboard.json --category motherboard --output results/motherboard-pos.jsonl
```

Output is one JSONL row per product, with `name`, `category`, and `tokens`; each token includes `text`, Universal POS (`pos`), detailed spaCy tag (`tag`), `lemma`, `field`, and `value_index`.

## Train

Run commands from the repository root:

```bash
python -m src.pc_hardware_ner.train --data data/train.jsonl --output models/pc-hardware-ner --epochs 20
```

Options include `--base-model`, `--epochs`, `--test-size` (default `0.2`), `--seed`, and `--test-data path/to/heldout.jsonl`. Prefer a separate held-out file split by source/thread to reduce leakage; `--test-data` overrides the random split. For small datasets that cannot produce a meaningful split, evaluation is skipped. Review precision/recall/F-score and test on data not used during training before relying on predictions. The saved pipeline contains the POS base model plus the trained NER component.

## Predict

Single text:

```bash
python -m src.pc_hardware_ner.predict --model models/pc-hardware-ner --text "No display after I reseated the RTX 4070 Super."
```

JSONL batch input (each line must contain a `text` string):

```bash
python -m src.pc_hardware_ner.predict --model models/pc-hardware-ner --input data/unseen.jsonl --output predictions.jsonl
```

Each output record contains the original text, entities with character offsets, and POS-tagged tokens. Use `--pretty` for readable single-text output. If no trained model exists yet, inference reports an error rather than silently using an untrained model.

## Chunking

Run parser-based base noun-phrase chunking on text already prepared by the data team. Output includes NP spans with character offsets and per-token `B-NP`/`I-NP`/`O` tags.

```bash
python -m src.pc_hardware_ner.chunking --text "My PC will not boot after I replaced the motherboard." --pretty
python -m src.pc_hardware_ner.chunking --input data/processed.jsonl --output results/chunks.jsonl
```

This is a spaCy dependency-parser baseline, not a separately trained chunking model. It requires a model with a dependency parser.

## PII detection/redaction

Run on user-generated troubleshooting text supplied by the data team. It detects common email addresses, phone numbers, IPv4 addresses, usernames, and spaCy-recognized people/locations, then returns a redacted copy and original span offsets.

```bash
python -m src.pc_hardware_ner.pii --text "Email alex@example.com about the no-display issue." --pretty
python -m src.pc_hardware_ner.pii --input data/processed.jsonl --output results/redacted.jsonl
```

PII detection is a safety baseline, not exhaustive: names/locations can be missed or over-redacted. Review its output before sharing text. It is intended for prose, not hardware specification fields where number patterns can be misidentified.

Punctuation restoration is not included because Stack Exchange question/answer prose is already punctuated. It would need its own punctuation-corrupted inputs and clean reference targets to train/evaluate meaningfully.

## Tests

The unit tests cover prediction, the optional catalog POS helper, noun-phrase chunking, and common PII redaction.

```bash
python -m unittest discover -s tests -v
```

The request test uses fixed `EntityRuler` patterns to make expected extraction deterministic. Tests cover output behavior, not learned model accuracy, exhaustive PII recall, or hardware compatibility.

## CI/CD

GitHub Actions runs compilation and the unit tests on Python 3.10–3.12 for pushes, pull requests, and manual runs. Pushing a version tag such as `v0.1.0` runs the same checks and, if they pass, publishes a source bundle as a GitHub Release. The bundle does not contain local datasets or a trained model; there is no deployment target configured yet.

## Suggested workflow

1. Prepare cleaned text and useful features upstream; keep source records and splits traceable.
2. Agree on the entity definitions and edge-case policy in the table above.
3. Annotate a representative set, including negative examples and both product/build and troubleshooting language.
4. Split by source/thread (not random near-duplicate sentences) where applicable to reduce leakage.
5. Train, inspect errors and per-label metrics, revise guidelines/data, and retrain.
6. Add more labeled troubleshooting examples before judging `SYMPTOM`, `ERROR_CODE`, or `ACTION` quality.

## Data source and attribution

The planned text source is the [Stack Exchange API](https://api.stackexchange.com/), with the focused [Super User hardware query](https://api.stackexchange.com/2.3/questions?site=superuser&tagged=cpu%3Bmotherboard%3Bmemory%3Bpower-supply%3Bboot&pagesize=100&order=desc&sort=creation&filter=withbody). Follow Stack Exchange API terms and the per-post `content_license`; retain source links and attribution with the prepared corpus. Data collection and preparation are handled outside this project.

## Limitations

- No trained project-specific model or usable training corpus is included; the tiny synthetic annotation file is only a format example.
- A pretrained POS tagger may mis-tag hardware model strings, abbreviations, and specs.
- NER quality depends on consistent span annotations and sufficient examples for every desired label.
- Chunking and PII detection are parser/rule-based baselines and should be evaluated on the team's prepared troubleshooting text.
- This is a research/starter pipeline, not a hardware compatibility checker or diagnostic system.

## License

See [LICENSE](LICENSE) for this project's license. Dataset licensing remains with its original authors.
