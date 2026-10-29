# PC Hardware Build & Troubleshooting Entity Extractor

A Python NLP starter project for extracting PC-hardware entities and returning POS tags for every token. It uses **spaCy**: the English pipeline supplies POS tagging, while a trainable NER component is fine-tuned on your annotated, pre-cleaned examples.

> **Data note:** [docyx/pc-part-dataset](https://github.com/docyx/pc-part-dataset) is a catalog of PC parts (JSON, JSONL, and CSV). It is not an annotated NER/POS corpus and does not contain the troubleshooting conversations needed to learn symptoms or fixes. Use it as a source of component vocabulary/features in your upstream data preparation; provide labeled text examples to train this extractor. Scraping and cleaning are intentionally out of scope here.

## What it does

- Predicts token-level POS tags for ordinary text and for product-name/spec values in the catalog. Dataset POS output keeps each token linked to its source feature field (such as `socket`, `speed`, or `form_factor`).
- Predicts custom, span-level NER tags for hardware/build and troubleshooting text.
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

POS tags are produced by spaCy's English model and are **not** custom NER labels. The catalog has structured values rather than POS-annotated sentences, so this project does not claim to train a POS model from it. The dataset POS path applies standard English POS tags to lexical and numeric field values and preserves the feature name as context: for example, `3200` is tagged `NUM`, with `field: "speed"` identifying what that number means. Field names themselves are schema metadata, not POS-tagged sentence text. The included model is English-language; other languages need a matching POS base model.

## Project layout

```text
data/
  annotations.example.jsonl # Tiny synthetic format example
  train.jsonl             # Your annotated examples (local)
models/                   # Generated model output (not included)
src/pc_hardware_ner/
  __init__.py
  data.py                 # Read and validate JSONL annotations
  labels.py               # Supported entity labels
  train.py                # Train and evaluate
  predict.py              # CLI prediction
  dataset_pos.py          # POS-tag product/catalog JSON, JSONL, or CSV fields
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

## POS-tag PC-part catalog records

The PC-part repository separates records by category and provides JSON arrays, JSONL, and CSV exports. `dataset_pos.py` reads these formats, POS-tags each non-empty scalar value (including product names and textual/spec values), and preserves the originating field name and array index on each token. This is the dataset-aware POS path; it does not require NER character-span annotations. Supply the category because category is represented by the source file, not necessarily a field in each record.

```bash
python -m src.pc_hardware_ner.dataset_pos --input data/json/motherboard.json --category motherboard --output results/motherboard-pos.jsonl
```

Output is one JSONL row per product, with `name`, `category`, and `tokens`; each token includes `text`, Universal POS (`pos`), detailed spaCy tag (`tag`), `lemma`, `field`, and `value_index`. For a memory record such as `"speed": [4, 3200]`, the tags stay connected to the `speed` field and retain their array indices. It deliberately does not turn catalog properties into artificial sentences or pretend that the catalog contains gold POS annotations.

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

## Tests

The unit tests exercise both final stages: simple hardware requests are checked for both counterpart entities and offsets, and catalog-shaped product rows are checked for POS output that retains field/index context.

```bash
python -m unittest discover -s tests -v
```

The request test uses fixed `EntityRuler` patterns to make expected extraction deterministic; the dataset POS test uses a record shape from the source dataset. These tests cover output/pipeline behavior, not learned model accuracy or hardware compatibility.

## Suggested workflow

1. Prepare cleaned text and useful features upstream; keep source records and splits traceable.
2. Agree on the entity definitions and edge-case policy in the table above.
3. Annotate a representative set, including negative examples and both product/build and troubleshooting language.
4. Split by source/thread (not random near-duplicate sentences) where applicable to reduce leakage.
5. Train, inspect errors and per-label metrics, revise guidelines/data, and retrain.
6. Add more labeled troubleshooting examples before judging `SYMPTOM`, `ERROR_CODE`, or `ACTION` quality.

## Data source and attribution

The catalog source is [docyx/pc-part-dataset](https://github.com/docyx/pc-part-dataset), which describes its dataset as PCPartPicker parts and provides an MIT license. Review the source repository's current license and upstream terms before redistribution or commercial use. This project does not bundle or download that dataset.

## Limitations

- No trained project-specific model or usable training corpus is included; the tiny synthetic annotation file is only a format example.
- A pretrained POS tagger may mis-tag hardware model strings, abbreviations, and specs. Field context is preserved, but numeric/structured specs are not POS-supervised gold annotations.
- NER quality depends on consistent span annotations and sufficient examples for every desired label.
- Catalog data supports part/product vocabulary, but not troubleshooting intent or symptom extraction by itself.
- This is a research/starter pipeline, not a hardware compatibility checker or diagnostic system.

## License

See [LICENSE](LICENSE) for this project's license. Dataset licensing remains with its original authors.
