# PII Detection

Cost-aware detection of GDPR personal data in documents. The pipeline combines
deterministic detectors, a transformer pre-filter and an LLM review stage, so
that expensive LLM calls are spent only on the documents that actually need
them.

## Pipeline

```text
Document
    ↓
Sweep 1 — deterministic detection (Presidio + spaCy + regex)
    ├─ clear personal data found      → done, no LLM call
    ├─ no signal at all               → done, no LLM call
    └─ ambiguous                      → transformer pre-filter
                                            ├─ confident non-PII → no LLM call
                                            ├─ confident PII     → no LLM call
                                            └─ uncertain         → LLM review
    ↓
LLM review (Sweep 2)
    ↓
Final prediction
```

The pre-filter sits between Sweep 1 and the LLM review and decides which of
Sweep 1's *ambiguous* documents are worth an LLM call. Its success criterion is
not accuracy: it is how far the LLM call volume drops while document-level
recall stays at or above the rule-based baseline of 0.9833.

See `classification/README.md` for the classification pipeline and
`classification/prefilter/README.md` for the pre-filter.

## Setup

```bash
# Create an isolated Python environment for the project
python -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install backend and classification dependencies
pip install -r requirements.txt

# Install the project locally in editable mode
pip install -e .
```

The pre-filter additionally needs the deep-learning stack, which
`requirements.txt` does not cover:

```bash
pip install torch transformers scikit-learn matplotlib mlflow
```

### Pretrained weights offline

`huggingface.co` is blocked in this execution environment, so
`from_pretrained("distilbert-base-uncased")` fails. Fetch the weights into the
git-ignored `models/` directory instead:

```bash
python -m classification.prefilter.fetch_model
```

Anyone with normal HuggingFace access can skip this step.

## Entry points

`pip install -e .` installs these console scripts:

| command | what it does |
|---|---|
| `classify` | runs the classification pipeline (`classification.pipeline`) |
| `evaluate` | runs the evaluation pipeline (`classification.evaluation.evaluate_pipeline`) |
| `update-dataset` | updates the dataset schema (`classification.data.update_dataset_schema`) |

The classification pipeline can also be run as a module:

```bash
python -m classification.pipeline
```

The pre-filter has its own module entry points:

| command | what it does |
|---|---|
| `python -m classification.prefilter.fetch_model` | downloads the pretrained encoder into `models/` |
| `python -m classification.prefilter.eda` | dataset report and split sanity check |
| `python -m classification.prefilter.train` | trains the model and calibrates the routing thresholds |
| `python -m classification.prefilter.predict` | writes evaluation-compatible predictions |
| `python -m classification.prefilter.error_report` | error and routing-cost slices |
| `python -m pytest classification/prefilter/tests/` | routing-logic tests, no model required |

`predict` writes into `classification/results/runs/<run_id>/`, the same place
the rest of the pipeline writes to, so its output can be scored with
`evaluate --run-id <run_id>`.

## Pre-filter status

The model is a DistilBERT encoder shared by two heads: a binary personal-data
head and a 12-label GDPR entity head.

On the current 500-row pilot dataset it reaches accuracy, precision, recall and
F1 of 1.0000 on validation and test and routes 0.0% of documents to the LLM, at
63.9 ms inference per document. **These numbers should not be quoted without
their caveat.** The pilot is close to linearly separable — negatives score
0.00–0.05, positives 0.93–1.00, with an empty band in between — so there is no
uncertain zone left to route and the 0.0% is a degenerate outcome, not a
validated capability. The routing rate only becomes a meaningful figure on the
larger 1,400-row dataset.

The full picture, including the entity-head numbers and the open findings about
the evaluation module and the dataset splits, is in
`classification/prefilter/README.md`, section "Results on the 500-row pilot".
