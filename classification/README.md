# Classification Module

## Overview

This module implements a cost-aware GDPR personal data classification pipeline using a hybrid architecture.

```text
Document
    ↓
Sweep 1 (Deterministic Detection)
    ↓
Route Decision
    ↓
Pre-Filter (Transformer, ambiguous documents only)
    ↓
Sweep 2 (LLM Review)
    ↓
Final Prediction
```

The objective is to maximize detection quality while minimizing unnecessary LLM calls.

---

## Architecture

```text
classification/
├── pipeline.py
├── sweep1.py
├── config.py
│
├── detectors/
│   ├── regex_detector.py
│   ├── presidio_detector.py
│   └── evidence_fusion.py
│
├── prefilter/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── thresholds.py
│   ├── train.py
│   ├── predict.py
│   ├── eda.py
│   ├── error_report.py
│   ├── fetch_model.py
│   ├── mlflow_utils.py
│   └── tests/
│
├── review/
│   ├── routing.py
│   └── llm_reviewer.py
│
├── infrastructure/
│   ├── io.py
│   ├── metadata.py
│   ├── outputs.py
│   ├── runtime.py
│   └── provider_runner.py
│
├── evaluation/
├── data/
├── data_generation/
├── prompts/
└── results/
```

---

## Sweep 1: Deterministic Detection

Implemented in:

```text
sweep1.py
```

Components:

```text
regex_detector.py
presidio_detector.py
evidence_fusion.py
routing.py
```

Responsibilities:

- Detect structured identifiers using regex
- Detect entities using Microsoft Presidio
- Fuse detector evidence
- Categorize evidence as strong or contextual
- Decide whether LLM review is required

Outputs:

```text
detected_categories
strong_pii_categories
potential_pii_categories
detected_any_pii
detected_pii
entities
per_type_conf
route
routing_reason
needs_llm_review
```

---

## Pre-Filter: Transformer Routing

Implemented in:

```text
prefilter/
```

Sits between Sweep 1 and the LLM review and decides which of Sweep 1's
ambiguous documents actually need an LLM call.

Model:

```text
DistilBERT, one shared encoder
├── binary head       (personal data yes/no)
└── multi-label head  (12 GDPR entity labels)
```

Responsibilities:

- Score ambiguous documents with a local model
- Calibrate a low and a high routing threshold
- Auto-decide confident documents, escalate the uncertain ones
- Write evaluation-compatible predictions

Success criterion:

```text
Not accuracy.
How far LLM call volume drops
while document-level recall stays at or above
the rule-based baseline of 0.9833.
```

Entry points:

```bash
python -m classification.prefilter.fetch_model
python -m classification.prefilter.eda
python -m classification.prefilter.train
python -m classification.prefilter.predict
python -m classification.prefilter.error_report

python -m pytest classification/prefilter/tests/
```

Outputs:

```text
pii_probability
routed_to_llm
routing_zone
t_low
t_high
per_type_conf
<ENTITY>_predicted
inference_ms
```

Status on the 500-row pilot: accuracy, precision, recall and F1 of 1.0000 on
validation and test, 0.0% of documents routed to the LLM, 63.9 ms inference per
document. The pilot is close to linearly separable (negatives 0.00–0.05,
positives 0.93–1.00, nothing in between), so the 0.0% is a degenerate outcome
and not a validated capability — the figure only becomes meaningful on the
larger 1,400-row dataset.

Details, caveats and the open findings about the splits and the evaluation
module:

```text
classification/prefilter/README.md
```

---

## Sweep 2: LLM Review

Implemented in:

```text
review/llm_reviewer.py
```

Responsibilities:

- Review ambiguous documents
- Build provider-specific prompts
- Execute LLM requests
- Parse structured responses
- Produce contextual PII decisions

Current providers:

```text
openrouter
qwen
```

Entry point:

```python
run_llm(df, provider)
```

Outputs:

```text
llm_pii
llm_reason
```

---

## Pipeline

Entry point:

```bash
python -m classification.pipeline
```

Execution flow:

```text
Load Dataset
      ↓
Run Sweep 1
      ↓
Save Sweep 1 Baseline
      ↓
For Each Provider
      ↓
Run LLM Review
      ↓
Compute Final Prediction
      ↓
Save Provider Output
      ↓
Save Run Metadata
```

---

## Input Dataset

Dataset location:

```text
classification/data/pii_dataset.csv
```

Required column:

```text
full_text
```

Optional columns:

```text
language
```

Supported languages:

```text
en
de
```

---

## Results

Outputs are stored in:

```text
classification/results/
└── runs/
    └── <run_id>/
```

Example:

```text
results/runs/20260810_153000/
├── sweep1.csv
├── qwen.csv
├── openrouter.csv
└── run_metadata.json
```

---

## Metadata

Every prediction output contains:

```text
run_id
provider
model_family
model_name
prediction_source
prediction_stage
pipeline_name
predicted_pii
```

These fields support benchmarking across:

```text
Rule-Based Baseline
OpenRouter Models
Qwen Models
Future BERT Models
Future Hybrid Strategies
```

---

## Evaluation

Evaluation is implemented separately.

See:

```text
classification/evaluation/README.md
```

Current evaluation includes:

- Accuracy
- Precision
- Recall
- F1
- Confusion Matrix
- Error Analysis
- Provider Benchmarking
- Runtime Metrics
- Routing Metrics
- MLflow Tracking

To be added: - Prompt Benchmarking
---

## Design Principle

The classification module answers:

```text
What prediction should we make?
```

The evaluation module answers:

```text
How good was that prediction?
```

Prediction and evaluation are intentionally separated to support reproducible benchmarking and future model comparison.

---

## Running the Pipeline

```bash
source .venv/bin/activate

python -m classification.pipeline
```

---

## Dependencies

Install all project dependencies:

```bash
pip install -r requirements.txt
```

Or classification-specific dependencies only:

```bash
pip install -r 