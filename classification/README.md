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
ollama
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
Local Ollama Models
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

## Local Ollama

`rule_plus_ollama` is included in the default `classify` run. The default
strategy remains `rule_plus_qwen`.

1. Install and start Ollama (`ollama serve`).
2. Pull the configured model:

```bash
ollama pull qwen2.5
```

3. Set values in `.env` (see `.env.example`):

```text
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5
```

4. Run the pipeline. A default `classify` run executes `rule_based`,
   `rule_plus_qwen`, `rule_plus_gpt4o_mini`, and `rule_plus_ollama`.
   To run only the local strategy:

```bash
classify --strategies rule_plus_ollama
```

If Ollama is not running, the default `classify` run fails when
`rule_plus_ollama` starts, with a message to run `ollama serve` and
`ollama pull qwen2.5`.

---

## Running the Pipeline

```bash
source .venv/bin/activate

python -m classification.pipeline
```

Or:

```bash
classify
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