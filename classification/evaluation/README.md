# Evaluation Framework

## Overview

This module evaluates the performance of the two-stage GDPR PII detection pipeline.

The evaluation framework measures classification quality at multiple levels:

1. Document-level classification
2. Sweep 1 performance
3. Sweep 2 performance
4. End-to-end pipeline performance
5. Per-entity-type detection performance

The purpose is to quantify:

- Accuracy
- Precision
- Recall
- F1 Score

and identify strengths and weaknesses of both detection stages.

---

## Evaluation Architecture

```text
Labeled Dataset
        |
        v
Ground Truth Normalization
        |
        v
Sweep 1: Presidio + Regex
        |
        v
+-----------------------+
| Provider Evaluation   |
+-----------------------+
        |
        +--> OpenRouter
        |
        +--> Qwen
        |
        v
Final Classification
        |
        v
Metric Computation
```

---

## Files

```text
evaluation/
├── evaluate_detector.py
├── evaluate_pipeline.py
├── README.md
└── results/
```

---
## Multi-Provider Evaluation

The evaluation pipeline supports benchmarking multiple LLM providers against the same dataset.

Current supported providers:

```text
openrouter
qwen
```

The workflow is:

1. Run Sweep 1 once.
2. Create a copy of the Sweep 1 results.
3. Execute Sweep 2 using a specific provider.
4. Compute final predictions.
5. Generate metrics.
6. Save provider-specific outputs.

This design ensures that all providers are evaluated on identical Sweep 1 inputs, allowing direct comparison of:

- Accuracy
- Precision
- Recall
- F1 Score
- False Positive Rate
- False Negative Rate

---

## Evaluation Dataset

The evaluation pipeline uses a manually labeled dataset containing:

```text
classification/data/pii_dataset.xlsx
```

Each document contains:

- Document text
- Ground-truth document label
- Ground-truth entity labels

The primary document-level label is:

```text
contains_personal_data
```

Additional ground-truth entity labels follow the convention:

```text
<ENTITY_TYPE>_yes_no
```

Examples:

```text
PERSON_yes_no
EMAIL_ADDRESS_yes_no
PHONE_NUMBER_yes_no
IBAN_CODE_yes_no
CREDIT_CARD_yes_no
MEDICAL_LICENSE_yes_no
```

---

## Ground Truth Normalization

The evaluation framework supports multiple label formats.

Examples:

```text
yes
true
1
y
ja
```

are all normalized to:

```python
True
```

Missing values are treated as:

```python
False
```

Normalization is performed using:

```python
normalise_ground_truth(df)
```

---

## Evaluated Entity Types

The current evaluation covers:

```text
PERSON
EMAIL_ADDRESS
PHONE_NUMBER
IBAN_CODE
CREDIT_CARD
PASSPORT
MEDICAL_LICENSE
```

These can be extended by modifying:

```python
ENTITY_TYPES
```

inside:

```python
evaluate_detector.py
```

---

## Evaluation Levels

### 1. Sweep 1 – Strong PII

Evaluates:

```python
detected_pii
```

against:

```python
ground_truth_pii
```

This measures the performance of high-confidence deterministic detection.

Examples:

- Credit cards
- Phone numbers
- IBANs
- Medical license identifiers

---

### 2. Sweep 1 – Any PII

Evaluates:

```python
detected_any_pii
```

against:

```python
ground_truth_pii
```

This includes both:

- Strong PII
- Potential PII

and measures the sensitivity of the first sweep.

---

### 3. Sweep 2 – LLM Review

Evaluates:

```python
llm_pii
```

against:

```python
ground_truth_pii
```

Only documents routed to:

```python
needs_llm_review == True
```

are included.

This measures the quality of the LLM decision layer independently from Sweep 1.

---

### 4. Final Pipeline Performance

Evaluates:

```python
final_pii
```

against:

```python
ground_truth_pii
```

Current aggregation logic:

```python
final_pii = detected_pii | llm_pii
```

This represents the performance of the complete production pipeline.

---

### 5. Per-Entity-Type Evaluation

Each entity type is evaluated individually.

Example:

```text
EMAIL_ADDRESS
```

Ground truth:

```text
EMAIL_ADDRESS_yes_no
```

Prediction:

```python
EMAIL_ADDRESS in per_type_conf
```

This helps identify which entity categories are performing well and which require further tuning.

---

## Metrics

The framework computes:

### Accuracy

Measures overall correctness.

```text
(TP + TN) / Total
```

---

### Precision

Measures prediction quality.

```text
TP / (TP + FP)
```

A high precision score indicates few false positives.

---

### Recall

Measures detection sensitivity.

```text
TP / (TP + FN)
```

A high recall score indicates few false negatives.

---

### F1 Score

Harmonic mean of precision and recall.

```text
2 * Precision * Recall
-----------------------
 Precision + Recall
```

F1 is the primary metric used to compare different configurations.

---

## Confusion Matrix

For every evaluation stage, the framework computes:

```text
TP = True Positives
TN = True Negatives
FP = False Positives
FN = False Negatives
```

These values are used to derive all performance metrics.

---

## Running the Evaluation

From the project root:

```bash
source .venv/bin/activate
python -m classification.evaluation.evaluate_pipeline
```

The evaluation pipeline:

1. Loads the labeled dataset
2. Normalizes ground-truth columns
3. Executes Sweep 1 once
4. Creates provider-specific copies of the Sweep 1 results
5. Executes Sweep 2 for each configured provider
6. Computes final predictions
7. Calculates performance metrics
8. Exports provider-specific outputs
9. Prints benchmark results

---

## Example Output

Document-level metrics:

```text
Sweep 1 — Strong PII

Accuracy : 0.9500
Precision: 0.9231
Recall   : 0.8750
F1 Score : 0.8984
```

Per-entity metrics:

```text
Entity Type         Acc   Prec   Rec    F1
EMAIL_ADDRESS     0.980  0.960  0.990  0.975
PHONE_NUMBER      0.995  1.000  0.980  0.990
```

---
## Results Export

Evaluation outputs are stored under:

```text
classification/evaluation/results/
```

Provider-specific results are written to separate folders.

Example:

```text
classification/evaluation/results/
├── openrouter/
│   ├── openai_gpt-4o-mini_20260728_153000_predictions.csv
│   └── openai_gpt-4o-mini_20260728_153000_metrics.xlsx
│
└── qwen/
    ├── qwen3.7-plus_20260728_153000_predictions.csv
    └── qwen3.7-plus_20260728_153000_metrics.xlsx
```

Metric dictionaries can be converted to tabular format using:

```python
metrics_to_dataframe(metrics)
```

and exported for:

- experiment tracking
- benchmark comparisons
- model evaluation
- reporting
- thesis documentation

---

## Design Goals

The evaluation framework was designed to:

- Measure the effectiveness of each pipeline stage separately
- Detect sources of false positives and false negatives
- Quantify the impact of LLM review
- Compare multiple LLM providers on identical inputs
- Support future model and rule-set comparisons
- Provide reproducible evaluation results

---

## Future Improvements

Planned enhancements include:

- ROC and Precision-Recall curves
- Confidence threshold experiments
- Error category analysis
- Automatic benchmark reporting
- Multiple dataset support
- Entity extraction evaluation at span level
- Evaluation result dashboards
- Automated provider comparison reports
- Cost-per-document benchmarking
- Latency benchmarking
- Statistical significance testing between models