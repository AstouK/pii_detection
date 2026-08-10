# Evaluation Framework

## Overview

This module evaluates saved outputs from the GDPR PII classification pipeline.

The evaluation framework supports:

1. Document-level metrics
2. Per-entity-type metrics
3. Error analysis (TP/TN/FP/FN)
4. Provider benchmarking
5. MLflow experiment tracking

The framework operates on previously generated classification outputs and does not rerun the production classification pipeline.

---

## Evaluation Architecture

```text
Classification Run
        |
        v
Load Prediction Outputs
        |
        +--> sweep1.csv
        |
        +--> qwen.csv
        |
        +--> openrouter.csv
        |
        v
Metric Computation
        |
        v
Error Analysis
        |
        v
Benchmark Summary
        |
        v
Optional MLflow Logging
```

---

## Files

```text
evaluation/
├── benchmarking.py
├── config.py
├── error_analysis.py
├── evaluate_pipeline.py
├── io.py
├── metrics.py
├── mlflow_logger.py
├── reporting.py
├── README.md
└── results/
```

---
## Multi-Provider Evaluation

The evaluation framework compares prediction outputs generated from the same classification run.

Supported output types include:

```text
sweep1
openrouter
qwen
future local models (BERT, DistilBERT, etc.)
```

Each output is evaluated independently using the same ground truth.

This enables direct comparison of:

- Accuracy
- Precision
- Recall
- F1 Score
- False Positive Rate
- False Negative Rate

without rerunning the classification pipeline.

---

## Evaluation Dataset

Evaluation is performed on prediction outputs generated from:

```text
classification/results/runs/<run_id>/
```

Ground truth is expected to be included in those outputs.

Primary document-level label:

```text
ground_truth_pii
```

Fallback label:

```text
contains_personal_data
```

Additional entity-level labels follow the convention:

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

Ground-truth labels are normalized automatically during metric computation and error analysis.

Supported positive values:

```text
yes
true
1
y
ja
```

Supported negative values:

```text
no
false
0
n
nein
```

The evaluation framework automatically resolves:

```text
ground_truth_pii
```

or

```text
contains_personal_data
```

when computing metrics and error analysis.

---

## Evaluated Entity Types

Entity-level evaluation is automatically derived from columns matching:

```text
<ENTITY_TYPE>_yes_no
```

Examples:

```text
PERSON_yes_no
EMAIL_ADDRESS_yes_no
PHONE_NUMBER_yes_no
IBAN_CODE_yes_no
```

No static entity list is required.

---

## Evaluation Levels

### Standardized Prediction Evaluation

Primary evaluation uses:

```python
predicted_pii
```

This standardized prediction column allows evaluation of:

- Sweep 1 outputs
- LLM provider outputs
- Future local model outputs

using the same evaluation workflow.

### Additional Stage Evaluation

When available, the framework can also evaluate:

```python
detected_pii
detected_any_pii
llm_pii
final_pii
```

against:

```python
ground_truth_pii
```

These stage-specific metrics are generated automatically when the corresponding columns exist.

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

Evaluate the latest classification run:

```bash
evaluate
```

Evaluate a specific classification run:

```bash
evaluate --run-id 20260810_153000
```

Evaluate a different prediction column:

```bash
evaluate --prediction-col final_pii
```

Log results to MLflow:

```bash
evaluate --log-mlflow
```

The evaluation pipeline:

1. Loads a classification run
2. Loads all prediction outputs
3. Computes metrics
4. Runs error analysis
5. Generates benchmark summaries
6. Saves evaluation artifacts
7. Optionally logs results to MLflow

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

Evaluation outputs are written to:

```text
classification/evaluation/results/
└── runs/
    └── <classification_run_id>/
```

Each evaluated output receives its own directory.

Example:

```text
classification/evaluation/results/
└── runs/
    └── 20260810_153000/
        ├── sweep1/
        │   ├── metrics.csv
        │   ├── predictions_with_error_labels.csv
        │   ├── false_positives.csv
        │   ├── false_negatives.csv
        │   └── error_summary.csv
        ├── qwen/
        │   ├── metrics.csv
        │   ├── predictions_with_error_labels.csv
        │   ├── false_positives.csv
        │   ├── false_negatives.csv
        │   └── error_summary.csv
        ├── benchmark_summary.csv
        └── evaluation_metadata.json
```

### Metrics Output

Metrics are stored in:

```text
metrics.csv
```

and include:

```text
accuracy
precision
recall
f1
TP
TN
FP
FN
```

### Error Analysis Output

Error analysis artifacts include:

```text
predictions_with_error_labels.csv
false_positives.csv
false_negatives.csv
true_positives.csv
true_negatives.csv
error_summary.csv
```

### Benchmark Summary

```text
benchmark_summary.csv
```

contains comparable metrics across all evaluated outputs from the same classification run.

---

## Design Goals

The evaluation framework was designed to:

- Evaluate saved classification outputs independently of production execution
- Measure document-level and entity-level performance
- Identify sources of false positives and false negatives
- Compare providers and future local models using a common schema
- Generate reproducible evaluation artifacts
- Support MLflow experiment tracking
- Enable future prompt, cost, and latency benchmarking

---

## Future Improvements

Planned enhancements include:

## Future Improvements

Planned enhancements include:

- Prompt benchmarking
- Cost-per-document benchmarking
- Latency benchmarking
- Routing analysis
- ROC and Precision-Recall curves
- Confidence threshold experiments
- Statistical significance testing
- Automated benchmark reports
- Model comparison dashboards
- Entity extraction evaluation at span level