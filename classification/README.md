# Classification Module

## Overview

This module implements a production-oriented, cost-aware PII classification pipeline for GDPR compliance support.

The system uses a two-stage architecture:

```text
Document
    |
    v
Sweep 1: Presidio + Regex
    |
    v
Strong structured PII?
    |
    |-- Yes --> Flag as PII
    |
    |-- No
          |
          v
    Ambiguous person-related signal?
          |
          |-- No --> Non-PII
          |
          |-- Yes
                |
                v
          Sweep 2: LLM Review
                |
                v
          Final classification
```

The objective is to preserve high recall while reducing unnecessary LLM calls and operational cost.

---

## Project Structure

```text
classification/
├── config.py
├── pii_detector.py
├── llm_reviewer.py
├── pipeline.py
├── evaluation/
├── data/
├── results/
│   └── runs/
└── README.md
```

---

## Production Components

### `pii_detector.py`

Implements the first classification sweep.

#### Responsibilities

- Runs Microsoft Presidio entity detection
- Applies custom regular expressions for structured identifiers
- Fuses Presidio and regex results
- Distinguishes between:
  - Strong PII
  - Potential PII
- Routes ambiguous documents to the LLM review stage

#### Supported Languages

- English (`en`)
- German (`de`)

#### Strong PII Categories

The following categories immediately classify a document as containing PII:

```text
PHONE_NUMBER
CREDIT_CARD
IBAN_CODE
MEDICAL_LICENSE
```

#### Potential PII Categories

These categories require additional context:

```text
PERSON
EMAIL_ADDRESS
```

#### Additional Heuristics

A keyword detector identifies person-related content even when no hard identifier is detected.

Examples:

```text
Name
Vorname
Nachname
Employee
Mitarbeiter
Passport
Reisepass
Address
Adresse
Employee ID
Personalnummer
```

Documents matching these heuristics are routed to Sweep 2.

#### Main Entry Point

```python
run_presidio_regex(df)
```

Input:

```python
pandas.DataFrame
```

Output:

```python
pandas.DataFrame
```

Additional columns include:

```text
detected_categories
strong_pii_categories
potential_pii_categories
detected_any_pii
detected_pii
entities
per_type_conf
needs_llm_review
```

---

### `llm_reviewer.py`

Implements the second classification sweep.

The LLM reviewer is only executed for documents where:

```python
needs_llm_review == True
```

This minimizes token usage and operational cost.

#### Responsibilities

- Extracts relevant text surrounding detected entities
- Builds a GDPR-focused classification prompt
- Routes requests to the configured LLM provider
- Supports multiple LLM providers
- Parses and validates JSON responses
- Adds final LLM classifications

#### Classification Strategy

Only ambiguous documents reach the LLM.

Examples:

- Person names without sufficient context
- Employee references
- Context-dependent identifiers
- False-positive Presidio detections

#### LLM Output Format

```json
{
  "contains_pii": true,
  "reason": "Contains an identifiable person's name and email address."
}
```

#### Main Entry Point

```python
run_llm(df, provider)
```

Adds:

```text
llm_pii
llm_reason
```

to the DataFrame.

#### Supported Providers

The LLM reviewer supports multiple providers through a provider routing layer.

Current providers:

```text
openrouter
qwen
```

Provider selection is passed into:

```python
run_llm(df, provider)
```

which internally routes requests through:

```python
call_llm(prompt, provider)
```

Examples:

```python
run_llm(df, provider="openrouter")
```

```python
run_llm(df, provider="qwen")
```

---

### `pipeline.py`

Production entry point for the complete classification workflow.

Current execution flow:

```text
Load dataset
        |
        v
Run Sweep 1 once
        |
        +--> Save Sweep 1 baseline results
        |
        v
For each configured provider:
        |
        +--> Run Sweep 2
        |
        +--> Compute final prediction
        |
        +--> Add model metadata
        |
        +--> Save provider-specific results
```

#### Pipeline Steps

1. Load the input dataset
2. Run Sweep 1 (Presidio + Regex)
3. Save Sweep 1 baseline output
4. Create a copy of Sweep 1 results for each provider
5. Run Sweep 2 (LLM Review)
6. Compute the final prediction
7. Add provider/model metadata
8. Save provider-specific results

This design avoids rerunning Sweep 1 for every provider and allows direct comparison of LLM performance.

#### Output

Pipeline outputs are written to:

```text
classification/results/
```

---

## Classification Logic

### Sweep 1: Deterministic Detection

Technologies:

- Microsoft Presidio
- SpaCy
- Custom Regex Rules

Purpose:

- Identify high-confidence structured PII
- Avoid unnecessary LLM calls

Examples:

```text
Email addresses
IBAN numbers
Credit card numbers
Medical license identifiers
Person names
```

---

### Sweep 2: LLM Review

Technology:

```text
Multiple LLM providers: OpenRouter, Qwen
```

Current models:

```text
openai/gpt-4o-mini, qwen3.7-plus
```

Purpose:

- Resolve ambiguous cases
- Reduce false positives
- Improve overall classification quality

Only documents flagged during Sweep 1 are reviewed.

---

## Input Dataset

Current dataset location:

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
file_created_date
last_modified_date
```

If no language column is provided, English is used by default.

---

## Running the Pipeline

From the project root:

```bash
source .venv/bin/activate
python -m classification.pipeline
```

The module should always be executed from the repository root to ensure imports resolve correctly.

---

## Output Columns

### Core Classification Columns

```text
detected_pii
detected_any_pii
needs_llm_review

llm_pii
llm_reason

final_pii
predicted_pii
```

### Output Metadata

```text
run_id

provider
model_family
model_name

prediction_source
prediction_stage

pipeline_name
```

### Interpretation

#### `detected_pii`

```text
True
```

High-confidence PII detected during Sweep 1.

#### `detected_any_pii`

```text
True
```

At least one strong or potential PII signal was detected during Sweep 1.

#### `needs_llm_review`

```text
True
```

The document was routed to Sweep 2 for contextual review.

#### `llm_pii`

```text
True
```

The LLM determined the document contains personal data.

#### `llm_reason`

Explanation returned by the LLM.

#### `final_pii`

```text
True
```

Final classification produced by the two-stage pipeline.

#### `predicted_pii`

```text
True
```

Standardized prediction column used for benchmarking, evaluation, and comparison across models.

#### `run_id`

Unique identifier for the pipeline execution.

Example:

```text
20260810_153000
```

#### `provider`

Provider used for classification.

Examples:

```text
qwen
openrouter
local
```

#### `model_family`

Model family associated with the prediction.

Examples:

```text
qwen
gpt
rule_based
```

#### `model_name`

Specific model that produced the prediction.

Examples:

```text
qwen3.7-plus
openai/gpt-4o-mini
presidio_regex_v1
```

#### `prediction_source`

Origin of the prediction.

Examples:

```text
presidio_regex
llm
```

#### `prediction_stage`

Pipeline stage represented by the output.

Examples:

```text
sweep1
final
```

#### `pipeline_name`

Name of the classification pipeline that produced the result.

Example:

```text
two_stage_pii_pipeline
```
---

## Design Decisions

### Why Not Use Only Presidio?

Presidio performs well on structured identifiers but struggles with contextual interpretation.

Example:

```text
John Smith works in HR.
```

The presence of a name does not always imply meaningful personal data.

---

### Why Not Use Only an LLM?

LLMs are:

- More expensive
- Slower
- Less deterministic

Running every document through an LLM would significantly increase operational cost.

---

### Why a Hybrid Approach?

The combination provides:

- High recall
- Lower cost
- Reduced latency
- Better handling of ambiguous cases

Only uncertain documents are escalated to the LLM layer.

---

## Dependencies

Classification-specific dependencies are maintained in:

```text
classification/requirements.txt
```

Install all project dependencies:

```bash
pip install -r requirements.txt
```

Or install only classification dependencies:

```bash
pip install -r classification/requirements.txt
```

---

## Main Entry Points

### First Sweep

```python
run_presidio_regex(df)
```

File:

```text
pii_detector.py
```

### Second Sweep

```python
run_llm(df, provider)
```

File:

```text
llm_reviewer.py
```

### Full Pipeline

Run:

```bash
python -m classification.pipeline
```

File:

```text
pipeline.py
```

## Results

Pipeline outputs are written to:

```text
classification/results/
└── runs/
    └── <run_id>/
        ├── sweep1.csv
        ├── qwen.csv
        ├── openrouter.csv
        └── run_metadata.json
```

Each pipeline execution creates a dedicated run directory containing all generated outputs.

### Sweep 1 Output

The deterministic Presidio + Regex detector is saved as a standalone baseline.

Metadata:

```text
provider          = local
model_family      = rule_based
model_name        = presidio_regex_v1
prediction_source = presidio_regex
prediction_stage  = sweep1
```

The standardized prediction column is:

```text
predicted_pii = detected_pii
```

### Provider Outputs

Each configured provider generates its own prediction file within the same run directory.

Metadata:

```text
run_id
provider
model_family
model_name
prediction_source
prediction_stage
pipeline_name
```

The standardized prediction column is:

```text
predicted_pii = final_pii
```

### Run Metadata

Each run directory contains a `run_metadata.json` file.

Typical fields include:

```text
run_id
pipeline_name
providers

documents_total
documents_sent_to_llm
llm_calls_avoided

routing_rate
local_processing_rate

sweep1_runtime_seconds
sweep2_runtime_seconds
pipeline_runtime_seconds
```

This metadata supports benchmarking, runtime analysis, routing analysis, and future MLflow integration.
---

## Evaluation

The evaluation framework is documented separately.

See:

```text
classification/evaluation/README.md

## Provider Benchmarking

The pipeline supports running multiple LLM providers against the same dataset.

Example configuration:

```python
PROVIDERS_TO_RUN = [
    "openrouter",
    "qwen",
]
```

For each provider:

1. Sweep 1 results are reused.
2. Sweep 2 is executed independently.
3. A separate output file is generated.
4. Results can be evaluated using the evaluation framework.

This enables direct comparison of:

- Accuracy
- Precision
- Recall
- F1 Score
- Cost
- Latency
- False positive rate
- False negative rate

without changing the classification logic.

## Compliance Note

The current implementation uses OpenRouter for experimentation and evaluation.

For production use involving real personal data, the provider should be replaced with a GDPR-compliant alternative such as:

- Azure OpenAI in an EU region
- Mistral API with appropriate contractual safeguards
- Another approved enterprise provider

Do not process real personal data using a provider that has not been validated for compliance requirements.

---

## Future Improvements

Planned enhancements include:

- Structured logging instead of `print()` statements
- Automated unit tests
- Configurable input/output paths
- Improved batch processing
- Better retry and failure handling
- Local model support (BERT, DistilBERT, etc.)
- GDPR-compliant production LLM provider
- Additional LLM providers (Azure OpenAI, Claude, Gemini)Show more lines

---