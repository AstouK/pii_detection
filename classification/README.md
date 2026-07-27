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
├── pii_detector.py
├── llm_reviewer.py
├── pipeline.py
├── evaluation/
├── data/
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
- Sends requests to OpenRouter
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
run_llm(df)
```

Adds:

```text
llm_pii
llm_reason
```

to the DataFrame.

---

### `pipeline.py`

Production entry point for the complete classification workflow.

Current execution flow:

```python
df = pd.read_excel(
    "classification/data/pii_dataset.xlsx",
    parse_dates=["file_created_date", "last_modified_date"],
)

df = run_presidio_regex(df)
df = run_llm(df)

df.to_csv("final_output.csv", index=False)
```

#### Pipeline Steps

1. Load input dataset
2. Run Sweep 1 (Presidio + Regex)
3. Run Sweep 2 (LLM Review)
4. Export results

#### Output

```text
final_output.csv
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
OpenRouter
```

Current model:

```text
openai/gpt-4o-mini
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
classification/data/pii_dataset.xlsx
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

Do not run:

```bash
cd classification
python pipeline.py
```

The module should always be executed from the repository root to ensure imports resolve correctly.

---

## Output Columns

Important generated columns include:

```text
detected_pii
detected_any_pii
needs_llm_review
llm_pii
llm_reason
```

### Interpretation

#### `detected_pii`

```text
True
```

Strong PII was identified during Sweep 1.

#### `needs_llm_review`

```text
True
```

Document is ambiguous and requires Sweep 2.

#### `llm_pii`

```text
True
```

The LLM determined the document contains personal data.

#### `llm_reason`

Contains the explanation returned by the model.

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
- Provider abstraction layer
- GDPR-compliant production LLM provider

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
run_llm(df)
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
## Evaluation

The evaluation framework is documented separately.

See:

```text
classification/evaluation/README.md