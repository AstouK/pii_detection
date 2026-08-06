# Classification Dataset

## Overview

This folder contains datasets used for training, evaluation, benchmarking, synthetic data generation, and experimentation within the GDPR PII Detection project.

The dataset serves as the central source of truth for:

- Deterministic detection evaluation (Presidio + Regex)
- LLM evaluation and benchmarking
- Prompt engineering experiments
- Synthetic data generation
- Future machine-learning model training
- Feedback collection and continuous improvement

Current dataset:

```text
pii_dataset.csv
```

---

## Dataset Purpose

The dataset is used to:

- Evaluate Sweep 1 (Presidio + Regex)
- Evaluate Sweep 2 (LLM Review)
- Benchmark multiple LLM providers
- Compare prompt versions
- Support synthetic data generation
- Support future BERT-style supervised training
- Support feedback-based continuous improvement
- Provide reproducible evaluation benchmarks

---

## Dataset Structure

Each row represents a single document or document excerpt.

---

## Document Metadata

### Document Identification

| Column | Description |
|----------|----------|
| `document_id` | Unique document identifier. |
| `file_name` | Original file name. |
| `document_type` | High-level business document category. |
| `scenario_type` | Classification scenario used for benchmarking and synthetic-data generation. |

Example values:

```text
expense_report
employee_record
contract
invoice
customer_support
medical_record
passport_record
general_document
```

---

### Language

| Column | Description |
|----------|----------|
| `language` | Language of the document. |

Supported values:

```text
en
de
```

---

### Source Information

| Column | Description |
|----------|----------|
| `source_system` | Originating system or repository. |
| `responsible_owner` | Document owner. |
| `owner_email` | Owner contact information. |
| `data_source` | Dataset origin. |

Typical values:

```text
manual
synthetic
feedback
```

---

### File Metadata

| Column | Description |
|----------|----------|
| `file_created_date` | Original file creation date. |
| `last_modified_date` | Last modification date. |
| `dataset_created_at` | Dataset record creation date. |
| `file_size_mb` | File size in megabytes. |

---

## Core Text

| Column | Description |
|----------|----------|
| `full_text` | Full document text used for classification. |

The raw text should be preserved as-is, including line breaks, to maintain realistic document structure.

---

## Document-Level Ground Truth

| Column | Description |
|----------|----------|
| `contains_personal_data` | Ground-truth GDPR classification label. |

Allowed values:

```text
yes
no
```

During evaluation, these values are normalized to boolean labels.

---

## Entity-Level Ground Truth

The following columns indicate whether the corresponding entity type appears within the document.

```text
PERSON_yes_no
EMAIL_ADDRESS_yes_no
PHONE_NUMBER_yes_no
LOCATION_yes_no
IBAN_CODE_yes_no
CREDIT_CARD_yes_no
PASSPORT_yes_no
NRP_yes_no
DATE_TIME_yes_no
IP_ADDRESS_yes_no
URL_yes_no
MEDICAL_LICENSE_yes_no
```

Allowed values:

```text
yes
no
```

Example:

```text
EMAIL_ADDRESS_yes_no = yes
```

indicates the document contains at least one email address.

---

## Category Metadata

| Column | Description |
|----------|----------|
| `personal_data_categories` | Semicolon-separated list of detected categories. |
| `primary_pii_type` | Main personal-data category associated with the document. |
| `category_count` | Number of distinct personal-data categories. |
| `pii_count` | Total number of PII entity categories detected in the document. |

Example:

```text
personal_data_categories =
first_name; last_name; email_address; phone_number
```

```text
primary_pii_type = MULTIPLE
```

```text
category_count = 4
```

---

## Difficulty and Benchmark Metadata

| Column | Description |
|----------|----------|
| `difficulty` | Labeling difficulty assessment. |
| `edge_case` | Indicates whether the sample is a challenging example. |
| `challenge_category` | Category of challenge or failure mode. |
| `recommended_split` | Recommended dataset split. |

### Difficulty Levels

```text
easy
medium
hard
```

Examples:

```text
easy:
    Name + email address

medium:
    Employee ID with context

hard:
    Invoice ID versus passport number
```

### Challenge Categories

Examples:

```text
none
invoice_number
vat_number
employee_id
passport_context
medical_context
other_edge_case
```

These categories support failure analysis and synthetic-data generation.

---

## Compliance Metadata

| Column | Description |
|----------|----------|
| `retention_period_exceeded_3y` | Indicates whether the document exceeds the configured retention threshold. |

---

## Synthetic Data Metadata

Synthetic-data support is built directly into the dataset schema.

| Column | Description |
|----------|----------|
| `synthetic` | Indicates whether the record is synthetic. |
| `synthetic_generator` | Model or tool used to generate the sample. |
| `generation_prompt_version` | Synthetic-data generation prompt version. |

Example:

```text
synthetic = yes
synthetic_generator = qwen3.7-plus
generation_prompt_version = v2
```

---

## Feedback Loop Metadata

These fields support future human review, retraining, and continuous improvement workflows.

| Column | Description |
|----------|----------|
| `human_reviewed` | Indicates whether a human reviewed the prediction. |
| `review_status` | Review outcome. |
| `review_notes` | Reviewer comments. |

Example:

```text
review_status = corrected
```

---

## Error Analysis Metadata

| Column | Description |
|----------|----------|
| `error_type` | Classification outcome type generated during evaluation. |

Possible values:

```text
false_positive
false_negative
true_positive
true_negative
```

These values are intended to be populated by evaluation pipelines.

---

## Experiment Metadata

These columns support prompt benchmarking and provider comparison.

| Column | Description |
|----------|----------|
| `prompt_version` | Prompt version used during evaluation. |
| `model_family` | Model family (e.g. GPT, Qwen, BERT). |
| `model_name` | Specific model name. |
| `dataset_version` | Dataset release version. |

Examples:

```text
prompt_version = v3

model_family = qwen

model_name = qwen3.7-plus

dataset_version = v1
```

---

## Annotation Metadata

| Column | Description |
|----------|----------|
| `labeling_notes` | Notes explaining labeling decisions. |

Example:

```text
Straightforward labeled personal-data example.
```

---

## Dataset Lifecycle

Current workflow:

```text
Manual Data
        ↓
Evaluation
        ↓
Benchmarking
        ↓
Error Analysis
        ↓
Prompt Optimization
        ↓
Synthetic Data Generation
        ↓
Model Training
        ↓
Feedback Collection
        ↓
Continuous Improvement
```

---

## Dataset Versioning

The dataset should evolve through explicit versions:

```text
v1
v2
v3
...
```

Major schema modifications, synthetic-data additions, and feedback-driven updates should increment the dataset version.

---

## Future Improvements

Planned enhancements include:

- Larger multilingual datasets
- Dedicated challenge datasets
- Automated error-set generation
- Dataset version tracking
- Human review workflows
- Feedback-driven retraining support
- Additional benchmarking metadata
- Parquet-based storage for large-scale datasets

---

## Storage Format

Current primary format:

```text
CSV
```

CSV is easy to inspect, version, and share.

For larger datasets and training pipelines, migration to:

```text
Parquet
```

is recommended due to:

- smaller file size
- faster loading
- better datatype preservation
- improved scalability