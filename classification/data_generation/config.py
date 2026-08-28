"""Configuration for synthetic dataset planning and schema validation."""
 
 
SUPPORTED_LANGUAGES = {
    "en",
    "de",
}
 
SUPPORTED_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}
 
SUPPORTED_ENTITY_TYPES = {
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "IBAN_CODE",
    "CREDIT_CARD",
    "PASSPORT",
    "NRP",
    "DATE_TIME",
    "IP_ADDRESS",
    "URL",
    "MEDICAL_LICENSE",
}
 
 
# Scenarios based on existing source document families.
SOURCE_BACKED_SCENARIOS = {
    "expense_report",
    "it_access_request",
    "incident_report",
    "supplier_onboarding",
    "training_evaluation",
}
 
 
# Additional scenarios added for broader enterprise-document coverage.
DESIGNED_SCENARIOS = {
    "employee_record",
    "invoice",
    "contract",
    "customer_support",
    "medical_record",
    "passport_record",
    "general_document",
    "internal_email",
    "meeting_notes",
}
 
 
SUPPORTED_SCENARIOS = (
    SOURCE_BACKED_SCENARIOS
    | DESIGNED_SCENARIOS
)
 
 
SCENARIO_ENTITY_COMBINATIONS = {
    "expense_report": [
        ("PERSON",),
        ("PERSON", "EMAIL_ADDRESS"),
        ("PERSON", "IBAN_CODE"),
        ("PERSON", "PHONE_NUMBER"),
        ("PERSON", "CREDIT_CARD"),
    ],
    "it_access_request": [
        ("PERSON",),
        ("PERSON", "EMAIL_ADDRESS"),
        ("PERSON", "IP_ADDRESS"),
        ("PERSON", "URL"),
    ],
    "incident_report": [
        ("PERSON",),
        ("LOCATION", "DATE_TIME"),
        ("PERSON", "LOCATION"),
        ("PERSON", "DATE_TIME"),
        ("PERSON", "IP_ADDRESS"),
    ],
    "supplier_onboarding": [
        ("PERSON", "EMAIL_ADDRESS"),
        ("PERSON", "PHONE_NUMBER"),
        ("LOCATION",),
        ("PERSON", "IBAN_CODE"),
        ("PERSON", "PASSPORT"),
        ("PERSON", "NRP"),
        ("PERSON", "MEDICAL_LICENSE"),
    ],
    "training_evaluation": [
        ("PERSON",),
        ("PERSON", "EMAIL_ADDRESS"),
        ("PERSON", "DATE_TIME"),
        ("PERSON", "URL"),
    ],
    "employee_record": [
        ("PERSON",),
        ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"),
        ("PERSON", "LOCATION"),
        ("PERSON", "NRP"),
        ("PERSON", "DATE_TIME"),
        ("PERSON", "MEDICAL_LICENSE"),
        ("PERSON", "PASSPORT"),
    ],
    "invoice": [
        ("PERSON",),
        ("PERSON", "EMAIL_ADDRESS"),
        ("PERSON", "IBAN_CODE"),
        ("PERSON", "CREDIT_CARD"),
        ("LOCATION",),
    ],
    "contract": [
        ("PERSON",),
        ("PERSON", "EMAIL_ADDRESS"),
        ("PERSON", "LOCATION"),
        ("PERSON", "DATE_TIME"),
        ("PERSON", "IBAN_CODE"),
    ],
    "customer_support": [
        ("PERSON",),
        ("EMAIL_ADDRESS",),
        ("PHONE_NUMBER",),
        ("IP_ADDRESS",),
        ("URL",),
        ("PERSON", "EMAIL_ADDRESS"),
        ("PERSON", "CREDIT_CARD"),
        ("PERSON", "NRP"),
    ],
    "medical_record": [
        ("PERSON",),
        ("PERSON", "DATE_TIME"),
        ("PERSON", "MEDICAL_LICENSE"),
        ("PERSON", "LOCATION"),
    ],
    "passport_record": [
        ("PERSON", "PASSPORT"),
        ("PERSON", "PASSPORT", "NRP"),
        ("PASSPORT",),
        ("PERSON", "DATE_TIME"),
    ],
    "general_document": [
        ("PERSON",),
        ("EMAIL_ADDRESS",),
        ("PHONE_NUMBER",),
        ("LOCATION",),
        ("DATE_TIME",),
        ("URL",),
        ("PERSON", "IP_ADDRESS"),
    ],
    "internal_email": [
        ("PERSON",),
        ("EMAIL_ADDRESS",),
        ("PERSON", "EMAIL_ADDRESS"),
        ("PHONE_NUMBER",),
        ("URL",),
        ("PERSON", "IP_ADDRESS"),
        ("PERSON", "CREDIT_CARD"),
    ],
    "meeting_notes": [
        ("PERSON",),
        ("PERSON", "DATE_TIME"),
        ("PERSON", "EMAIL_ADDRESS"),
        ("LOCATION", "DATE_TIME"),
    ],
}
 
 
SUPPORTED_SPLITS = {
    "train",
    "validation",
    "test",
}
 
 
SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
 
 
TARGET_POSITIVE_RATE = 0.12
 
 
# Minimum number of examples each entity type must have in EVERY split
# (train/validation/test), counted across the whole dataset, not per
# scenario. A scenario with 6 configured combinations can't guarantee
# 6 distinct entities in a 2-document validation slice on its own --
# this floor is enforced across scenarios by
# planner.top_up_rare_entity_coverage() after the per-scenario plans
# are built. Without this, rare entities (CREDIT_CARD, MEDICAL_LICENSE,
# NRP, IBAN_CODE at documents_per_scenario=100) can silently land at
# zero examples in validation or test -- see audit_entity_split_coverage
# in audit.py, which enforces this floor as part of the dataset audit.
MIN_ENTITY_EXAMPLES_PER_SPLIT = 3
 
 
# Entities Max reported as insufficiently accurate for DistilBERT
# training (2026-08-28), plus URL/PASSPORT which sit in the same thin
# tier of the real dataset but weren't flagged -- included
# preventatively since a category with almost no test examples can't
# produce a trustworthy accuracy number in the first place, flagged
# or not.
PRIORITY_ENTITIES = {
    "IBAN_CODE",
    "CREDIT_CARD",
    "MEDICAL_LICENSE",
    "PHONE_NUMBER",
    "IP_ADDRESS",
    "NRP",
    "URL",
    "PASSPORT",
}
 
 
# Per-split targets for PRIORITY_ENTITIES only. Train is sized for
# DistilBERT to actually learn the pattern (Max's estimate: 30-40+).
# Validation/test are raised well above MIN_ENTITY_EXAMPLES_PER_SPLIT
# because the original complaint ("not enough accuracy") is at least
# partly an eval-noise problem: an entity with 1-2 test examples
# produces an accuracy number that is 0% or 100% by chance, not a
# real measurement. All other (non-priority) entities keep the
# MIN_ENTITY_EXAMPLES_PER_SPLIT floor.
PRIORITY_ENTITY_SPLIT_TARGETS = {
    "train": 40,
    "validation": 10,
    "test": 10,
}
 
 
# Hard ceiling on how skewed any single scenario/split's positive rate
# is allowed to become as a side effect of rare-entity top-up. Without
# this, an entity confined to one scenario (e.g. CREDIT_CARD in
# invoice) can push that scenario/split's positive rate to 50%+ while
# converting negatives -- this caps it at a level close to the
# dataset-wide realized rate instead. CREDIT_CARD and MEDICAL_LICENSE
# will land a bit under their 40/10/10 target because of this cap --
# expected, see PRIORITY_ENTITY_SPLIT_TARGETS comment above.
MAX_LOCAL_POSITIVE_RATE = 0.24
 
 
TARGET_LANGUAGE_RATIOS = {
    "en": 0.50,
    "de": 0.50,
}
 
 
# Blank forms are useful negative examples, but should remain a minority rather than dominating form-based scenarios.
TARGET_BLANK_RATE = 0.05
 
 
DEFAULT_DOCUMENTS_PER_SCENARIO = 100
 
 
DATASET_COLUMNS = [
    "document_id",
    "file_name",
    "document_type",
    "scenario_type",
    "language",
 
    "source_system",
    "responsible_owner",
    "owner_email",
    "data_source",
 
    "file_created_date",
    "last_modified_date",
    "dataset_created_at",
    "file_size_mb",
 
    "full_text",
 
    "contains_personal_data",
 
    "PERSON_yes_no",
    "EMAIL_ADDRESS_yes_no",
    "PHONE_NUMBER_yes_no",
    "LOCATION_yes_no",
    "IBAN_CODE_yes_no",
    "CREDIT_CARD_yes_no",
    "PASSPORT_yes_no",
    "NRP_yes_no",
    "DATE_TIME_yes_no",
    "IP_ADDRESS_yes_no",
    "URL_yes_no",
    "MEDICAL_LICENSE_yes_no",
 
    "personal_data_categories",
    "primary_pii_type",
    "category_count",
    "pii_count",
 
    "difficulty",
    "edge_case",
    "challenge_category",
    "retention_period_exceeded_3y",
    "recommended_split",
 
    "synthetic",
    "synthetic_generator",
    "generation_prompt_version",
 
    "human_reviewed",
    "review_status",
    "review_notes",
 
    "dataset_version",
    "labeling_notes",
]