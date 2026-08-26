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
    ],
    "supplier_onboarding": [
        ("PERSON", "EMAIL_ADDRESS"),
        ("PERSON", "PHONE_NUMBER"),
        ("LOCATION",),
        ("PERSON", "IBAN_CODE"),
    ],
    "training_evaluation": [
        ("PERSON",),
        ("PERSON", "EMAIL_ADDRESS"),
        ("PERSON", "DATE_TIME"),
    ],
    "employee_record": [
        ("PERSON",),
        ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"),
        ("PERSON", "LOCATION"),
        ("PERSON", "NRP"),
        ("PERSON", "DATE_TIME"),
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
    ],
    "customer_support": [
        ("PERSON",),
        ("EMAIL_ADDRESS",),
        ("PHONE_NUMBER",),
        ("IP_ADDRESS",),
        ("URL",),
        ("PERSON", "EMAIL_ADDRESS"),
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
    ],
    "internal_email": [
        ("PERSON",),
        ("EMAIL_ADDRESS",),
        ("PERSON", "EMAIL_ADDRESS"),
        ("PHONE_NUMBER",),
        ("URL",),
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