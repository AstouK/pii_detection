"""
Update classification dataset schema.

Adds metadata columns needed for:
- synthetic data generation
- feedback loops
- continuous improvement
- future model training
- experiment tracking
- provider/model evaluation

This script does not overwrite existing column values.
It only creates missing columns with safe default values.

Input:
    classification/data/pii_dataset.xlsx

Output:
    classification/data/pii_dataset_updated.csv
"""

from pathlib import Path
from datetime import datetime

import csv
import pandas as pd


# ── Paths ───────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent

INPUT_FILE = DATA_DIR / "pii_dataset.xlsx"

OUTPUT_FILE = DATA_DIR / "pii_dataset_updated.csv"

PREVIEW_OUTPUT_FILE = DATA_DIR / "pii_dataset_updated_preview.csv"


# ── Constants ───────────────────────────────────────

TODAY = datetime.now().strftime("%Y-%m-%d")

TRUTHY_VALUES = {"yes", "true", "1", "y", "ja"}

ENTITY_COLUMNS = [
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
]


# ── New schema columns ──────────────────────────────

NEW_COLUMNS_WITH_DEFAULTS = {
    # Dataset identity / versioning
    "document_id": "",
    "language": "",
    "data_source": "manual",
    "dataset_version": "v1",
    "dataset_created_at": TODAY,

    # Synthetic data support
    "synthetic": "no",
    "synthetic_generator": "",
    "generation_prompt_version": "",
    "scenario_type": "",

    # Evaluation grouping
    "primary_pii_type": "",
    "challenge_category": "",
    "pii_count": "",

    # Feedback loop support
    "human_reviewed": "no",
    "review_status": "",
    "review_notes": "",

    # Error analysis support
    "error_type": "",

    # Prompt/model experiment support
    "prompt_version": "",
    "model_family": "",
    "model_name": "",
}


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing schema columns without modifying existing values.
    """

    for column, default_value in NEW_COLUMNS_WITH_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default_value

    return df


def assign_document_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign stable document IDs to rows that do not already have one.

    Existing document_id values are preserved.
    """

    if "document_id" not in df.columns:
        df["document_id"] = ""

    for idx in df.index:
        value = str(df.at[idx, "document_id"]).strip()

        if not value or value.lower() == "nan":
            df.at[idx, "document_id"] = f"DOC-{idx + 1:04d}"

    return df


def infer_language(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer language when the language column is empty.

    This is a simple heuristic:
    - If common German terms are found, mark as 'de'
    - Otherwise default to 'en'

    Manual review is recommended.
    """

    if "language" not in df.columns:
        df["language"] = ""

    german_terms = [
        "rechnung",
        "mitarbeiter",
        "personalnummer",
        "reisepass",
        "geburtsdatum",
        "adresse",
        "führerschein",
        "herr",
        "frau",
        "abteilung",
        "genehmigt",
    ]

    for idx, row in df.iterrows():
        current_value = str(row.get("language", "")).strip()

        if current_value and current_value.lower() != "nan":
            continue

        text = str(row.get("full_text", "")).lower()

        if any(term in text for term in german_terms):
            df.at[idx, "language"] = "de"
        else:
            df.at[idx, "language"] = "en"

    return df


def infer_scenario_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer scenario_type from document_type, file_name, and full_text.

    Existing scenario_type values are preserved.
    """

    if "scenario_type" not in df.columns:
        df["scenario_type"] = ""

    for idx, row in df.iterrows():
        current_value = str(row.get("scenario_type", "")).strip()

        if current_value and current_value.lower() != "nan":
            continue

        document_type = str(row.get("document_type", "")).lower()
        file_name = str(row.get("file_name", "")).lower()
        text = str(row.get("full_text", "")).lower()

        combined = f"{document_type} {file_name} {text}"

        if "expense" in combined or "reimbursement" in combined:
            df.at[idx, "scenario_type"] = "expense_report"

        elif "invoice" in combined or "rechnung" in combined:
            df.at[idx, "scenario_type"] = "invoice"

        elif "contract" in combined or "vertrag" in combined:
            df.at[idx, "scenario_type"] = "contract"

        elif "employee" in combined or "mitarbeiter" in combined:
            df.at[idx, "scenario_type"] = "employee_record"

        elif "passport" in combined or "reisepass" in combined:
            df.at[idx, "scenario_type"] = "passport_record"

        elif "medical" in combined or "arzt" in combined:
            df.at[idx, "scenario_type"] = "medical_record"

        elif "support" in combined or "ticket" in combined:
            df.at[idx, "scenario_type"] = "customer_support"

        else:
            df.at[idx, "scenario_type"] = "general_document"

    return df


def infer_primary_pii_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer primary_pii_type from *_yes_no columns if the field is empty.

    If multiple entity columns are positive, uses MULTIPLE.
    If none are positive, uses NONE.
    Existing primary_pii_type values are preserved.
    """

    existing_entity_columns = [
        col for col in ENTITY_COLUMNS
        if col in df.columns
    ]

    if "primary_pii_type" not in df.columns:
        df["primary_pii_type"] = ""

    for idx, row in df.iterrows():
        current_value = str(row.get("primary_pii_type", "")).strip()

        if current_value and current_value.lower() != "nan":
            continue

        positive_entities = []

        for col in existing_entity_columns:
            value = str(row.get(col, "")).strip().lower()

            if value in TRUTHY_VALUES:
                entity_name = col.replace("_yes_no", "")
                positive_entities.append(entity_name)

        if len(positive_entities) == 0:
            df.at[idx, "primary_pii_type"] = "NONE"

        elif len(positive_entities) == 1:
            df.at[idx, "primary_pii_type"] = positive_entities[0]

        else:
            df.at[idx, "primary_pii_type"] = "MULTIPLE"

    return df


def infer_pii_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer pii_count from category_count where possible.

    If category_count is unavailable, use the number of positive *_yes_no columns.
    Existing non-empty pii_count values are preserved.
    """

    if "pii_count" not in df.columns:
        df["pii_count"] = 0

    # Ensure pii_count can safely hold numeric values
    df["pii_count"] = pd.to_numeric(
        df["pii_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    existing_entity_columns = [
        col for col in ENTITY_COLUMNS
        if col in df.columns
    ]

    for idx, row in df.iterrows():
        current_value = row.get("pii_count", 0)

        # Preserve existing non-zero pii_count values
        if pd.notna(current_value) and int(current_value) > 0:
            continue

        category_count = row.get("category_count", "")

        if pd.notna(category_count) and str(category_count).strip() != "":
            try:
                df.at[idx, "pii_count"] = int(float(category_count))
                continue
            except ValueError:
                pass

        positive_count = 0

        for col in existing_entity_columns:
            value = str(row.get(col, "")).strip().lower()

            if value in TRUTHY_VALUES:
                positive_count += 1

        df.at[idx, "pii_count"] = positive_count

    return df


def infer_challenge_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill challenge_category using edge_case, labeling_notes, and full_text.

    This is a simple first pass. Manual refinement is recommended.
    """

    if "challenge_category" not in df.columns:
        df["challenge_category"] = ""

    for idx, row in df.iterrows():
        current_value = str(row.get("challenge_category", "")).strip()

        if current_value and current_value.lower() != "nan":
            continue

        edge_case = str(row.get("edge_case", "")).strip().lower()
        notes = str(row.get("labeling_notes", "")).strip().lower()
        text = str(row.get("full_text", "")).strip().lower()

        combined = f"{notes} {text}"

        if edge_case not in TRUTHY_VALUES:
            df.at[idx, "challenge_category"] = "none"

        elif "invoice" in combined or "rechnung" in combined:
            df.at[idx, "challenge_category"] = "invoice_number"

        elif "vat" in combined or "tax id" in combined or "steuer" in combined:
            df.at[idx, "challenge_category"] = "vat_number"

        elif "employee" in combined or "personalnummer" in combined or "mitarbeiter" in combined:
            df.at[idx, "challenge_category"] = "employee_id"

        elif "passport" in combined or "reisepass" in combined:
            df.at[idx, "challenge_category"] = "passport_context"

        elif "medical" in combined or "arzt" in combined or "license" in combined:
            df.at[idx, "challenge_category"] = "medical_context"

        elif "url" in combined or "http" in combined:
            df.at[idx, "challenge_category"] = "url_or_web_identifier"

        elif "ip address" in combined or "ip_address" in combined:
            df.at[idx, "challenge_category"] = "ip_address"

        else:
            df.at[idx, "challenge_category"] = "other_edge_case"

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns into a human-readable schema.

    Any unexpected extra columns are appended at the end.
    """

    preferred_order = [
        # Document identity
        "document_id",
        "file_name",
        "document_type",
        "scenario_type",
        "language",

        # Source / ownership
        "source_system",
        "responsible_owner",
        "owner_email",
        "data_source",

        # File metadata
        "file_created_date",
        "last_modified_date",
        "dataset_created_at",
        "file_size_mb",

        # Text
        "full_text",

        # Document-level ground truth
        "contains_personal_data",

        # Entity-level ground truth
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

        # Category metadata
        "personal_data_categories",
        "primary_pii_type",
        "category_count",
        "pii_count",

        # Difficulty / benchmark metadata
        "difficulty",
        "edge_case",
        "challenge_category",
        "retention_period_exceeded_3y",
        "recommended_split",

        # Synthetic data metadata
        "synthetic",
        "synthetic_generator",
        "generation_prompt_version",

        # Feedback loop metadata
        "human_reviewed",
        "review_status",
        "review_notes",
        "error_type",

        # Experiment metadata
        "prompt_version",
        "model_family",
        "model_name",

        # Dataset versioning / notes
        "dataset_version",
        "labeling_notes",
    ]

    existing_preferred_columns = [
        col for col in preferred_order
        if col in df.columns
    ]

    remaining_columns = [
        col for col in df.columns
        if col not in existing_preferred_columns
    ]

    return df[existing_preferred_columns + remaining_columns]


def main() -> None:
    """
    Load the dataset, update schema, and save a CSV version.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)

    df = add_missing_columns(df)
    df = assign_document_ids(df)
    df = infer_language(df)
    df = infer_scenario_type(df)
    df = infer_primary_pii_type(df)
    df = infer_pii_count(df)
    df = infer_challenge_category(df)
    df = reorder_columns(df)

    TEXT_METADATA_COLUMNS = [
        "synthetic_generator",
        "generation_prompt_version",
        "review_status",
        "review_notes",
        "error_type",
        "prompt_version",
        "model_family",
        "model_name",
    ]

    for col in TEXT_METADATA_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Save full machine-readable dataset.
    # This preserves real line breaks inside full_text.
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
    )

    # Save viewer-friendly preview dataset.
    # This replaces real line breaks inside full_text with visible \n.
    df_preview = df.copy()

    if "full_text" in df_preview.columns:
        df_preview["full_text"] = (
            df_preview["full_text"]
            .astype(str)
            .str.replace("\r\n", "\\n", regex=False)
            .str.replace("\n", "\\n", regex=False)
            .str.replace("\r", "\\n", regex=False)
        )

    df_preview.to_csv(
        PREVIEW_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
    )

    print(f"Updated dataset saved to: {OUTPUT_FILE}")
    print(f"Preview dataset saved to: {PREVIEW_OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()