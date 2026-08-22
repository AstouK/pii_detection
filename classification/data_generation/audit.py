"""Reusable audit CLI for synthetic PII datasets."""

from __future__ import annotations

import argparse
import json
import re

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from classification.data_generation.config import (
    DATASET_COLUMNS,
    SUPPORTED_LANGUAGES,
    SUPPORTED_SCENARIOS,
    SUPPORTED_SPLITS,
)


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

MANIFEST_COLUMNS = [
    "document_id",
    "scenario_type",
    "variant",
    "language",
    "contains_personal_data",
    "entity_types",
    "recommended_split",
    "difficulty",
    "edge_case",
    "challenge_category",
    "scenario_seed",
    "row_seed",
]

IDENTIFIER_PATTERNS = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "url": re.compile(
        r"https?://[^\s]+"
    ),
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
    "iban": re.compile(
        r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){10,30}\b"
    ),
}


# Result model

@dataclass
class AuditFinding:
    level: str
    check: str
    message: str


class AuditReport:
    """Collect audit findings and render terminal/file output."""

    def __init__(
        self,
        dataset_path: Path,
        strict: bool,
    ) -> None:
        self.dataset_path = dataset_path
        self.strict = strict
        self.findings: list[AuditFinding] = []
        self.metrics: dict = {}

    def add(
        self,
        level: str,
        check: str,
        message: str,
    ) -> None:
        self.findings.append(
            AuditFinding(
                level=level,
                check=check,
                message=message,
            )
        )

    def pass_(
        self,
        check: str,
        message: str,
    ) -> None:
        self.add("PASS", check, message)

    def warn(
        self,
        check: str,
        message: str,
    ) -> None:
        self.add("WARN", check, message)

    def fail(
        self,
        check: str,
        message: str,
    ) -> None:
        self.add("FAIL", check, message)

    @property
    def failure_count(self) -> int:
        return sum(
            finding.level == "FAIL"
            for finding in self.findings
        )

    @property
    def warning_count(self) -> int:
        return sum(
            finding.level == "WARN"
            for finding in self.findings
        )

    @property
    def status(self) -> str:
        if self.failure_count:
            return "FAIL"

        if self.warning_count:
            return "PASS_WITH_WARNINGS"

        return "PASS"

    def to_dict(self) -> dict:
        return {
            "dataset": str(self.dataset_path),
            "strict_mode": self.strict,
            "status": self.status,
            "failure_count": self.failure_count,
            "warning_count": self.warning_count,
            "metrics": self.metrics,
            "findings": [
                asdict(finding)
                for finding in self.findings
            ],
        }


# Helpers

def normalize_yes(value) -> bool:
    """Return True when a yes/no field represents yes."""

    return (
        str(value)
        .strip()
        .lower()
        == "yes"
    )


def active_entities(
    row: pd.Series,
) -> list[str]:
    """Return entity types marked yes for one dataset row."""

    return [
        column.replace("_yes_no", "")
        for column in ENTITY_COLUMNS
        if column in row.index
        and normalize_yes(row[column])
    ]


def safe_int(value) -> int | None:
    """Convert a numeric-like value to int."""

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# Individual audit checks

def audit_schema(
    df: pd.DataFrame,
    report: AuditReport,
) -> None:
    """Validate the fixed 44-column dataset contract."""

    actual = list(df.columns)

    missing = [
        column
        for column in DATASET_COLUMNS
        if column not in actual
    ]

    unexpected = [
        column
        for column in actual
        if column not in DATASET_COLUMNS
    ]

    report.metrics["rows"] = len(df)
    report.metrics["columns"] = len(actual)

    if missing:
        report.fail(
            "schema",
            f"Missing columns: {missing}",
        )

    if unexpected:
        report.fail(
            "schema",
            f"Unexpected columns: {unexpected}",
        )

    if not missing and not unexpected:
        report.pass_(
            "schema",
            f"Dataset matches the expected "
            f"{len(DATASET_COLUMNS)}-column schema.",
        )

    if actual != DATASET_COLUMNS:
        report.warn(
            "column_order",
            "Columns exist but are not in the canonical order.",
        )
    else:
        report.pass_(
            "column_order",
            "Column order matches the canonical schema.",
        )


def audit_manifest_integrity(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    report: AuditReport,
) -> bool:
    """Validate alignment between dataset and generation manifest."""

    missing = [
        column
        for column in MANIFEST_COLUMNS
        if column not in manifest.columns
    ]

    report.metrics["manifest_rows"] = len(manifest)

    if missing:
        report.fail(
            "manifest_schema",
            f"Manifest is missing columns: {missing}",
        )
        return False

    report.pass_(
        "manifest_schema",
        f"Manifest matches the expected "
        f"{len(MANIFEST_COLUMNS)}-column schema.",
    )

    duplicate_ids = int(
        manifest["document_id"].duplicated().sum()
    )

    if duplicate_ids:
        report.fail(
            "manifest_document_id",
            f"{duplicate_ids} duplicate document IDs "
            "found in manifest.",
        )
        return False

    dataset_ids = set(
        df["document_id"]
    )

    manifest_ids = set(
        manifest["document_id"]
    )

    missing_from_manifest = (
        dataset_ids - manifest_ids
    )

    missing_from_dataset = (
        manifest_ids - dataset_ids
    )

    if missing_from_manifest or missing_from_dataset:
        report.fail(
            "manifest_alignment",
            (
                f"{len(missing_from_manifest)} dataset IDs are "
                "missing from the manifest and "
                f"{len(missing_from_dataset)} manifest IDs are "
                "missing from the dataset."
            ),
        )
        return False

    comparison = df[
        [
            "document_id",
            "scenario_type",
            "language",
            "contains_personal_data",
            "recommended_split",
        ]
    ].merge(
        manifest[
            [
                "document_id",
                "scenario_type",
                "language",
                "contains_personal_data",
                "recommended_split",
            ]
        ],
        on="document_id",
        suffixes=("_dataset", "_manifest"),
        validate="one_to_one",
    )

    mismatch_count = 0

    for column in [
        "scenario_type",
        "language",
        "recommended_split",
    ]:
        mismatch_count += int(
            (
                comparison[f"{column}_dataset"].astype(str)
                != comparison[f"{column}_manifest"].astype(str)
            ).sum()
        )

    dataset_labels = (
        comparison["contains_personal_data_dataset"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    manifest_labels = (
        comparison["contains_personal_data_manifest"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mismatch_count += int(
        (dataset_labels != manifest_labels).sum()
    )

    if mismatch_count:
        report.fail(
            "manifest_alignment",
            f"{mismatch_count} dataset/manifest metadata "
            "values are inconsistent.",
        )
        return False

    report.metrics["manifest_variant_counts"] = (
        manifest["variant"]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    report.metrics["manifest_blank_rows"] = int(
        (
            manifest["variant"]
            == "blank"
        ).sum()
    )

    report.pass_(
        "manifest_alignment",
        "Dataset and manifest align one-to-one.",
    )

    return True


def audit_basic_integrity(
    df: pd.DataFrame,
    report: AuditReport,
    manifest: pd.DataFrame | None = None,
) -> None:
    """Check IDs, text presence, and duplicate text."""

    duplicate_ids = int(
        df["document_id"].duplicated().sum()
    )

    if duplicate_ids:
        report.fail(
            "document_id",
            f"{duplicate_ids} duplicate document IDs found.",
        )
    else:
        report.pass_(
            "document_id",
            "All document IDs are unique.",
        )

    empty_text = (
        df["full_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    empty_count = int(empty_text.sum())

    if empty_count:
        report.fail(
            "full_text",
            f"{empty_count} rows have empty full_text.",
        )
    else:
        report.pass_(
            "full_text",
            "All rows contain document text.",
        )

    duplicated_text = df.duplicated(
        subset=["full_text"],
        keep=False,
    )

    duplicate_rows = int(
        duplicated_text.sum()
    )

    duplicate_groups = int(
        df.loc[
            duplicated_text,
            "full_text",
        ].nunique()
    )

    excess_duplicates = max(
        0,
        duplicate_rows - duplicate_groups,
    )

    report.metrics["duplicate_text_rows"] = duplicate_rows
    report.metrics["duplicate_text_groups"] = duplicate_groups
    report.metrics["excess_duplicate_texts"] = excess_duplicates

    if not excess_duplicates:
        report.pass_(
            "duplicate_text",
            "No exact duplicate full_text values found.",
        )
        return

    # Without a manifest we cannot tell whether duplicate text
    # belongs to an intentional blank template.
    if manifest is None:
        report.warn(
            "duplicate_text",
            f"{duplicate_rows} rows participate in "
            f"{duplicate_groups} duplicate-text groups "
            f"({excess_duplicates} excess copies).",
        )
        return

    variant_lookup = (
        manifest
        .set_index("document_id")["variant"]
        .to_dict()
    )

    duplicate_frame = df.loc[
        duplicated_text,
        [
            "document_id",
            "full_text",
        ],
    ].copy()

    duplicate_frame["variant"] = (
        duplicate_frame["document_id"]
        .map(variant_lookup)
    )

    intentional_blank_groups = 0
    intentional_blank_rows = 0

    unexpected_groups = 0
    unexpected_rows = 0

    for _, group in duplicate_frame.groupby(
        "full_text",
        sort=False,
    ):
        if group["variant"].eq("blank").all():
            intentional_blank_groups += 1
            intentional_blank_rows += len(group)
        else:
            unexpected_groups += 1
            unexpected_rows += len(group)

    report.metrics[
        "intentional_blank_duplicate_groups"
    ] = intentional_blank_groups

    report.metrics[
        "intentional_blank_duplicate_rows"
    ] = intentional_blank_rows

    report.metrics[
        "unexpected_duplicate_groups"
    ] = unexpected_groups

    report.metrics[
        "unexpected_duplicate_rows"
    ] = unexpected_rows

    if unexpected_groups:
        report.warn(
            "duplicate_text",
            (
                f"{unexpected_rows} rows participate in "
                f"{unexpected_groups} unexpected duplicate-text "
                f"group(s). "
                f"{intentional_blank_rows} rows in "
                f"{intentional_blank_groups} duplicate group(s) "
                "are intentional blank templates."
            ),
        )
    else:
        report.pass_(
            "duplicate_text",
            (
                f"All exact duplicate texts belong to intentional "
                f"blank templates "
                f"({intentional_blank_rows} rows across "
                f"{intentional_blank_groups} groups)."
            ),
        )


def audit_supported_values(
    df: pd.DataFrame,
    report: AuditReport,
) -> None:
    """Validate scenarios, languages, and splits."""

    checks = {
        "scenario_type": set(SUPPORTED_SCENARIOS),
        "language": set(SUPPORTED_LANGUAGES),
        "recommended_split": set(SUPPORTED_SPLITS),
    }

    for column, supported in checks.items():
        actual = set(
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        invalid = actual - supported

        if invalid:
            report.fail(
                column,
                f"Unsupported values found: {sorted(invalid)}",
            )
        else:
            report.pass_(
                column,
                f"All values are supported: {sorted(actual)}",
            )


def audit_document_labels(
    df: pd.DataFrame,
    report: AuditReport,
) -> None:
    """Check document-level labels against entity flags."""

    positive_without_entities = []
    negative_with_entities = []

    for _, row in df.iterrows():
        entities = active_entities(row)
        positive = normalize_yes(
            row["contains_personal_data"]
        )

        if positive and not entities:
            positive_without_entities.append(
                row["document_id"]
            )

        if not positive and entities:
            negative_with_entities.append(
                {
                    "document_id": row["document_id"],
                    "entities": entities,
                }
            )

    if positive_without_entities:
        report.fail(
            "positive_entity_consistency",
            f"{len(positive_without_entities)} "
            "PII-positive rows have zero entity flags.",
        )
    else:
        report.pass_(
            "positive_entity_consistency",
            "Every PII-positive row has at least one entity flag.",
        )

    if negative_with_entities:
        report.fail(
            "negative_entity_consistency",
            f"{len(negative_with_entities)} "
            "PII-negative rows contain positive entity flags.",
        )
    else:
        report.pass_(
            "negative_entity_consistency",
            "No PII-negative row contains a positive entity flag.",
        )


def audit_entity_metadata(
    df: pd.DataFrame,
    report: AuditReport,
) -> None:
    """Validate derived category/entity metadata."""

    primary_errors = 0
    count_errors = 0
    category_errors = 0

    for _, row in df.iterrows():
        entities = active_entities(row)

        if not entities:
            expected_primary = "NONE"
        elif len(entities) == 1:
            expected_primary = entities[0]
        else:
            expected_primary = "MULTIPLE"

        actual_primary = str(
            row["primary_pii_type"]
        ).strip()

        if actual_primary != expected_primary:
            primary_errors += 1

        expected_count = len(entities)

        category_count = safe_int(
            row["category_count"]
        )
        pii_count = safe_int(
            row["pii_count"]
        )

        if (
            category_count != expected_count
            or pii_count != expected_count
        ):
            count_errors += 1

        categories_raw = str(
            row.get(
                "personal_data_categories",
                "",
            )
        ).strip()

        if categories_raw.lower() == "nan":
            categories_raw = ""

        actual_categories = {
            item.strip()
            for item in categories_raw.split(";")
            if item.strip()
        }

        if actual_categories != set(entities):
            category_errors += 1

    if primary_errors:
        report.fail(
            "primary_pii_type",
            f"{primary_errors} rows have inconsistent "
            "primary_pii_type.",
        )
    else:
        report.pass_(
            "primary_pii_type",
            "primary_pii_type is consistent with entity flags.",
        )

    if count_errors:
        report.fail(
            "entity_counts",
            f"{count_errors} rows have inconsistent "
            "category_count or pii_count.",
        )
    else:
        report.pass_(
            "entity_counts",
            "category_count and pii_count are consistent.",
        )

    if category_errors:
        report.fail(
            "personal_data_categories",
            f"{category_errors} rows have inconsistent "
            "personal_data_categories.",
        )
    else:
        report.pass_(
            "personal_data_categories",
            "personal_data_categories matches entity flags.",
        )


def audit_negative_identifier_patterns(
    df: pd.DataFrame,
    report: AuditReport,
) -> None:
    """Detect obvious identifier literals in negative documents."""

    negative = df[
        ~df["contains_personal_data"]
        .map(normalize_yes)
    ]

    hits = []

    for _, row in negative.iterrows():
        text = str(
            row["full_text"]
        )

        row_hits = {}

        for name, pattern in IDENTIFIER_PATTERNS.items():
            matches = pattern.findall(text)

            if matches:
                row_hits[name] = matches

        if row_hits:
            hits.append(
                {
                    "document_id": row["document_id"],
                    "scenario": row["scenario_type"],
                    "matches": row_hits,
                }
            )

    report.metrics["negative_identifier_hits"] = len(hits)

    if hits:
        report.fail(
            "negative_identifier_leakage",
            f"{len(hits)} negative documents contain "
            "obvious identifier patterns.",
        )
    else:
        report.pass_(
            "negative_identifier_leakage",
            "No obvious email, URL, IP, or IBAN patterns "
            "were found in negative documents.",
        )


def audit_generation_metadata(
    df: pd.DataFrame,
    report: AuditReport,
) -> None:
    """Summarize generation methods and fallback usage."""

    generator_counts = (
        df["synthetic_generator"]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    prompt_counts = (
        df["generation_prompt_version"]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    fallback_count = int(
        (
            df["synthetic_generator"]
            == "faker_archetype_fallback"
        ).sum()
    )

    report.metrics["generator_counts"] = generator_counts
    report.metrics["prompt_version_counts"] = prompt_counts
    report.metrics["fallback_rows"] = fallback_count

    if fallback_count:
        report.warn(
            "llm_fallbacks",
            f"{fallback_count} rows used deterministic "
            "fallback after LLM generation failed.",
        )
    else:
        report.pass_(
            "llm_fallbacks",
            "No LLM fallback rows were generated.",
        )


def audit_entity_coverage(
    df: pd.DataFrame,
    report: AuditReport,
) -> None:
    """Report entity representation."""

    coverage = {}

    for column in ENTITY_COLUMNS:
        entity = column.replace(
            "_yes_no",
            "",
        )

        coverage[entity] = int(
            df[column]
            .map(normalize_yes)
            .sum()
        )

    report.metrics["entity_coverage"] = coverage

    missing = [
        entity
        for entity, count in coverage.items()
        if count == 0
    ]

    if missing:
        report.warn(
            "entity_coverage",
            f"Entity types absent from dataset: {missing}",
        )
    else:
        report.pass_(
            "entity_coverage",
            "All configured entity types are represented.",
        )


def audit_distributions(
    df: pd.DataFrame,
    report: AuditReport,
    strict: bool,
) -> None:
    """Summarize scenario/class/language/split coverage."""

    scenario_counts = (
        df["scenario_type"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    language_counts = (
        df["language"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    split_counts = (
        df["recommended_split"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    class_counts = (
        df["contains_personal_data"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .to_dict()
    )

    report.metrics["scenario_counts"] = scenario_counts
    report.metrics["language_counts"] = language_counts
    report.metrics["split_counts"] = split_counts
    report.metrics["class_counts"] = class_counts

    scenario_split = pd.crosstab(
        df["scenario_type"],
        df["recommended_split"],
    )

    missing_scenario_split = []

    for scenario in SUPPORTED_SCENARIOS:
        for split in SUPPORTED_SPLITS:
            count = (
                scenario_split
                .get(split, pd.Series(dtype=int))
                .get(scenario, 0)
            )

            if count == 0:
                missing_scenario_split.append(
                    f"{scenario}/{split}"
                )

    if missing_scenario_split:
        level = report.fail if strict else report.warn

        level(
            "scenario_split_coverage",
            "Missing scenario/split combinations: "
            f"{missing_scenario_split}",
        )
    else:
        report.pass_(
            "scenario_split_coverage",
            "Every scenario is represented in every split.",
        )

    split_class = pd.crosstab(
        df["recommended_split"],
        df["contains_personal_data"]
        .astype(str)
        .str.strip()
        .str.lower(),
    )

    missing_split_class = []

    for split in SUPPORTED_SPLITS:
        for label in {"yes", "no"}:
            count = (
                split_class
                .get(label, pd.Series(dtype=int))
                .get(split, 0)
            )

            if count == 0:
                missing_split_class.append(
                    f"{split}/{label}"
                )

    if missing_split_class:
        level = report.fail if strict else report.warn

        level(
            "split_class_coverage",
            "Missing split/class combinations: "
            f"{missing_split_class}",
        )
    else:
        report.pass_(
            "split_class_coverage",
            "Every split contains both target classes.",
        )


# Main audit

def run_audit(
    dataset_path: Path,
    strict: bool = False,
    manifest_path: Path | None = None,
) -> AuditReport:
    """Run the reusable synthetic dataset audit."""

    df = pd.read_csv(
        dataset_path
    )

    manifest = None

    if manifest_path is not None:
        manifest = pd.read_csv(
            manifest_path
        )

    report = AuditReport(
        dataset_path=dataset_path,
        strict=strict,
    )

    audit_schema(
        df=df,
        report=report,
    )

    # Stop deeper checks when essential schema is missing.
    missing_required = [
        column
        for column in DATASET_COLUMNS
        if column not in df.columns
    ]

    if missing_required:
        return report

    manifest_valid = False

    if manifest is not None:
        manifest_valid = audit_manifest_integrity(
            df=df,
            manifest=manifest,
            report=report,
        )

    audit_basic_integrity(
        df=df,
        report=report,
        manifest=(
            manifest
            if manifest_valid
            else None
        ),
    )

    audit_supported_values(
        df=df,
        report=report,
    )

    audit_document_labels(
        df=df,
        report=report,
    )

    audit_entity_metadata(
        df=df,
        report=report,
    )

    audit_negative_identifier_patterns(
        df=df,
        report=report,
    )

    audit_generation_metadata(
        df=df,
        report=report,
    )

    audit_entity_coverage(
        df=df,
        report=report,
    )

    audit_distributions(
        df=df,
        report=report,
        strict=strict,
    )

    return report


# Output 

def render_text_report(
    report: AuditReport,
) -> str:
    """Render a human-readable audit report."""

    lines = [
        "=" * 72,
        "SYNTHETIC DATASET AUDIT",
        "=" * 72,
        f"Dataset: {report.dataset_path}",
        f"Mode: {'STRICT' if report.strict else 'STANDARD'}",
        f"Status: {report.status}",
        "",
    ]

    grouped = {
        "FAIL": [],
        "WARN": [],
        "PASS": [],
    }

    for finding in report.findings:
        grouped[finding.level].append(
            finding
        )

    for level in [
        "FAIL",
        "WARN",
        "PASS",
    ]:
        lines.append(
            f"{level} CHECKS"
        )
        lines.append(
            "-" * 72
        )

        findings = grouped[level]

        if not findings:
            lines.append(
                f"[{level}] None"
            )
        else:
            for finding in findings:
                lines.append(
                    f"[{level}] "
                    f"{finding.check}: "
                    f"{finding.message}"
                )

        lines.append("")

    lines.append(
        "METRICS"
    )
    lines.append(
        "-" * 72
    )

    for key, value in report.metrics.items():
        lines.append(
            f"{key}: {value}"
        )

    lines.extend(
        [
            "",
            "=" * 72,
            "SUMMARY",
            "=" * 72,
            f"Status: {report.status}",
            f"Failures: {report.failure_count}",
            f"Warnings: {report.warning_count}",
        ]
    )

    return "\n".join(lines)


def save_reports(
    report: AuditReport,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Save text and JSON audit results."""

    if output_dir is None:
        output_dir = (
            report.dataset_path.parent
            / "audits"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    stem = report.dataset_path.stem

    text_path = (
        output_dir
        / f"{stem}_{timestamp}_audit.txt"
    )

    json_path = (
        output_dir
        / f"{stem}_{timestamp}_audit.json"
    )

    text_report = render_text_report(
        report
    )

    text_path.write_text(
        text_report,
        encoding="utf-8",
    )

    json_path.write_text(
        json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return text_path, json_path


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a synthetic PII dataset "
            "and save reusable QA reports."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the dataset CSV.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat important distribution/coverage "
            "gaps as failures."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional audit report directory. "
            "Defaults to <dataset folder>/audits/."
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional generation manifest CSV. "
            "Used for variant-aware and duplicate-aware checks."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.dataset.exists():
        print(
            f"Dataset does not exist: "
            f"{args.dataset}"
        )
        return 1

    report = run_audit(
        dataset_path=args.dataset,
        strict=args.strict,
        manifest_path=args.manifest,
    )

    text_report = render_text_report(
        report
    )

    print(text_report)

    text_path, json_path = save_reports(
        report=report,
        output_dir=args.output_dir,
    )

    print()
    print(
        f"Text report saved to: {text_path}"
    )
    print(
        f"JSON report saved to: {json_path}"
    )

    return (
        1
        if report.failure_count
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())