"""
Metric computation helpers for GDPR PII detection evaluation.

This module computes document-level and entity-level metrics.
It does not print or save anything.
"""

from __future__ import annotations

import ast
import json
from typing import Any

import pandas as pd

from classification.evaluation.config import (
    GROUND_TRUTH_COL,
    RAW_GROUND_TRUTH_COL,
    DEFAULT_PREDICTION_COL,
)


# ─────────────────────────────────────────────────────────────
# Boolean normalization
# ─────────────────────────────────────────────────────────────

_TRUTHY = {"yes", "true", "1", "y", "ja", "pii"}
_FALSY = {"no", "false", "0", "n", "nein", "non_pii", "no_pii", "safe"}


def to_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to booleans.

    Missing or unknown values default to False.
    """

    def _to_bool(value: Any) -> bool:
        if pd.isna(value):
            return False

        if isinstance(value, bool):
            return value

        value_str = str(value).strip().lower()

        if value_str in _TRUTHY:
            return True

        if value_str in _FALSY:
            return False

        return False

    return series.apply(_to_bool)


def resolve_ground_truth_col(
    df: pd.DataFrame,
    preferred_col: str = GROUND_TRUTH_COL,
) -> str:
    """
    Resolve the ground-truth column used for evaluation.

    Preferred:
        ground_truth_pii

    Fallback:
        contains_personal_data
    """

    if preferred_col in df.columns:
        return preferred_col

    if RAW_GROUND_TRUTH_COL in df.columns:
        return RAW_GROUND_TRUTH_COL

    raise ValueError(
        "Missing ground-truth column. Expected either "
        f"'{preferred_col}' or '{RAW_GROUND_TRUTH_COL}'."
    )


# ─────────────────────────────────────────────────────────────
# Core metric helpers
# ─────────────────────────────────────────────────────────────

def confusion(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[int, int, int, int]:
    """
    Compute TP, TN, FP, FN.
    """

    y_true = to_bool_series(y_true)
    y_pred = to_bool_series(y_pred)

    tp = (y_pred & y_true).sum()
    tn = (~y_pred & ~y_true).sum()
    fp = (y_pred & ~y_true).sum()
    fn = (~y_pred & y_true).sum()

    return int(tp), int(tn), int(fp), int(fn)


def compute_binary_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    label: str,
    metadata: dict | None = None,
) -> dict:
    """
    Compute binary classification metrics.
    """

    tp, tn, fp, fn = confusion(y_true, y_pred)

    n = len(y_true)

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    result = {
        "label": label,
        "n": int(n),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }

    if metadata:
        result.update(metadata)

    return result


# ─────────────────────────────────────────────────────────────
# Document-level metrics
# ─────────────────────────────────────────────────────────────

def compute_document_metrics(
    df: pd.DataFrame,
    prediction_col: str = DEFAULT_PREDICTION_COL,
    ground_truth_col: str | None = None,
    label: str | None = None,
) -> dict:
    """
    Compute document-level metrics for one prediction column.

    Defaults to:
        ground truth: ground_truth_pii or contains_personal_data
        prediction: predicted_pii
    """

    resolved_ground_truth_col = resolve_ground_truth_col(
        df=df,
        preferred_col=ground_truth_col or GROUND_TRUTH_COL,
    )

    if prediction_col not in df.columns:
        raise ValueError(
            f"Missing prediction column: {prediction_col}"
        )

    metric_label = label or prediction_col

    metadata = extract_output_metadata(df)

    return compute_binary_metrics(
        y_true=df[resolved_ground_truth_col],
        y_pred=df[prediction_col],
        label=metric_label,
        metadata=metadata,
    )


def compute_available_document_metrics(
    df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Compute metrics for all available document-level prediction columns.

    This supports:
        predicted_pii
        detected_pii
        detected_any_pii
        llm_pii
        final_pii
    """

    candidate_columns = {
        "standardized": "predicted_pii",
        "sweep1_strong": "detected_pii",
        "sweep1_any": "detected_any_pii",
        "sweep2_llm": "llm_pii",
        "final": "final_pii",
    }

    metrics = {}

    for label, prediction_col in candidate_columns.items():
        if prediction_col not in df.columns:
            continue

        # For llm_pii, evaluate only routed rows if possible.
        if prediction_col == "llm_pii" and "needs_llm_review" in df.columns:
            eval_df = df[df["needs_llm_review"].fillna(False).astype(bool)].copy()
        else:
            eval_df = df

        if eval_df.empty:
            continue

        metrics[label] = compute_document_metrics(
            df=eval_df,
            prediction_col=prediction_col,
            label=label,
        )

    return metrics


# ─────────────────────────────────────────────────────────────
# Entity-level metrics
# ─────────────────────────────────────────────────────────────

def get_entity_types_from_columns(df: pd.DataFrame) -> list:
    """
    Infer entity types from ground-truth columns ending in '_yes_no'.
    """

    return [
        col.removesuffix("_yes_no")
        for col in df.columns
        if col.endswith("_yes_no")
    ]


def parse_dict_like(value: Any) -> dict:
    """
    Parse dict-like values that may have been read from CSV.

    Handles:
        dict
        JSON string
        Python-literal string
    """

    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        return {}

    value = value.strip()

    if not value:
        return {}

    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def entity_detected(
    row: pd.Series,
    entity_type: str,
    per_type_col: str = "per_type_conf",
) -> bool:
    """
    Return True if entity_type appears in per_type_conf.
    """

    conf_dict = parse_dict_like(row.get(per_type_col, {}))

    return entity_type in conf_dict


def compute_entity_metrics(
    df: pd.DataFrame,
    per_type_col: str = "per_type_conf",
) -> dict[str, dict]:
    """
    Compute per-entity-type metrics using '<ENTITY_TYPE>_yes_no' columns.

    This is mainly useful for Sweep 1 outputs.
    """

    entity_types = get_entity_types_from_columns(df)

    metrics = {}

    for entity_type in entity_types:
        gt_col = f"{entity_type}_yes_no"

        if gt_col not in df.columns:
            continue

        gt_bool = to_bool_series(df[gt_col])

        pred_bool = df.apply(
            lambda row: entity_detected(
                row=row,
                entity_type=entity_type,
                per_type_col=per_type_col,
            ),
            axis=1,
        )

        metrics[entity_type] = compute_binary_metrics(
            y_true=gt_bool,
            y_pred=pred_bool,
            label=f"entity_{entity_type}",
            metadata=extract_output_metadata(df),
        )

    return metrics


# ─────────────────────────────────────────────────────────────
# Combined metric entrypoint
# ─────────────────────────────────────────────────────────────

def extract_output_metadata(df: pd.DataFrame) -> dict:
    """
    Extract stable output metadata from the dataframe.

    If a column has one unique value, that value is attached to metric rows.
    """

    metadata_cols = [
        "run_id",
        "provider",
        "model_family",
        "model_name",
        "prediction_source",
        "prediction_stage",
        "pipeline_name",
    ]

    metadata = {}

    for col in metadata_cols:
        if col not in df.columns:
            continue

        values = df[col].dropna().unique()

        if len(values) == 1:
            metadata[col] = values[0]

    return metadata


def compute_all_metrics(
    df: pd.DataFrame,
    include_entity_metrics: bool = True,
) -> dict:
    """
    Compute all available metrics for one prediction output.

    Returns:
        {
            "document": {...},
            "entity": {...}
        }
    """

    document_metrics = compute_available_document_metrics(df)

    entity_metrics = {}

    if include_entity_metrics and "per_type_conf" in df.columns:
        entity_metrics = compute_entity_metrics(df)

    return {
        "document": document_metrics,
        "entity": entity_metrics,
    }


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    """
    Flatten nested metrics into a tidy dataframe.
    """

    rows = []

    for metric_group, values in metrics.items():
        if not isinstance(values, dict):
            continue

        for metric_name, metric_values in values.items():
            if not isinstance(metric_values, dict):
                continue

            row = {
                "metric_group": metric_group,
                "metric_name": metric_name,
                **metric_values,
            }

            rows.append(row)

    return pd.DataFrame(rows)