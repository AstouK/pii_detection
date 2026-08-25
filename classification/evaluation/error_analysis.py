"""
Row-level error analysis for GDPR PII detection evaluation.

This module assigns TP/TN/FP/FN labels to prediction outputs and creates
error-analysis dataframes that can be saved as evaluation artifacts.

It is provider/model agnostic:
- Works for gpt4o-mini, Qwen, BERT, DistilBERT, Presidio-only, or hybrid outputs.
- Works with any prediction column, but defaults to the standardized
  classification output column: predicted_pii.
"""

from __future__ import annotations

import pandas as pd

from classification.evaluation.config import (
    ERROR_TYPE_COL,
    IS_CORRECT_COL,
    GROUND_TRUTH_COL,
    DEFAULT_PREDICTION_COL,
)

from classification.schemas.experiment_schema import (
    MODEL_GROUPBY_FIELDS,
    PREDICTION_SOURCE_GROUPBY_FIELDS,
    STRATEGY_GROUPBY_FIELDS,
)


# ─────────────────────────────────────────────────────────────
# Boolean normalization
# ─────────────────────────────────────────────────────────────

_TRUTHY = {"yes", "true", "1", "y", "ja", "pii"}
_FALSY = {"no", "false", "0", "n", "nein", "non_pii", "no_pii", "safe"}


def to_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert a Series to boolean values.

    Supported truthy values:
        yes, true, 1, y, ja, pii

    Supported falsy values:
        no, false, 0, n, nein, non_pii, no_pii, safe

    Missing or unknown values default to False.
    """

    def _to_bool(value) -> bool:
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


# ─────────────────────────────────────────────────────────────
# Ground-truth handling
# ─────────────────────────────────────────────────────────────

def resolve_ground_truth_col(
    df: pd.DataFrame,
    preferred_col: str = GROUND_TRUTH_COL,
) -> str:
    """
    Resolve the ground-truth column to use for evaluation.

    Preferred:
        ground_truth_pii

    Fallback:
        contains_personal_data
    """

    if preferred_col in df.columns:
        return preferred_col

    fallback_col = "contains_personal_data"

    if fallback_col in df.columns:
        return fallback_col

    raise ValueError(
        "Cannot run error analysis. Missing ground-truth column. "
        f"Expected either '{preferred_col}' or '{fallback_col}'."
    )


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

def validate_error_analysis_input(
    df: pd.DataFrame,
    ground_truth_col: str,
    prediction_col: str = DEFAULT_PREDICTION_COL,
) -> None:
    """
    Validate that required columns exist before running error analysis.
    """

    required_cols = [
        ground_truth_col,
        prediction_col,
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            "Cannot run error analysis. "
            f"Missing required columns: {missing_cols}"
        )


# ─────────────────────────────────────────────────────────────
# Core error assignment
# ─────────────────────────────────────────────────────────────

def assign_error_types(
    df: pd.DataFrame,
    ground_truth_col: str | None = None,
    prediction_col: str = DEFAULT_PREDICTION_COL,
    error_type_col: str = ERROR_TYPE_COL,
    is_correct_col: str = IS_CORRECT_COL,
) -> pd.DataFrame:
    """
    Assign row-level error labels.

    Adds:
        error_type:
            true_positive
            true_negative
            false_positive
            false_negative

        is_correct:
            True for TP/TN
            False for FP/FN

        evaluated_ground_truth_col:
            Name of the ground-truth column used for this analysis.

        evaluated_prediction_col:
            Name of the prediction column used for this analysis.

    This function does not mutate the input dataframe.
    """

    resolved_ground_truth_col = resolve_ground_truth_col(
        df=df,
        preferred_col=ground_truth_col or GROUND_TRUTH_COL,
    )

    validate_error_analysis_input(
        df=df,
        ground_truth_col=resolved_ground_truth_col,
        prediction_col=prediction_col,
    )

    analysed_df = df.copy()

    y_true = to_bool_series(analysed_df[resolved_ground_truth_col])
    y_pred = to_bool_series(analysed_df[prediction_col])

    analysed_df["_ground_truth_bool"] = y_true
    analysed_df["_prediction_bool"] = y_pred

    analysed_df[error_type_col] = "unknown"

    analysed_df.loc[
        (y_true == True) & (y_pred == True),
        error_type_col,
    ] = "true_positive"

    analysed_df.loc[
        (y_true == False) & (y_pred == False),
        error_type_col,
    ] = "true_negative"

    analysed_df.loc[
        (y_true == False) & (y_pred == True),
        error_type_col,
    ] = "false_positive"

    analysed_df.loc[
        (y_true == True) & (y_pred == False),
        error_type_col,
    ] = "false_negative"

    analysed_df[is_correct_col] = analysed_df[error_type_col].isin(
        ["true_positive", "true_negative"]
    )

    analysed_df["evaluated_ground_truth_col"] = resolved_ground_truth_col
    analysed_df["evaluated_prediction_col"] = prediction_col

    return analysed_df


# ─────────────────────────────────────────────────────────────
# Splits
# ─────────────────────────────────────────────────────────────

def split_by_error_type(
    df: pd.DataFrame,
    error_type_col: str = ERROR_TYPE_COL,
) -> dict[str, pd.DataFrame]:
    """
    Split analysed dataframe into TP/TN/FP/FN dataframes.
    """

    if error_type_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{error_type_col}'. "
            "Run assign_error_types() first."
        )

    return {
        "false_positives": df[df[error_type_col] == "false_positive"].copy(),
        "false_negatives": df[df[error_type_col] == "false_negative"].copy(),
        "true_positives": df[df[error_type_col] == "true_positive"].copy(),
        "true_negatives": df[df[error_type_col] == "true_negative"].copy(),
    }


# ─────────────────────────────────────────────────────────────
# Summaries
# ─────────────────────────────────────────────────────────────

def create_error_summary(
    df: pd.DataFrame,
    error_type_col: str = ERROR_TYPE_COL,
) -> pd.DataFrame:
    """
    Create count and percentage summary by error type.
    """

    if error_type_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{error_type_col}'. "
            "Run assign_error_types() first."
        )

    total = len(df)

    summary = (
        df[error_type_col]
        .value_counts()
        .rename_axis(error_type_col)
        .reset_index(name="count")
    )

    summary["percentage"] = (
        summary["count"] / total if total > 0 else 0.0
    )

    return summary.sort_values(error_type_col).reset_index(drop=True)


def create_grouped_error_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    error_type_col: str = ERROR_TYPE_COL,
) -> pd.DataFrame:
    """
    Create grouped error summaries.

    Example group columns:
        ["run_id"]
        ["primary_pii_type"]
        ["language"]
        ["challenge_category"]
        ["scenario_type"]
        ["strategy"]
        ["provider", "model_family", "model_name"]
        ["prediction_source"]
    """

    if error_type_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{error_type_col}'. "
            "Run assign_error_types() first."
        )

    available_group_cols = [col for col in group_cols if col in df.columns]

    if not available_group_cols:
        return pd.DataFrame()

    grouped = (
        df.groupby(available_group_cols + [error_type_col])
        .size()
        .reset_index(name="count")
        .sort_values(available_group_cols + [error_type_col])
        .reset_index(drop=True)
    )

    return grouped


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def run_error_analysis(
    df: pd.DataFrame,
    ground_truth_col: str | None = None,
    prediction_col: str = DEFAULT_PREDICTION_COL,
) -> dict[str, pd.DataFrame]:
    """
    Run full row-level error analysis.

    Defaults:
        ground_truth_col:
            Uses ground_truth_pii if available.
            Falls back to contains_personal_data.

        prediction_col:
            Uses predicted_pii by default.

    Returns a dictionary of dataframes:
        predictions_with_error_labels
        false_positives
        false_negatives
        true_positives
        true_negatives
        error_summary
        per_run_error_summary
        per_entity_error_summary
        per_language_error_summary
        per_challenge_error_summary
        per_scenario_error_summary
        per_model_error_summary
        per_prediction_source_error_summary
    """

    analysed_df = assign_error_types(
        df=df,
        ground_truth_col=ground_truth_col,
        prediction_col=prediction_col,
    )

    splits = split_by_error_type(analysed_df)

    error_summary = create_error_summary(analysed_df)

    per_run_error_summary = create_grouped_error_summary(
        analysed_df,
        group_cols=["run_id"],
    )

    per_entity_error_summary = create_grouped_error_summary(
        analysed_df,
        group_cols=["primary_pii_type"],
    )

    per_language_error_summary = create_grouped_error_summary(
        analysed_df,
        group_cols=["language"],
    )

    per_challenge_error_summary = create_grouped_error_summary(
        analysed_df,
        group_cols=["challenge_category"],
    )

    per_scenario_error_summary = create_grouped_error_summary(
        analysed_df,
        group_cols=["scenario_type"],
    )

    per_model_error_summary = create_grouped_error_summary(
        analysed_df,
        group_cols=list(
            MODEL_GROUPBY_FIELDS
        ),
    )

    per_prediction_source_error_summary = (
        create_grouped_error_summary(
            analysed_df,
            group_cols=list(
                PREDICTION_SOURCE_GROUPBY_FIELDS
            ),
        )
    )

    per_strategy_error_summary = (
        create_grouped_error_summary(
            analysed_df,
            group_cols=list(
                STRATEGY_GROUPBY_FIELDS
            ),
        )
    )
    
    return {
        "predictions_with_error_labels": analysed_df,
        "false_positives": splits["false_positives"],
        "false_negatives": splits["false_negatives"],
        "true_positives": splits["true_positives"],
        "true_negatives": splits["true_negatives"],
        "error_summary": error_summary,
        "per_run_error_summary": per_run_error_summary,
        "per_entity_error_summary": per_entity_error_summary,
        "per_language_error_summary": per_language_error_summary,
        "per_challenge_error_summary": per_challenge_error_summary,
        "per_scenario_error_summary": per_scenario_error_summary,
        "per_model_error_summary": per_model_error_summary,
        "per_prediction_source_error_summary": per_prediction_source_error_summary,
        "per_strategy_error_summary": per_strategy_error_summary,
    }