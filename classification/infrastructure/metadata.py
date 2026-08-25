"""
Prediction metadata helpers.

Responsibilities:
- Add model metadata
- Add pipeline metadata
- Add strategy metadata
- Compute final prediction fields

This module contains output-enrichment logic only.
"""

from dataclasses import asdict

import pandas as pd

from classification.config import (
    DEFAULT_DATASET_VERSION,
    get_model_config,
    get_strategy_config,
)
from classification.schemas.experiment_schema import (
    ExperimentMetadata,
)


def resolve_dataset_version(
    df: pd.DataFrame,
    default_version: str,
) -> str:
    """
    Resolve and validate the dataset version for a classification run.

    Resolution rules:
    1. If the dataset_version column is missing, use default_version.
    2. If the column contains only missing or empty values, use default_version.
    3. If exactly one non-empty version exists, return that version.
    4. If multiple versions exist, raise an error.
    """

    if "dataset_version" not in df.columns:
        return default_version

    versions = (
        df["dataset_version"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    versions = versions[
        versions != ""
    ].unique().tolist()

    if not versions:
        return default_version

    if len(versions) > 1:
        raise ValueError(
            "Cannot determine one dataset version for this run. "
            f"Found multiple values: {versions}"
        )

    return versions[0]


def add_metadata_columns(
    df: pd.DataFrame,
    metadata: ExperimentMetadata,
) -> pd.DataFrame:
    """
    Add canonical experiment metadata columns.
    """

    result_df = df.copy()

    for column, value in asdict(metadata).items():
        result_df[column] = value

    return result_df


def add_sweep1_metadata(
    df: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    """
    Add metadata to Sweep 1 outputs.

    Sweep 1 is a local rule-based baseline using Presidio + regex.
    """

    result_df = df.copy()

    dataset_version = resolve_dataset_version(
        result_df,
        DEFAULT_DATASET_VERSION,
    )

    metadata = ExperimentMetadata(
        run_id=run_id,
        strategy="rule_based",
        provider="local",
        model_family="rules",
        model_name="regex_presidio_fusion_v1",
        prompt_version="not_applicable",
        dataset_version=dataset_version,
    )

    result_df = add_metadata_columns(
        result_df,
        metadata,
    )

    result_df["prediction_source"] = "rules"

    result_df["predicted_pii"] = (
        result_df["detected_pii"]
        .fillna(False)
        .astype(bool)
    )

    return result_df


def compute_final_prediction(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the final production PII decision.

    Logic:
    - Strong Sweep 1 detection means final PII.
    - If no strong Sweep 1 detection but LLM reviewed the row,
      use the LLM decision.
    - Otherwise classify as non-PII.
    """

    result_df = df.copy()

    if "llm_pii" not in result_df.columns:
        result_df["llm_pii"] = False

    result_df["detected_pii"] = (
        result_df["detected_pii"]
        .fillna(False)
        .astype(bool)
    )

    result_df["llm_pii"] = (
        result_df["llm_pii"]
        .fillna(False)
        .astype(bool)
    )

    result_df["final_pii"] = (
        result_df["detected_pii"]
        | result_df["llm_pii"]
    )

    # Generic prediction column used by evaluation
    # and future model adapters.
    result_df["predicted_pii"] = (
        result_df["final_pii"]
    )

    return result_df


def add_output_metadata(
    df: pd.DataFrame,
    strategy: str,
    run_id: str,
    prompt_version: str | None,
) -> pd.DataFrame:
    """
    Add strategy, model, provider, and pipeline metadata
    to classification outputs.

    These fields describe how the prediction was produced
    and are used by evaluation, benchmarking, and MLflow.
    """

    strategy = strategy.lower().strip()

    strategy_config = get_strategy_config(
        strategy
    )

    result_df = df.copy()

    if strategy == "rule_based":

        provider = "local"
        model_family = "rules"
        model_name = "presidio_regex_v1"
        prediction_source = "rules"

        resolved_prompt_version = (
            "not_applicable"
        )

    else:

        model_id = strategy_config["model"]

        model_config = get_model_config(
            model_id
        )

        provider = model_config["provider"]
        model_family = model_config["model_family"]
        model_name = model_config["model_name"]
        prediction_source = (
            model_config["prediction_source"]
        )

        resolved_prompt_version = (
            prompt_version
            or "unknown"
        )

    dataset_version = resolve_dataset_version(
        result_df,
        DEFAULT_DATASET_VERSION,
    )

    metadata = ExperimentMetadata(
        run_id=run_id,
        strategy=strategy,
        provider=provider,
        model_family=model_family,
        model_name=model_name,
        prompt_version=resolved_prompt_version,
        dataset_version=dataset_version,
    )

    result_df = add_metadata_columns(
        result_df,
        metadata,
    )

    result_df["prediction_source"] = (
        prediction_source
    )

    return result_df