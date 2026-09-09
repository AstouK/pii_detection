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


def compute_bert_final_prediction(
    df: pd.DataFrame,
    standalone_threshold: float,
) -> pd.DataFrame:
    """
    Final decision for the rule + DistilBERT strategy (no LLM stage).

    Logic:
    - Sweep 1 strong detection (``detected_pii``) stays positive.
    - Sweep 1 ambiguous documents scored by BERT (``needs_bert_review``) are
      positive when the model's probability clears the calibrated standalone
      threshold.
    - Everything else is negative.

    Sweep 1 non-ambiguous, non-detected documents (``local_non_pii``) never
    reach BERT and therefore stay negative.
    """

    result_df = df.copy()

    detected = (
        result_df["detected_pii"].fillna(False).astype(bool)
    )

    scored = (
        result_df.get("needs_bert_review", False)
    )
    scored = pd.Series(scored, index=result_df.index).fillna(False).astype(bool)

    probability = pd.to_numeric(
        result_df.get("pii_probability"),
        errors="coerce",
    ).fillna(0.0)

    bert_pii = scored & (probability >= standalone_threshold)

    result_df["bert_pii"] = bert_pii
    result_df["final_pii"] = detected | bert_pii
    result_df["predicted_pii"] = result_df["final_pii"]

    return result_df


def compute_hybrid_final_prediction(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final decision for the rule + DistilBERT + Qwen hybrid strategy.

    Logic:
    - Sweep 1 strong detection stays positive.
    - Among Sweep 1 ambiguous documents scored by BERT:
        - ``confident_pii`` is positive.
        - ``confident_non_pii`` is negative.
        - ``routed_to_llm`` defers to Qwen's decision on exactly those rows.

    Confident BERT decisions are preserved: only rows BERT actually routed to
    the LLM can be flipped by ``llm_pii``.
    """

    result_df = df.copy()

    detected = (
        result_df["detected_pii"].fillna(False).astype(bool)
    )

    scored = pd.Series(
        result_df.get("needs_bert_review", False),
        index=result_df.index,
    ).fillna(False).astype(bool)

    zone = (
        result_df.get("routing_zone", "")
        .astype(str)
        if "routing_zone" in result_df.columns
        else pd.Series("", index=result_df.index)
    )

    confident_pii = scored & (zone == "confident_pii")

    routed = pd.Series(
        result_df.get("routed_to_llm", False),
        index=result_df.index,
    ).fillna(False).astype(bool)
    routed = scored & routed

    if "llm_pii" not in result_df.columns:
        result_df["llm_pii"] = False

    llm_pii = (
        result_df["llm_pii"].fillna(False).astype(bool)
    )

    result_df["bert_pii"] = confident_pii
    result_df["final_pii"] = (
        detected
        | confident_pii
        | (routed & llm_pii)
    )
    result_df["predicted_pii"] = result_df["final_pii"]

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

    runner = strategy_config.get("runner")

    result_df = df.copy()

    if strategy == "rule_based":

        provider = "local"
        model_family = "rules"
        model_name = "presidio_regex_v1"
        prediction_source = "rules"

        resolved_prompt_version = (
            "not_applicable"
        )

    elif runner == "hybrid":

        # Hybrid combines a local BERT pre-filter with an LLM reviewer, so it
        # has no single "model" entry. Provider/model_name describe the LLM
        # stage that produces the escalated decisions, while model_family and
        # prediction_source mark the combined pipeline.
        bert_config = get_model_config(
            strategy_config["bert_model"]
        )
        llm_config = get_model_config(
            strategy_config["llm_model"]
        )

        provider = llm_config["provider"]
        model_family = "bert+llm"
        model_name = (
            f"{bert_config['model_name']}+{llm_config['model_name']}"
        )
        prediction_source = "hybrid"

        resolved_prompt_version = (
            prompt_version
            or "unknown"
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