"""
Prediction metadata helpers.

Responsibilities:
- Add model metadata
- Add pipeline metadata
- Add strategy metadata
- Compute final prediction fields

This module contains output-enrichment logic only.
"""

import pandas as pd

from classification.config import (
    DEFAULT_PIPELINE_NAME,
    DEFAULT_PREDICTION_STAGE,
    get_provider_config,
)

def add_sweep1_metadata(
    df: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    """
    Add metadata to Sweep 1 outputs.

    Sweep 1 is a local rule-based baseline using Presidio + regex.
    """

    result_df = df.copy()

    result_df["run_id"] = run_id
    result_df["provider"] = "local"
    result_df["model_family"] = "rule_based"
    result_df["model_name"] = "presidio_regex_v1"
    result_df["prediction_source"] = "presidio_regex"
    result_df["prediction_stage"] = "sweep1"
    result_df["pipeline_name"] = DEFAULT_PIPELINE_NAME

    result_df["predicted_pii"] = result_df["detected_pii"].fillna(False).astype(bool)

    return result_df

def compute_final_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the final production PII decision.

    Logic:
    - Strong Sweep 1 detection means final PII.
    - If no strong Sweep 1 detection but LLM reviewed the row, use LLM decision.
    - Otherwise, classify as non-PII.
    """

    result_df = df.copy()

    if "llm_pii" not in result_df.columns:
        result_df["llm_pii"] = False

    result_df["detected_pii"] = result_df["detected_pii"].fillna(False).astype(bool)
    result_df["llm_pii"] = result_df["llm_pii"].fillna(False).astype(bool)

    result_df["final_pii"] = result_df["detected_pii"] | result_df["llm_pii"]

    # Generic prediction column used by evaluation and future model adapters.
    result_df["predicted_pii"] = result_df["final_pii"]

    return result_df


def add_output_metadata(
    df: pd.DataFrame,
    provider: str,
    run_id: str,
) -> pd.DataFrame:
    """
    Add provider/model/run metadata to classification outputs.

    These columns describe how the prediction was produced.
    They are not evaluation metrics.
    """

    provider_config = get_provider_config(provider)

    result_df = df.copy()

    result_df["run_id"] = run_id
    result_df["provider"] = provider_config["provider"]
    result_df["model_family"] = provider_config["model_family"]
    result_df["model_name"] = provider_config["model_name"]
    result_df["prediction_source"] = provider_config["prediction_source"]
    result_df["prediction_stage"] = DEFAULT_PREDICTION_STAGE
    result_df["pipeline_name"] = DEFAULT_PIPELINE_NAME

    return result_df