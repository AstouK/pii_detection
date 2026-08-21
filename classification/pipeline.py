"""
Production classification pipeline.

Executes:
1. Presidio + regex detection
2. LLM review of ambiguous documents
3. Final production prediction
4. Provider-specific result export
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from config.logging_config import setup_logging

from classification.config import (
    PROVIDERS_TO_RUN,
    DEFAULT_INPUT_FILE,
    RESULTS_DIR,
    DEFAULT_PREDICTION_STAGE,
    DEFAULT_PIPELINE_NAME,
    get_provider_config,
    get_model_name,
    make_safe_filename,
    validate_providers,
)

from classification.detectors.pii_detector import run_presidio_regex
from classification.review.llm_reviewer import run_llm

from classification.config import CLASSIFICATION_LIMIT


# ── Logging ─────────────────────────────────────────

setup_logging()
logger = logging.getLogger(__name__)


# ── Paths ───────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper functions ────────────────────────────────

def load_input_data(input_file: Path = DEFAULT_INPUT_FILE) -> pd.DataFrame:
    """
    Load the input dataset and optionally keep only the first CLASSIFICATION_LIMIT rows.
    """

    logger.info("Loading dataset from %s", input_file)

    df = pd.read_csv(input_file)

    date_cols = [
        "file_created_date",
        "last_modified_date",
        "dataset_created_at",
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    logger.info("Loaded full dataset with %s rows", len(df))

    if CLASSIFICATION_LIMIT:
        df = df.head(CLASSIFICATION_LIMIT)
        logger.info("Test mode enabled. Using first %s rows.", CLASSIFICATION_LIMIT)

    return df

def create_run_dir(timestamp: str) -> Path:
    """
    Create a run-specific output directory.

    Example:
        classification/results/runs/20260810_153000/
    """

    run_dir = RESULTS_DIR / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir

def build_output_file(
    provider: str,
    run_dir: Path,
) -> Path:
    """
    Build provider-specific output file inside a run directory.
    """

    return run_dir / f"{provider}.csv"

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

def save_sweep1_results(
    df_sweep1: pd.DataFrame,
    run_dir: Path,
    run_id: str,
) -> Path:
    """
    Save Sweep 1 baseline results.
    """

    output_file = run_dir / "sweep1.csv"

    df_sweep1_output = add_sweep1_metadata(
        df=df_sweep1,
        run_id=run_id,
    )

    df_sweep1_output.to_csv(
        output_file,
        index=False,
    )

    logger.info("Sweep 1 results saved to %s", output_file)

    return output_file

def compute_routing_metrics(df_sweep1: pd.DataFrame) -> dict:
    """
    Compute routing metrics from Sweep 1 output.
    """

    documents_total = len(df_sweep1)

    if "needs_llm_review" not in df_sweep1.columns:
        documents_sent_to_llm = 0
    else:
        documents_sent_to_llm = (
            df_sweep1["needs_llm_review"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    llm_calls_avoided = documents_total - documents_sent_to_llm

    routing_rate = (
        documents_sent_to_llm / documents_total
        if documents_total > 0
        else 0.0
    )

    local_processing_rate = (
        llm_calls_avoided / documents_total
        if documents_total > 0
        else 0.0
    )

    return {
        "documents_total": int(documents_total),
        "documents_sent_to_llm": int(documents_sent_to_llm),
        "llm_calls_avoided": int(llm_calls_avoided),
        "routing_rate": round(routing_rate, 4),
        "local_processing_rate": round(local_processing_rate, 4),
    }

def save_run_metadata(
    run_dir: Path,
    run_id: str,
    providers: list[str],
    saved_files: list[Path],
    routing_metrics: dict,
    runtime_metrics: dict,
    provider_usage: dict,
) -> Path:
    """
    Save run-level classification metadata.
    """

    metadata_file = run_dir / "run_metadata.json"

    metadata = {
        "run_id": run_id,
        "pipeline_name": DEFAULT_PIPELINE_NAME,
        "providers": providers,
        "saved_files": [str(path) for path in saved_files],
        **routing_metrics,
        **runtime_metrics,
        "provider_usage": provider_usage,
    }

    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    logger.info("Run metadata saved to %s", metadata_file)

    return metadata_file

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

def compute_llm_usage_summary(
    df_provider: pd.DataFrame,
) -> dict:
    """
    Aggregate per-document LLM usage for one provider.
    """

    def sum_column(column: str, cast_type):
        if column not in df_provider.columns:
            return cast_type(0)

        values = pd.to_numeric(
            df_provider[column],
            errors="coerce",
        ).fillna(0)

        return cast_type(values.sum())

    successful_requests = (
        int(
            df_provider["llm_request_success"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if "llm_request_success" in df_provider.columns
        else 0
    )

    return {
        "requests_attempted": int(
            df_provider["needs_llm_review"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "requests_successful": successful_requests,
        "prompt_tokens": sum_column(
            "llm_prompt_tokens",
            int,
        ),
        "completion_tokens": sum_column(
            "llm_completion_tokens",
            int,
        ),
        "total_tokens": sum_column(
            "llm_total_tokens",
            int,
        ),
        "reasoning_tokens": sum_column(
            "llm_reasoning_tokens",
            int,
        ),
        "cached_tokens": sum_column(
            "llm_cached_tokens",
            int,
        ),
        "provider_reported_cost": round(
            sum_column("llm_request_cost", float),
            8,
        ),
    }


def run_provider_pipeline(
    base_df: pd.DataFrame,
    provider: str,
    run_dir: Path,
    run_id: str,
) -> tuple[Path, float, dict]:
    """
    Run Sweep 2 for a single provider and save provider-specific results.

    Args:
        base_df: DataFrame after Sweep 1.
        provider: LLM provider to use.
        run_dir: Directory for this pipeline run.
        run_id: Shared run identifier.

    Returns:
        Tuple of saved output path and provider runtime in seconds.
    """

    provider = provider.lower().strip()

    logger.info("Starting Sweep 2 for provider: %s", provider)

    provider_start = perf_counter()

    df_provider = base_df.copy(deep=True)

    df_provider = run_llm(
        df_provider,
        provider=provider,
    )

    df_provider = compute_final_prediction(df_provider)

    df_provider = add_output_metadata(
        df=df_provider,
        provider=provider,
        run_id=run_id,
    )

    output_file = build_output_file(
        provider=provider,
        run_dir=run_dir,
    )

    df_provider.to_csv(
        output_file,
        index=False,
    )

    provider_runtime_seconds = round(perf_counter() - provider_start, 4)

    logger.info(
        "Completed provider '%s' in %.4f seconds. Results saved to %s",
        provider,
        provider_runtime_seconds,
        output_file,
    )

    usage_summary = compute_llm_usage_summary(df_provider)

    return output_file, provider_runtime_seconds, usage_summary


def main() -> None:
    """
    Run the full classification pipeline for all configured providers.
    """

    pipeline_start = perf_counter()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = create_run_dir(run_id)

    providers = validate_providers(PROVIDERS_TO_RUN)

    logger.info("Classification pipeline started")
    logger.info("Run ID: %s", run_id)
    logger.info("Providers selected: %s", providers)

    df = load_input_data()

    logger.info("Running Sweep 1: Presidio + regex")

    sweep1_start = perf_counter()
    df_sweep1 = run_presidio_regex(df)
    sweep1_runtime_seconds = round(perf_counter() - sweep1_start, 4)

    logger.info(
        "Sweep 1 completed in %.4f seconds",
        sweep1_runtime_seconds,
    )

    saved_files = []

    sweep1_file = save_sweep1_results(
        df_sweep1=df_sweep1,
        run_dir=run_dir,
        run_id=run_id,
    )

    saved_files.append(sweep1_file)

    provider_runtime_seconds = {}
    provider_usage = {}

    for provider in providers:
        output_file, runtime_seconds, usage_summary = run_provider_pipeline(
            base_df=df_sweep1,
            provider=provider,
            run_dir=run_dir,
            run_id=run_id,
        )

        saved_files.append(output_file)
        provider_runtime_seconds[provider] = runtime_seconds
        provider_usage[provider] = usage_summary

    routing_metrics = compute_routing_metrics(df_sweep1)

    sweep2_runtime_seconds = round(
        sum(provider_runtime_seconds.values()),
        4,
    )

    pipeline_runtime_seconds = round(
        perf_counter() - pipeline_start,
        4,
    )

    runtime_metrics = {
        "sweep1_runtime_seconds": sweep1_runtime_seconds,
        "sweep2_runtime_seconds": sweep2_runtime_seconds,
        "pipeline_runtime_seconds": pipeline_runtime_seconds,
        "provider_runtime_seconds": provider_runtime_seconds,
    }

    metadata_file = save_run_metadata(
        run_dir=run_dir,
        run_id=run_id,
        providers=providers,
        saved_files=saved_files,
        routing_metrics=routing_metrics,
        runtime_metrics=runtime_metrics,
        provider_usage=provider_usage,
    )

    saved_files.append(metadata_file)

    logger.info("Classification pipeline completed")

    for file_path in saved_files:
        logger.info("Result file created: %s", file_path)


if __name__ == "__main__":
    main()