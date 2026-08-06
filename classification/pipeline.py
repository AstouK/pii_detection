"""
Production classification pipeline.

Executes:
1. Presidio + regex detection
2. LLM review of ambiguous documents
3. Result export per LLM provider
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.logging_config import setup_logging
from config.settings import (
    OPENROUTER_MODEL,
    QWEN_MODEL,
)

from classification.pii_detector import run_presidio_regex
from classification.llm_reviewer import run_llm


# ── Logging ─────────────────────────────────────────

setup_logging()
logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────

PROVIDERS_TO_RUN = [
    # "openrouter",
    "qwen",
]

TEST_ROWS = 10  # Current max is 500


# ── Paths ───────────────────────────────────────────

BASE_DIR = Path(__file__).parent

DATA_FILE = BASE_DIR / "data" / "pii_dataset.xlsx"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper functions ────────────────────────────────


def get_model_name(provider: str) -> str:
    """
    Return the model name for a given provider.
    """

    provider = provider.lower().strip()

    if provider == "openrouter":
        return OPENROUTER_MODEL

    if provider == "qwen":
        return QWEN_MODEL

    raise ValueError(f"Unsupported provider: {provider}")


def make_safe_filename(value: str) -> str:
    """
    Convert model names into filesystem-safe strings.

    Example:
        openai/gpt-4o-mini -> openai_gpt-4o-mini
    """

    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def build_output_file(provider: str, timestamp: str) -> Path:
    """
    Build the output path for a provider-specific result file.

    Output example:
        classification/results/qwen/qwen3.7-plus_20260728_133500.csv
    """

    provider = provider.lower().strip()
    model_name = get_model_name(provider)
    safe_model_name = make_safe_filename(model_name)

    provider_dir = RESULTS_DIR / provider
    provider_dir.mkdir(parents=True, exist_ok=True)

    return provider_dir / f"{safe_model_name}_{timestamp}.csv"


def load_input_data() -> pd.DataFrame:
    """
    Load the input dataset and optionally keep only the first TEST_ROWS rows.
    """

    logger.info("Loading dataset from %s", DATA_FILE)

    df = pd.read_excel(
        DATA_FILE,
        parse_dates=["file_created_date", "last_modified_date"],
    )

    logger.info("Loaded full dataset with %s rows", len(df))

    if TEST_ROWS:
        df = df.head(TEST_ROWS)
        logger.info("Test mode enabled. Using first %s rows.", TEST_ROWS)

    return df


def run_provider_pipeline(
    base_df: pd.DataFrame,
    provider: str,
    timestamp: str,
) -> Path:
    """
    Run Sweep 2 for a single provider and save provider-specific results.

    Args:
        base_df: DataFrame after Sweep 1.
        provider: LLM provider to use.
        timestamp: Timestamp shared across this pipeline run.

    Returns:
        Path to the saved output CSV.
    """

    provider = provider.lower().strip()

    logger.info("Starting Sweep 2 for provider: %s", provider)

    df_provider = base_df.copy(deep=True)

    df_provider = run_llm(
        df_provider,
        provider=provider,
    )

    df_provider["final_pii"] = df_provider["detected_pii"] | df_provider["llm_pii"]

    output_file = build_output_file(
        provider=provider,
        timestamp=timestamp,
    )

    df_provider.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Completed provider '%s'. Results saved to %s",
        provider,
        output_file,
    )

    return output_file


def main() -> None:
    """
    Run the full classification pipeline for all configured providers.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Classification pipeline started")
    logger.info("Providers selected: %s", PROVIDERS_TO_RUN)

    df = load_input_data()

    logger.info("Running Sweep 1: Presidio + regex")
    df_sweep1 = run_presidio_regex(df)

    saved_files = []

    for provider in PROVIDERS_TO_RUN:
        output_file = run_provider_pipeline(
            base_df=df_sweep1,
            provider=provider,
            timestamp=timestamp,
        )

        saved_files.append(output_file)

    logger.info("Classification pipeline completed")

    for file_path in saved_files:
        logger.info("Result file created: %s", file_path)


if __name__ == "__main__":
    main()
