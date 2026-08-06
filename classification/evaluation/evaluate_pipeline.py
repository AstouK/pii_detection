"""
Evaluation pipeline for the two-stage GDPR PII detection system.

Runs:
1. Ground-truth normalization
2. Sweep 1 detection once
3. Sweep 2 LLM review for each configured provider
4. Final decision computation
5. Metric reporting and result export
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

from classification.pii_detector import run_presidio_regex, _has_person_hint
from classification.llm_reviewer import run_llm
from classification.evaluation.evaluate_detector import (
    normalise_ground_truth,
    print_metrics,
    metrics_to_dataframe,
)


# ── Logging ─────────────────────────────────────────

setup_logging()
logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────

PROVIDERS_TO_EVALUATE = [
    # "openrouter",
    "qwen",
]

TEST_ROWS = 10


# ── Paths ───────────────────────────────────────────

CLASSIFICATION_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = CLASSIFICATION_DIR / "data" / "pii_dataset.xlsx"

RESULTS_DIR = CLASSIFICATION_DIR / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────


def get_model_name(provider: str) -> str:
    """
    Return the configured model name for a provider.
    """

    provider = provider.lower().strip()

    if provider == "openrouter":
        return OPENROUTER_MODEL

    if provider == "qwen":
        return QWEN_MODEL

    raise ValueError(f"Unsupported provider: {provider}")


def make_safe_filename(value: str) -> str:
    """
    Convert provider/model names into filesystem-safe strings.
    """

    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def build_output_paths(provider: str, timestamp: str) -> tuple[Path, Path]:
    """
    Build provider-specific output paths for predictions and metrics.
    """

    provider = provider.lower().strip()
    model_name = make_safe_filename(get_model_name(provider))

    provider_dir = RESULTS_DIR / provider
    provider_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = provider_dir / f"{model_name}_{timestamp}_predictions.csv"
    metrics_file = provider_dir / f"{model_name}_{timestamp}_metrics.xlsx"

    return predictions_file, metrics_file


def load_evaluation_data() -> pd.DataFrame:
    """
    Load and prepare the labeled evaluation dataset.
    """

    logger.info("Loading evaluation dataset from %s", DATA_FILE)

    df = pd.read_excel(
        DATA_FILE,
        parse_dates=["file_created_date", "last_modified_date"],
    )

    logger.info("Loaded evaluation dataset with %s rows", len(df))

    if TEST_ROWS:
        df = df.head(TEST_ROWS)
        logger.info("Test mode enabled. Using first %s rows.", TEST_ROWS)

    df = normalise_ground_truth(df)

    df = df.rename(columns={"contains_personal_data": "ground_truth_pii"})

    if "ground_truth_pii" not in df.columns:
        raise ValueError(
            "Evaluation requires a 'ground_truth_pii' column. Expected original column: 'contains_personal_data'."
        )

    return df


def log_llm_routing_debug(df: pd.DataFrame) -> None:
    """
    Log summary information about rows routed to LLM review.
    """

    flagged = df[df["needs_llm_review"]]

    logger.info(
        "Rows flagged for LLM review: %s/%s",
        len(flagged),
        len(df),
    )

    logger.debug(
        "full_text null count: %s",
        df["full_text"].isna().sum(),
    )

    logger.debug(
        "full_text empty string count: %s",
        (df["full_text"] == "").sum(),
    )

    if not flagged.empty:
        sample = flagged.iloc[0]

        logger.debug("Sample flagged row:")
        logger.debug(
            "full_text length: %s",
            len(str(sample.get("full_text", "") or "")),
        )
        logger.debug(
            "full_text preview: %s",
            repr(str(sample.get("full_text", ""))[:200]),
        )
        logger.debug("entities: %s", sample.get("entities", "MISSING"))
        logger.debug("detected_pii: %s", sample.get("detected_pii"))
        logger.debug(
            "potential_pii_categories: %s",
            sample.get("potential_pii_categories"),
        )

    else:
        logger.warning("No rows flagged for LLM review. Check routing logic.")
        logger.warning(
            "detected_pii True count: %s",
            df["detected_pii"].sum(),
        )
        logger.warning(
            "potential_pii nonempty count: %s",
            df["potential_pii_categories"].apply(lambda x: len(x) > 0).sum(),
        )
        logger.warning(
            "person_hint True count: %s",
            df["full_text"].apply(_has_person_hint).sum(),
        )


def evaluate_provider(
    base_df: pd.DataFrame,
    provider: str,
    timestamp: str,
) -> dict:
    """
    Run Sweep 2 and evaluation for one provider.
    """

    provider = provider.lower().strip()

    logger.info("Starting evaluation for provider: %s", provider)

    df_provider = base_df.copy(deep=True)

    df_provider = run_llm(
        df_provider,
        provider=provider,
    )

    df_provider["final_pii"] = df_provider["detected_pii"] | df_provider["llm_pii"]

    metrics = print_metrics(df_provider)

    predictions_file, metrics_file = build_output_paths(
        provider=provider,
        timestamp=timestamp,
    )

    df_provider.to_csv(predictions_file, index=False)

    metrics_df = metrics_to_dataframe(metrics)

    if not metrics_df.empty:
        metrics_df.to_excel(metrics_file)

    logger.info(
        "Saved predictions for provider '%s' to %s",
        provider,
        predictions_file,
    )

    logger.info(
        "Saved metrics for provider '%s' to %s",
        provider,
        metrics_file,
    )

    return metrics


def main() -> None:
    """
    Run the evaluation pipeline for all configured providers.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Evaluation pipeline started")
    logger.info("Providers selected: %s", PROVIDERS_TO_EVALUATE)

    df = load_evaluation_data()

    logger.info("Running Sweep 1: Presidio + regex")
    df_sweep1 = run_presidio_regex(df)

    log_llm_routing_debug(df_sweep1)

    all_metrics = {}

    for provider in PROVIDERS_TO_EVALUATE:
        metrics = evaluate_provider(
            base_df=df_sweep1,
            provider=provider,
            timestamp=timestamp,
        )

        all_metrics[provider] = metrics

    logger.info("Evaluation pipeline completed")


if __name__ == "__main__":
    main()
