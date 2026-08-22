"""
Evaluation configuration.

This file contains non-secret defaults for benchmarking, error analysis,
prompt comparison, and future MLflow logging.

"""

from pathlib import Path

from classification.config1 import (
    CLASSIFICATION_DIR,
    PROVIDERS_TO_RUN,
    DEFAULT_INPUT_FILE,
    validate_providers,
)


# ─────────────────────────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────────────────────────

DATA_FILE = DEFAULT_INPUT_FILE

DATASET_VERSION = "v1"

GROUND_TRUTH_COL = "ground_truth_pii"

RAW_GROUND_TRUTH_COL = "contains_personal_data"

TEXT_COL = "full_text"

DOCUMENT_ID_COL = "document_id"


# Optional split filtering.
# Set to None to use all rows.
EVALUATION_SPLITS = [
    "eval",
    "test",
]

SPLIT_COL = "recommended_split"


# Development/test mode.
# Use None for full evaluation.
EVALUATION_LIMIT = None


# ─────────────────────────────────────────────────────────────
# Routing Strategies (bert models to be added later)
# ─────────────────────────────────────────────────────────────

STRATEGIES = {
    "rule_based": {
        "strategy": "rule_based",
        "provider": "local",
        "model_family": "rules",
        "model_name": "presidio_regex",
    },
    "rule_plus_qwen": {
        "strategy": "rule_plus_qwen",
        "provider": "qwen",
        "model_family": "qwen",
        "model_name": "qwen3.7-plus",
    },
    "rule_plus_openrouter": {
        "strategy": "rule_plus_openrouter",
        "provider": "openrouter",
        "model_family": "gpt",
        "model_name": "openai/gpt-4o-mini",
    },
}

# ─────────────────────────────────────────────────────────────
# Provider/model evaluation defaults
# ─────────────────────────────────────────────────────────────

PROVIDERS_TO_EVALUATE = validate_providers(PROVIDERS_TO_RUN)


# ─────────────────────────────────────────────────────────────
# Prompt and prediction-stage metadata
# ─────────────────────────────────────────────────────────────

PROMPT_VERSION = "prompt_v1"

EVALUATION_VERSION = "eval_v1"

DEFAULT_EVALUATION_STAGE = "standardized"

DEFAULT_PREDICTION_COL = "predicted_pii"


# Other useful prediction columns that may be evaluated later.
PREDICTION_COLUMNS = {
    "standardized": "predicted_pii",
    "sweep1_strong": "detected_pii",
    "sweep1_any": "detected_any_pii",
    "sweep2_llm": "llm_pii",
    "final": "final_pii",
}


# ─────────────────────────────────────────────────────────────
# Result paths
# ─────────────────────────────────────────────────────────────

EVALUATION_DIR = CLASSIFICATION_DIR / "evaluation"

RESULTS_DIR = EVALUATION_DIR / "results"

RUNS_DIR = RESULTS_DIR / "runs"

BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"

FEEDBACK_DIR = RESULTS_DIR / "feedback"


# ─────────────────────────────────────────────────────────────
# Error analysis outputs
# ─────────────────────────────────────────────────────────────

ERROR_TYPE_COL = "error_type"

IS_CORRECT_COL = "is_correct"

ERROR_ANALYSIS_OUTPUT_FILES = {
    "predictions_with_error_labels": "predictions_with_error_labels.csv",
    "false_positives": "false_positives.csv",
    "false_negatives": "false_negatives.csv",
    "true_positives": "true_positives.csv",
    "true_negatives": "true_negatives.csv",
    "error_summary": "error_summary.csv",
    "per_run_error_summary": "per_run_error_summary.csv",
    "per_entity_error_summary": "per_entity_error_summary.csv",
    "per_language_error_summary": "per_language_error_summary.csv",
    "per_challenge_error_summary": "per_challenge_error_summary.csv",
    "per_scenario_error_summary": "per_scenario_error_summary.csv",
    "per_model_error_summary": "per_model_error_summary.csv",
    "per_stage_error_summary": "per_stage_error_summary.csv",
}


# ─────────────────────────────────────────────────────────────
# MLflow placeholders for later
# ─────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT_NAME = "gdpr-pii-detection-evaluation"

ENABLE_MLFLOW_LOGGING = False


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def ensure_evaluation_directories() -> None:
    """
    Create evaluation output directories if they do not exist.
    """
    for directory in [
        RESULTS_DIR,
        RUNS_DIR,
        BENCHMARKS_DIR,
        FEEDBACK_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def get_prediction_col_for_stage(stage: str) -> str:
    """
    Return the prediction column associated with an evaluation stage.
    """
    stage = stage.lower().strip()

    if stage not in PREDICTION_COLUMNS:
        supported = ", ".join(PREDICTION_COLUMNS.keys())
        raise ValueError(
            f"Unsupported evaluation stage: {stage}. "
            f"Supported stages: {supported}"
        )

    return PREDICTION_COLUMNS[stage]


def get_evaluation_metadata() -> dict:
    """
    Return common metadata attached to evaluation outputs.
    """
    return {
        "dataset_version": DATASET_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "evaluation_stage": DEFAULT_EVALUATION_STAGE,
        "prediction_col": DEFAULT_PREDICTION_COL,
    }