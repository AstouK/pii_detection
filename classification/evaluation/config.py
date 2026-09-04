"""
Evaluation configuration.

This file contains evaluation-specific configuration only.

Design principle:
- Classification remains the source of truth for models and strategies.
- Evaluation consumes metadata from prediction outputs.
- No provider/model/strategy definitions should be duplicated here.
"""

from pathlib import Path

from classification.config import (
    CLASSIFICATION_DIR,
    DEFAULT_INPUT_FILE,
)

from classification.schemas.experiment_schema import (
    GROUPBY_FIELDS,
    METADATA_FIELD_NAMES,
    MLFLOW_PARAM_FIELDS,
    PREDICTION_METADATA_FIELD_NAMES,
)


# ─────────────────────────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────────────────────────

DATA_FILE = DEFAULT_INPUT_FILE

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
# Routing Strategies
# ─────────────────────────────────────────────────────────────
#
# NOTE (Max, feature/bert-prefilter): the two `bert` entries below were added
# for the transformer pre-filter work package. They describe the outputs written
# by `python -m classification.prefilter.predict`:
#
#   rule_plus_bert   three-zone router; uncertain documents carry
#                    routed_to_llm = True and are escalated to the LLM stage
#   bert_prefilter   the standalone model decision, no routing applied
#
# Raised in the 25.08. meeting rather than merged silently, since this file is
# owned by the evaluation work package.

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
    "rule_plus_bert": {
        "strategy": "rule_plus_bert",
        "provider": "local",
        "model_family": "bert",
        "model_name": "distilbert-base-uncased-v1",
    },
    "bert_prefilter": {
        "strategy": "bert_prefilter",
        "provider": "local",
        "model_family": "bert",
        "model_name": "distilbert-base-uncased-v1",
    },
}

EVALUATION_LIMIT = None


# ─────────────────────────────────────────────────────────────
# Prediction configuration
# ─────────────────────────────────────────────────────────────

EVALUATION_VERSION = "eval_v1"

DEFAULT_EVALUATION_STAGE = "standardized"

DEFAULT_PREDICTION_COL = "predicted_pii"

PREDICTION_COLUMNS = {
    "standardized": "predicted_pii",
    "sweep1_strong": "detected_pii",
    "sweep1_any": "detected_any_pii",
    "sweep2_llm": "llm_pii",
    "final": "final_pii",
}


# Metadata columns expected in prediction outputs.
# Defined centrally in experiment_schema.py.

METADATA_COLUMNS = list(
    METADATA_FIELD_NAMES
)

PREDICTION_METADATA_COLUMNS = list(
    PREDICTION_METADATA_FIELD_NAMES
)


# Schema-driven benchmarking and MLflow configuration.

BENCHMARK_GROUPBY_COLUMNS = GROUPBY_FIELDS

MLFLOW_PARAMETER_COLUMNS = MLFLOW_PARAM_FIELDS


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
    "predictions_with_error_labels":
        "predictions_with_error_labels.csv",

    "false_positives":
        "false_positives.csv",

    "false_negatives":
        "false_negatives.csv",

    "true_positives":
        "true_positives.csv",

    "true_negatives":
        "true_negatives.csv",

    "error_summary":
        "error_summary.csv",

    "per_run_error_summary":
        "per_run_error_summary.csv",

    "per_entity_error_summary":
        "per_entity_error_summary.csv",

    "per_language_error_summary":
        "per_language_error_summary.csv",

    "per_challenge_error_summary":
        "per_challenge_error_summary.csv",

    "per_scenario_error_summary":
        "per_scenario_error_summary.csv",

    "per_model_error_summary":
        "per_model_error_summary.csv",

    "per_strategy_error_summary":
        "per_strategy_error_summary.csv",

    # Review later if error_analysis.py still expects
    # prediction_stage, which no longer exists.
    "per_stage_error_summary":
        "per_stage_error_summary.csv",
}


# ─────────────────────────────────────────────────────────────
# MLflow configuration
# ─────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT_NAME = "pii-classification"

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
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def get_prediction_col_for_stage(
    stage: str,
) -> str:
    """
    Return the prediction column associated with an evaluation stage.
    """

    stage = stage.lower().strip()

    if stage not in PREDICTION_COLUMNS:
        supported = ", ".join(
            PREDICTION_COLUMNS.keys()
        )

        raise ValueError(
            f"Unsupported evaluation stage: {stage}. "
            f"Supported stages: {supported}"
        )

    return PREDICTION_COLUMNS[stage]


def get_evaluation_metadata() -> dict:
    """
    Return metadata describing the evaluation run.

    Experiment metadata such as strategy, provider,
    model family, model name, prompt version, and
    dataset version come from prediction outputs.
    """

    return {
        "evaluation_version": EVALUATION_VERSION,
        "evaluation_stage": DEFAULT_EVALUATION_STAGE,
    }