"""
Classification runtime configuration.

This file contains non-secret configuration for the production-oriented
classification pipeline.

"""

from pathlib import Path

from config.settings import (
    OPENROUTER_MODEL,
    QWEN_MODEL,
)

from classification.schemas.experiment_schema import (
    METADATA_FIELD_NAMES,
)

# Development/test mode.
# Use None for full classification.
CLASSIFICATION_LIMIT = None

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

CLASSIFICATION_DIR = Path(__file__).resolve().parent

DATA_DIR = CLASSIFICATION_DIR / "data"
RESULTS_DIR = CLASSIFICATION_DIR / "results"

DEFAULT_INPUT_FILE = DATA_DIR / "pii_dataset.csv"

# ─────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "gpt4o_mini": {
        "provider": "openrouter",
        "model_family": "llm",
        "model_name": OPENROUTER_MODEL,
        "prediction_source": "llm",
        "is_local": False,
    },

    "qwen3_7_plus": {
        "provider": "dashscope",
        "model_family": "llm",
        "model_name": QWEN_MODEL,
        "prediction_source": "llm",
        "is_local": False,
    },

    # Future
    "distilbert": {
        "provider": "local",
        "model_family": "bert",
        "model_name": "distilbert",
        "prediction_source": "bert",
        "is_local": True,
    },

    "modernbert": {
        "provider": "local",
        "model_family": "bert",
        "model_name": "modernbert",
        "prediction_source": "bert",
        "is_local": True,
    },
}

# ─────────────────────────────────────────────────────────────
# Strategy Registry
# ─────────────────────────────────────────────────────────────

STRATEGY_REGISTRY = {
    "rule_based": {
        "runner": "rule_based",
    },

    "rule_plus_qwen": {
        "runner": "rule_plus_llm",
        "model": "qwen3_7_plus",
    },

    "rule_plus_gpt4o_mini": {
        "runner": "rule_plus_llm",
        "model": "gpt4o_mini",
    },

    # Future
    "bert_distilbert": {
        "runner": "bert",
        "model": "distilbert",
    },

    "bert_modernbert": {
        "runner": "bert",
        "model": "modernbert",
    },

    "rule_plus_distilbert": {
        "runner": "rule_plus_bert",
        "model": "distilbert",
    },

    "rule_plus_modernbert": {
        "runner": "rule_plus_bert",
        "model": "modernbert",
    },

    "rule_plus_distilbert_plus_qwen": {
        "runner": "hybrid",
        "bert_model": "distilbert",
        "llm_model": "qwen3_7_plus",
    },
}

STRATEGIES_TO_RUN = [
    "rule_based",
    "rule_plus_qwen",
    "rule_plus_gpt4o_mini",
]

# ─────────────────────────────────────────────────────────────
# Output metadata
# ─────────────────────────────────────────────────────────────

DEFAULT_PROMPT_VERSION = "pii_review_v1"

DEFAULT_DATASET_VERSION = "v1"

DEFAULT_OUTPUT_COLUMNS_TO_ADD = list(
    METADATA_FIELD_NAMES
)

# ─────────────────────────────────────────────────────────────
# Routing Strategies
# ─────────────────────────────────────────────────────────────

DEFAULT_STRATEGY = "rule_plus_qwen"

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def get_model_config(model_id: str) -> dict:
    """
    Return model configuration.
    """

    model_id = model_id.lower().strip()

    if model_id not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())

        raise ValueError(
            f"Unsupported model: {model_id}. "
            f"Supported models: {supported}"
        )

    return MODEL_REGISTRY[model_id]


def get_model_name(model_id: str) -> str:
    """
    Return configured model name.
    """
    return get_model_config(model_id)["model_name"]


def get_model_family(model_id: str) -> str:
    """
    Return model family.
    """
    return get_model_config(model_id)["model_family"]

def get_provider(model_id: str) -> str:
    """
    Return provider for a model.
    """

    return get_model_config(model_id)["provider"]

def make_safe_filename(value: str) -> str:
    """
    Convert provider/model names into filesystem-safe strings.
    """
    return (
        str(value)
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def validate_models(models: list[str]) -> list:
    """
    Validate and normalize model identifiers.
    """

    normalized = []

    for model_id in models:
        model_id = model_id.lower().strip()

        get_model_config(model_id)

        normalized.append(model_id)

    return normalized

def get_strategy_config(strategy: str) -> dict:
    """
    Return strategy configuration.
    """

    strategy = strategy.lower().strip()

    if strategy not in STRATEGY_REGISTRY:
        supported = ", ".join(STRATEGY_REGISTRY.keys())

        raise ValueError(
            f"Unsupported strategy: {strategy}. "
            f"Supported strategies: {supported}"
        )

    return STRATEGY_REGISTRY[strategy]

def validate_strategies(
    strategies: list[str],
) -> list:
    """
    Validate and normalize strategy names.
    """

    normalized = []

    for strategy in strategies:
        strategy = strategy.lower().strip()

        get_strategy_config(strategy)

        normalized.append(strategy)

    return normalized
