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

# Development/test mode.
# Use None for full classification.
CLASSIFICATION_LIMIT = 50

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

CLASSIFICATION_DIR = Path(__file__).resolve().parent

DATA_DIR = CLASSIFICATION_DIR / "data"
RESULTS_DIR = CLASSIFICATION_DIR / "results"

DEFAULT_INPUT_FILE = DATA_DIR / "pii_dataset.csv"

# ─────────────────────────────────────────────────────────────
# Provider configuration
# ─────────────────────────────────────────────────────────────

PROVIDER_REGISTRY = {
    "openrouter": {
        "provider": "openrouter",
        "model_family": "gpt",
        "model_name": OPENROUTER_MODEL,
        "prediction_source": "llm",
        "is_local": False,
    },
    "qwen": {
        "provider": "qwen",
        "model_family": "qwen",
        "model_name": QWEN_MODEL,
        "prediction_source": "llm",
        "is_local": False,
    },
}


# Providers used by the normal classification pipeline.
# Change this list to run one or multiple providers.
PROVIDERS_TO_RUN = [
    "qwen",
    "openrouter",
]


# ─────────────────────────────────────────────────────────────
# Output metadata
# ─────────────────────────────────────────────────────────────

DEFAULT_PREDICTION_STAGE = "final"

DEFAULT_PIPELINE_NAME = "two_stage_pii_pipeline"

DEFAULT_OUTPUT_COLUMNS_TO_ADD = [
    "provider",
    "model_family",
    "model_name",
    "prediction_source",
    "prediction_stage",
    "pipeline_name",
]


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def get_provider_config(provider: str) -> dict:
    """
    Return provider configuration from the registry.
    """
    provider = provider.lower().strip()

    if provider not in PROVIDER_REGISTRY:
        supported = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {supported}"
        )

    return PROVIDER_REGISTRY[provider]


def get_model_name(provider: str) -> str:
    """
    Return configured model name for a provider.
    """
    return get_provider_config(provider)["model_name"]


def get_model_family(provider: str) -> str:
    """
    Return model family for a provider.
    """
    return get_provider_config(provider)["model_family"]


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


def validate_providers(providers: list[str]) -> list:
    """
    Validate and normalize a list of provider names.
    """
    normalized = []

    for provider in providers:
        provider = provider.lower().strip()
        get_provider_config(provider)
        normalized.append(provider)

    return normalized