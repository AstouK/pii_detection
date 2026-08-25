"""Canonical experiment schema.

Single source of truth for:

- experiment metadata
- prediction metadata
- evaluation metrics
- routing metrics
- LLM usage metrics
- BERT usage metrics
- cost metrics
- derived metrics

Adding, removing, or renaming a field should happen here first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# ============================================================================
# Core field definition
# ============================================================================


@dataclass(frozen=True)
class FieldDefinition:
    """Definition of one experiment field."""

    name: str
    dtype: str
    default: Any
    source_key: str | None = None

    @property
    def resolved_source_key(self) -> str:
        """Return the source key used to retrieve this field."""

        return self.source_key or self.name


# ============================================================================
# Experiment metadata
# ============================================================================


@dataclass(frozen=True)
class ExperimentMetadata:
    """Metadata identifying one experiment run."""

    run_id: str
    strategy: str
    provider: str
    model_family: str
    model_name: str
    prompt_version: str
    dataset_version: str


# ============================================================================
# Experiment metadata fields
# ============================================================================


METADATA_FIELDS = (
    FieldDefinition("run_id", "string", ""),
    FieldDefinition("strategy", "string", ""),
    FieldDefinition("provider", "string", "not_applicable"),
    FieldDefinition("model_family", "string", ""),
    FieldDefinition("model_name", "string", ""),
    FieldDefinition("prompt_version", "string", "not_applicable"),
    FieldDefinition("dataset_version", "string", ""),
    FieldDefinition("cost_currency", "string", "USD"),
)


# ============================================================================
# Prediction metadata
# ============================================================================

# Prediction metadata is row-level metadata.
#
# In hybrid strategies, different rows from the same experiment may receive
# their final prediction from different components, such as rules, BERT,
# or an LLM.

PREDICTION_METADATA_FIELDS = (
    FieldDefinition(
        name="prediction_source",
        dtype="string",
        default="unknown",
    ),
)


# ============================================================================
# Evaluation metrics
# ============================================================================

# Keep synchronized with compute_all_metrics() until that function is updated
# to consume these definitions directly.

EVALUATION_METRIC_FIELDS = (
    FieldDefinition("precision", "float64", 0.0),
    FieldDefinition("recall", "float64", 0.0),
    FieldDefinition("f1", "float64", 0.0),
    FieldDefinition("accuracy", "float64", 0.0),
    FieldDefinition("TP", "int64", 0),
    FieldDefinition("FP", "int64", 0),
    FieldDefinition("TN", "int64", 0),
    FieldDefinition("FN", "int64", 0),
)


# ============================================================================
# Routing metrics
# ============================================================================


ROUTING_FIELDS = (
    FieldDefinition("documents_total", "int64", 0),
    FieldDefinition("documents_sent_to_review", "int64", 0),
    FieldDefinition("documents_resolved_locally", "int64", 0),
    FieldDefinition("routing_rate", "float64", 0.0),
    FieldDefinition("local_processing_rate", "float64", 0.0),
)


# ============================================================================
# LLM usage metrics
# ============================================================================


LLM_USAGE_FIELDS = (
    FieldDefinition("llm_requests_attempted", "int64", 0),
    FieldDefinition("llm_requests_successful", "int64", 0),
    FieldDefinition("llm_prompt_tokens", "int64", 0),
    FieldDefinition("llm_completion_tokens", "int64", 0),
    FieldDefinition("llm_total_tokens", "int64", 0),
    FieldDefinition("llm_reasoning_tokens", "int64", 0),
    FieldDefinition("llm_cached_tokens", "int64", 0),
)


# ============================================================================
# LLM cost metrics
# ============================================================================


COST_FIELDS = (
    FieldDefinition(
        name="llm_reported_cost",
        dtype="float64",
        default=0.0,
        source_key="reported_cost",
    ),
)


# ============================================================================
# BERT usage metrics
# ============================================================================


BERT_USAGE_FIELDS = (
    FieldDefinition("bert_requests_attempted", "int64", 0),
    FieldDefinition("bert_requests_successful", "int64", 0),
    FieldDefinition("bert_runtime_seconds", "float64", 0.0),
)


# ============================================================================
# Derived metrics
# ============================================================================


DERIVED_FIELDS = (
    FieldDefinition(
        name="bert_average_runtime_seconds",
        dtype="float64",
        default=0.0,
    ),
    FieldDefinition(
        name="cost_per_document",
        dtype="float64",
        default=0.0,
    ),
    FieldDefinition(
        name="cost_per_llm_document",
        dtype="float64",
        default=0.0,
    ),
    FieldDefinition(
        name="tokens_per_llm_document",
        dtype="float64",
        default=0.0,
    ),
    FieldDefinition(
        name="reasoning_token_ratio",
        dtype="float64",
        default=0.0,
    ),
)


# ============================================================================
# Canonical field registry
# ============================================================================

# EXPERIMENT_FIELDS is the complete project-level field registry.
#
# The section tuples above organize fields by purpose. Consumers should use
# the derived lists below instead of rebuilding their own field-name lists.

EXPERIMENT_FIELDS = (
    *METADATA_FIELDS,
    *PREDICTION_METADATA_FIELDS,
    *EVALUATION_METRIC_FIELDS,
    *ROUTING_FIELDS,
    *LLM_USAGE_FIELDS,
    *COST_FIELDS,
    *BERT_USAGE_FIELDS,
    *DERIVED_FIELDS,
)


# ============================================================================
# Convenience field-name lists
# ============================================================================


METADATA_FIELD_NAMES = tuple(
    field.name
    for field in METADATA_FIELDS
)

PREDICTION_METADATA_FIELD_NAMES = tuple(
    field.name
    for field in PREDICTION_METADATA_FIELDS
)

EVALUATION_METRIC_NAMES = tuple(
    field.name
    for field in EVALUATION_METRIC_FIELDS
)

ROUTING_FIELD_NAMES = tuple(
    field.name
    for field in ROUTING_FIELDS
)

LLM_USAGE_FIELD_NAMES = tuple(
    field.name
    for field in LLM_USAGE_FIELDS
)

BERT_USAGE_FIELD_NAMES = tuple(
    field.name
    for field in BERT_USAGE_FIELDS
)

COST_FIELD_NAMES = tuple(
    field.name
    for field in COST_FIELDS
)

DERIVED_FIELD_NAMES = tuple(
    field.name
    for field in DERIVED_FIELDS
)

EXPERIMENT_FIELD_NAMES = tuple(
    field.name
    for field in EXPERIMENT_FIELDS
)


# ============================================================================
# Evaluation and MLflow definitions
# ============================================================================

# These fields identify comparable experiment configurations.
#
# prediction_source is deliberately excluded because it may vary row by row
# inside a hybrid strategy.

GROUPBY_FIELDS = (
    "strategy",
    "prompt_version",
    "dataset_version",
)

MODEL_GROUPBY_FIELDS = (
    "provider",
    "model_family",
    "model_name",
)

STRATEGY_GROUPBY_FIELDS = (
    "strategy",
)

PREDICTION_SOURCE_GROUPBY_FIELDS = (
    "prediction_source",
)

# Only stable run-level configuration belongs in MLflow parameters.
#
# run_id is already represented by the MLflow run itself.
# prediction_source is row-level information and is therefore excluded.

MLFLOW_PARAM_FIELDS = (
    "strategy",
    "provider",
    "model_family",
    "model_name",
    "prompt_version",
    "dataset_version",
)


MLFLOW_METRIC_FIELDS = (
    *EVALUATION_METRIC_NAMES,
    *ROUTING_FIELD_NAMES,
    *LLM_USAGE_FIELD_NAMES,
    *BERT_USAGE_FIELD_NAMES,
    *COST_FIELD_NAMES,
    *DERIVED_FIELD_NAMES,
)

BENCHMARK_COLUMN_ORDER = (
    "output_name",
    *METADATA_FIELD_NAMES,
    *PREDICTION_METADATA_FIELD_NAMES,
    "label",
    "n",
    *EVALUATION_METRIC_NAMES,
    *ROUTING_FIELD_NAMES,
    *LLM_USAGE_FIELD_NAMES,
    *BERT_USAGE_FIELD_NAMES,
    *COST_FIELD_NAMES,
    *DERIVED_FIELD_NAMES,
)

# ============================================================================
# Lookup helpers
# ============================================================================


FIELD_DEFINITIONS_BY_NAME = {
    field.name: field
    for field in EXPERIMENT_FIELDS
}


def get_field(
    name: str,
) -> FieldDefinition:
    """Return the definition of one field."""

    try:
        return FIELD_DEFINITIONS_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown experiment field: {name!r}"
        ) from exc


def get_defaults() -> dict[str, Any]:
    """Return defaults for all registered experiment fields."""

    return {
        field.name: field.default
        for field in EXPERIMENT_FIELDS
    }


def metadata_to_dict(
    metadata: ExperimentMetadata,
) -> dict[str, str]:
    """Convert experiment metadata to a dictionary."""

    return asdict(metadata)