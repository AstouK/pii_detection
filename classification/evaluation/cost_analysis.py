"""
Cost and routing analysis for GDPR PII evaluation.
"""

from __future__ import annotations

import pandas as pd

from classification.schemas.experiment_schema import (
    BERT_USAGE_FIELDS,
    COST_FIELDS,
    LLM_USAGE_FIELDS,
    ROUTING_FIELDS,
)


def get_strategy_usage(
    classification_metadata: dict,
    strategy: str,
) -> dict:
    """
    Extract strategy-specific usage summary from run metadata.
    """

    strategy_usage = classification_metadata.get(
        "strategy_usage",
        {},
    )

    return strategy_usage.get(
        strategy,
        {},
    )


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safely divide two values.
    """

    if denominator <= 0:
        return 0.0

    return float(numerator) / float(denominator)


def create_cost_summary(
    benchmark_df: pd.DataFrame,
    classification_metadata: dict,
) -> pd.DataFrame:
    """
    Enrich benchmark results with routing, usage,
    runtime, cost, and efficiency metrics.

    Uses strategy_usage generated during classification.
    """

    if benchmark_df.empty:
        return benchmark_df

    benchmark_df = benchmark_df.copy()

    # --------------------------------------------------
    # Routing metrics
    # --------------------------------------------------

    routing_values = {
        "documents_total": classification_metadata.get(
            "documents_total",
            0,
        ),
        "documents_sent_to_review": classification_metadata.get(
            "documents_sent_to_review",
            0,
        ),
        "documents_resolved_locally": classification_metadata.get(
            "documents_resolved_locally",
            0,
        ),
        "routing_rate": classification_metadata.get(
            "routing_rate",
            0.0,
        ),
        "local_processing_rate": classification_metadata.get(
            "local_processing_rate",
            0.0,
        ),
    }

    for field in ROUTING_FIELDS:
        benchmark_df[field.name] = routing_values.get(
            field.name,
            field.default,
        )

    # --------------------------------------------------
    # Strategy usage metrics
    # --------------------------------------------------

    strategy_usage_rows = []

    for _, row in benchmark_df.iterrows():

        strategy = row.get(
            "strategy",
            "",
        )

        usage = get_strategy_usage(
            classification_metadata,
            strategy,
        )

        strategy_usage_rows.append(
            usage
        )

    usage_fields = (
        list(LLM_USAGE_FIELDS)
        + list(BERT_USAGE_FIELDS)
        + list(COST_FIELDS)
    )

    for field in usage_fields:

        benchmark_df[field.name] = [
            usage.get(
                field.resolved_source_key,
                field.default,
            )
            for usage in strategy_usage_rows
        ]

    # --------------------------------------------------
    # Derived metrics
    # --------------------------------------------------

    benchmark_df[
        "bert_average_runtime_seconds"
    ] = benchmark_df.apply(
        lambda row: _safe_divide(
            row["bert_runtime_seconds"],
            row["bert_requests_successful"],
        ),
        axis=1,
    )

    benchmark_df[
        "cost_per_document"
    ] = benchmark_df.apply(
        lambda row: _safe_divide(
            row["llm_reported_cost"],
            row["documents_total"],
        ),
        axis=1,
    )

    benchmark_df[
        "cost_per_llm_document"
    ] = benchmark_df.apply(
        lambda row: _safe_divide(
            row["llm_reported_cost"],
            row["llm_requests_successful"],
        ),
        axis=1,
    )

    benchmark_df[
        "tokens_per_llm_document"
    ] = benchmark_df.apply(
        lambda row: _safe_divide(
            row["llm_total_tokens"],
            row["llm_requests_successful"],
        ),
        axis=1,
    )

    benchmark_df[
        "reasoning_token_ratio"
    ] = benchmark_df.apply(
        lambda row: _safe_divide(
            row["llm_reasoning_tokens"],
            row["llm_completion_tokens"],
        ),
        axis=1,
    )

    return benchmark_df