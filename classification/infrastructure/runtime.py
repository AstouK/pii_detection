"""
Runtime, routing, and operational usage metrics.

Responsibilities:
- Compute initial Sweep 1 routing metrics
- Aggregate LLM execution statistics
- Aggregate BERT execution statistics
- Aggregate token and provider-reported cost statistics

These are operational metrics, not evaluation-quality metrics.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from classification.schemas.experiment_schema import (
    BERT_USAGE_FIELD_NAMES,
    LLM_USAGE_FIELD_NAMES,
    ROUTING_FIELD_NAMES,
)


def _sum_numeric_column(
    df: pd.DataFrame,
    column: str,
    cast_type: Callable = int,
):
    """
    Sum a numeric dataframe column safely.

    Missing columns, missing values, and invalid values
    are treated as zero.
    """

    if column not in df.columns:
        return cast_type(0)

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(0)

    return cast_type(values.sum())


def _count_true_values(
    df: pd.DataFrame,
    column: str,
) -> int:
    """
    Count truthy values in a dataframe column.

    A missing column produces a count of zero.
    """

    if column not in df.columns:
        return 0

    return int(
        df[column]
        .fillna(False)
        .astype(bool)
        .sum()
    )


def compute_routing_metrics(
    df_sweep1: pd.DataFrame,
) -> dict:
    """
    Compute initial routing metrics from Sweep 1 output.

    These metrics describe how many documents Sweep 1 can classify locally
    and how many require an additional review stage.

    The review destination may later be BERT, an LLM, or a hybrid router.
    """

    documents_total = len(df_sweep1)

    if "needs_review" in df_sweep1.columns:
        documents_sent_to_review = _count_true_values(
            df_sweep1,
            "needs_review",
        )
    else:
        # Backward-compatible fallback.
        documents_sent_to_review = _count_true_values(
            df_sweep1,
            "needs_llm_review",
        )

    documents_resolved_locally = (
        documents_total - documents_sent_to_review
    )

    routing_rate = (
        documents_sent_to_review / documents_total
        if documents_total > 0
        else 0.0
    )

    local_processing_rate = (
        documents_resolved_locally / documents_total
        if documents_total > 0
        else 0.0
    )

    metrics = {
        "documents_total": int(documents_total),
        "documents_sent_to_review": int(
            documents_sent_to_review
        ),
        "documents_resolved_locally": int(
            documents_resolved_locally
        ),
        "routing_rate": round(routing_rate, 4),
        "local_processing_rate": round(
            local_processing_rate,
            4,
        ),
    }

    return {
        field_name: metrics[field_name]
        for field_name in ROUTING_FIELD_NAMES
    }


def compute_llm_usage_summary(
    df_strategy: pd.DataFrame,
) -> dict:
    """
    Aggregate LLM execution, token, and cost statistics.

    A strategy without an LLM produces zero values.
    """

    metrics = {
        "llm_requests_attempted": _count_true_values(
            df_strategy,
            "needs_llm_review",
        ),
        "llm_requests_successful": _count_true_values(
            df_strategy,
            "llm_request_success",
        ),
        "llm_prompt_tokens": _sum_numeric_column(
            df_strategy,
            "llm_prompt_tokens",
        ),
        "llm_completion_tokens": _sum_numeric_column(
            df_strategy,
            "llm_completion_tokens",
        ),
        "llm_total_tokens": _sum_numeric_column(
            df_strategy,
            "llm_total_tokens",
        ),
        "llm_reasoning_tokens": _sum_numeric_column(
            df_strategy,
            "llm_reasoning_tokens",
        ),
        "llm_cached_tokens": _sum_numeric_column(
            df_strategy,
            "llm_cached_tokens",
        ),
        "llm_reported_cost": round(
            _sum_numeric_column(
                df_strategy,
                "llm_request_cost",
                float,
            ),
            8,
        ),
    }

    result = {
        field_name: metrics[field_name]
        for field_name in LLM_USAGE_FIELD_NAMES
    }

    result["llm_reported_cost"] = metrics[
        "llm_reported_cost"
    ]

    return result


def compute_bert_usage_summary(
    df_strategy: pd.DataFrame,
) -> dict:
    """
    Aggregate BERT execution statistics.

    Expected future per-document columns:
        needs_bert_review
        bert_request_success
        bert_runtime_seconds

    A strategy without BERT produces zero values.
    """

    metrics = {
        "bert_requests_attempted": _count_true_values(
            df_strategy,
            "needs_bert_review",
        ),
        "bert_requests_successful": _count_true_values(
            df_strategy,
            "bert_request_success",
        ),
        "bert_runtime_seconds": round(
            _sum_numeric_column(
                df_strategy,
                "bert_runtime_seconds",
                float,
            ),
            4,
        ),
    }

    return {
        field_name: metrics[field_name]
        for field_name in BERT_USAGE_FIELD_NAMES
    }


def compute_strategy_usage_summary(
    df_strategy: pd.DataFrame,
) -> dict:
    """
    Aggregate operational usage for one classification strategy.

    This provides a stable schema across:
        rule_based
        rule_plus_llm
        bert_only
        rule_plus_bert
        rule_plus_bert_plus_llm

    Stages not used by a strategy produce zero-valued metrics.
    """

    llm_usage = compute_llm_usage_summary(
        df_strategy
    )

    bert_usage = compute_bert_usage_summary(
        df_strategy
    )

    return {
        **llm_usage,
        **bert_usage,
    }