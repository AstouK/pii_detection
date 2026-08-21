"""
Cost and routing analysis for GDPR PII evaluation.
"""

from __future__ import annotations

import pandas as pd


def get_provider_usage(
    classification_metadata: dict,
    provider: str,
) -> dict:
    """
    Extract provider-specific usage summary from run metadata.
    """

    provider_usage = classification_metadata.get(
        "provider_usage",
        {},
    )

    return provider_usage.get(provider, {})


def create_cost_summary(
    benchmark_df: pd.DataFrame,
    classification_metadata: dict,
) -> pd.DataFrame:
    """
    Enrich benchmark results with routing and cost information.

    Uses provider_usage generated during classification.
    """

    if benchmark_df.empty:
        return benchmark_df

    benchmark_df = benchmark_df.copy()

    documents_total = classification_metadata.get(
        "documents_total",
        0,
    )

    documents_sent_to_llm = classification_metadata.get(
        "documents_sent_to_llm",
        0,
    )

    llm_calls_avoided = classification_metadata.get(
        "llm_calls_avoided",
        0,
    )

    routing_rate = classification_metadata.get(
        "routing_rate",
        0.0,
    )

    local_processing_rate = classification_metadata.get(
        "local_processing_rate",
        0.0,
    )

    benchmark_df["documents_total"] = documents_total
    benchmark_df["documents_sent_to_llm"] = documents_sent_to_llm
    benchmark_df["llm_calls_avoided"] = llm_calls_avoided
    benchmark_df["routing_rate"] = routing_rate
    benchmark_df["local_processing_rate"] = (
        local_processing_rate
    )

    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    reasoning_tokens = []
    cached_tokens = []
    provider_costs = []
    requests_attempted = []
    requests_successful = []

    for _, row in benchmark_df.iterrows():

        provider = row.get("provider", "")

        usage = get_provider_usage(
            classification_metadata,
            provider,
        )

        requests_attempted.append(
            usage.get("requests_attempted", 0)
        )

        requests_successful.append(
            usage.get("requests_successful", 0)
        )

        prompt_tokens.append(
            usage.get("prompt_tokens", 0)
        )

        completion_tokens.append(
            usage.get("completion_tokens", 0)
        )

        total_tokens.append(
            usage.get("total_tokens", 0)
        )

        reasoning_tokens.append(
            usage.get("reasoning_tokens", 0)
        )

        cached_tokens.append(
            usage.get("cached_tokens", 0)
        )

        provider_costs.append(
            usage.get(
                "provider_reported_cost",
                0.0,
            )
        )

    benchmark_df["requests_attempted"] = (
        requests_attempted
    )

    benchmark_df["requests_successful"] = (
        requests_successful
    )

    benchmark_df["prompt_tokens"] = (
        prompt_tokens
    )

    benchmark_df["completion_tokens"] = (
        completion_tokens
    )

    benchmark_df["total_tokens"] = (
        total_tokens
    )

    benchmark_df["reasoning_tokens"] = (
        reasoning_tokens
    )

    benchmark_df["cached_tokens"] = (
        cached_tokens
    )

    benchmark_df["provider_reported_cost"] = (
        provider_costs
    )

    benchmark_df["cost_per_document"] = (
        benchmark_df.apply(
            lambda row: (
                row["provider_reported_cost"]
                / row["documents_total"]
                if row["documents_total"] > 0
                else 0.0
            ),
            axis=1,
        )
    )

    benchmark_df["cost_per_llm_document"] = (
        benchmark_df.apply(
            lambda row: (
                row["provider_reported_cost"]
                / row["requests_successful"]
                if row["requests_successful"] > 0
                else 0.0
            ),
            axis=1,
        )
    )

    benchmark_df["tokens_per_llm_document"] = (
        benchmark_df.apply(
            lambda row: (
                row["total_tokens"]
                / row["requests_successful"]
                if row["requests_successful"] > 0
                else 0.0
            ),
            axis=1,
        )
    )

    benchmark_df["reasoning_token_ratio"] = (
        benchmark_df.apply(
            lambda row: (
                row["reasoning_tokens"]
                / row["completion_tokens"]
                if row["completion_tokens"] > 0
                else 0.0
            ),
            axis=1,
        )
    )

    return benchmark_df