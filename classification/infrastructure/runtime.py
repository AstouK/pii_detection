"""
Runtime and usage metrics.

Responsibilities:
- Compute routing metrics
- Compute LLM usage metrics
- Aggregate token statistics
- Aggregate cost statistics

These are operational metrics, not evaluation metrics.
"""

import pandas as pd

def compute_routing_metrics(df_sweep1: pd.DataFrame) -> dict:
    """
    Compute routing metrics from Sweep 1 output.
    """

    documents_total = len(df_sweep1)

    if "needs_llm_review" not in df_sweep1.columns:
        documents_sent_to_llm = 0
    else:
        documents_sent_to_llm = (
            df_sweep1["needs_llm_review"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    llm_calls_avoided = documents_total - documents_sent_to_llm

    routing_rate = (
        documents_sent_to_llm / documents_total
        if documents_total > 0
        else 0.0
    )

    local_processing_rate = (
        llm_calls_avoided / documents_total
        if documents_total > 0
        else 0.0
    )

    return {
        "documents_total": int(documents_total),
        "documents_sent_to_llm": int(documents_sent_to_llm),
        "llm_calls_avoided": int(llm_calls_avoided),
        "routing_rate": round(routing_rate, 4),
        "local_processing_rate": round(local_processing_rate, 4),
    }


def compute_llm_usage_summary(
    df_provider: pd.DataFrame,
) -> dict:
    """
    Aggregate per-document LLM usage for one provider.
    """

    def sum_column(column: str, cast_type):
        if column not in df_provider.columns:
            return cast_type(0)

        values = pd.to_numeric(
            df_provider[column],
            errors="coerce",
        ).fillna(0)

        return cast_type(values.sum())

    successful_requests = (
        int(
            df_provider["llm_request_success"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if "llm_request_success" in df_provider.columns
        else 0
    )

    return {
        "requests_attempted": int(
            df_provider["needs_llm_review"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "requests_successful": successful_requests,
        "prompt_tokens": sum_column(
            "llm_prompt_tokens",
            int,
        ),
        "completion_tokens": sum_column(
            "llm_completion_tokens",
            int,
        ),
        "total_tokens": sum_column(
            "llm_total_tokens",
            int,
        ),
        "reasoning_tokens": sum_column(
            "llm_reasoning_tokens",
            int,
        ),
        "cached_tokens": sum_column(
            "llm_cached_tokens",
            int,
        ),
        "provider_reported_cost": round(
            sum_column("llm_request_cost", float),
            8,
        ),
    }
