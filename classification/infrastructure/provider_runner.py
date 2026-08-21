"""
Provider execution layer.

Responsibilities:
- Execute Sweep 2 for one provider
- Apply final prediction logic
- Add output metadata
- Save provider-specific results

This module is provider-agnostic and supports:
- OpenRouter
- Qwen
- Future local models
"""

from pathlib import Path
from time import perf_counter
import logging

import pandas as pd

from classification.review.llm_reviewer import run_llm
from classification.infrastructure.metadata import (
    compute_final_prediction,
    add_output_metadata,
)
from classification.infrastructure.outputs import (
    build_output_file,
)
from classification.infrastructure.runtime import (
    compute_llm_usage_summary,
)

logger = logging.getLogger(__name__)

def run_provider_pipeline(
    base_df: pd.DataFrame,
    provider: str,
    run_dir: Path,
    run_id: str,
) -> tuple[Path, float, dict]:
    """
    Run Sweep 2 for a single provider and save provider-specific results.

    Args:
        base_df: DataFrame after Sweep 1.
        provider: LLM provider to use.
        run_dir: Directory for this pipeline run.
        run_id: Shared run identifier.

    Returns:
        Tuple of saved output path and provider runtime in seconds.
    """

    provider = provider.lower().strip()

    logger.info("Starting Sweep 2 for provider: %s", provider)

    provider_start = perf_counter()

    df_provider = base_df.copy(deep=True)

    df_provider = run_llm(
        df_provider,
        provider=provider,
    )

    df_provider = compute_final_prediction(df_provider)

    df_provider = add_output_metadata(
        df=df_provider,
        provider=provider,
        run_id=run_id,
    )

    output_file = build_output_file(
        provider=provider,
        run_dir=run_dir,
    )

    df_provider.to_csv(
        output_file,
        index=False,
    )

    provider_runtime_seconds = round(perf_counter() - provider_start, 4)

    logger.info(
        "Completed provider '%s' in %.4f seconds. Results saved to %s",
        provider,
        provider_runtime_seconds,
        output_file,
    )

    usage_summary = compute_llm_usage_summary(df_provider)

    return output_file, provider_runtime_seconds, usage_summary