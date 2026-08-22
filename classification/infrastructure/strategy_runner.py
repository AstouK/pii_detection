"""
Strategy execution layer.

Responsibilities:
- Execute one classification strategy
- Apply final prediction logic
- Add output metadata
- Save strategy-specific results

This module is strategy-aware and model-agnostic.

Examples:
- rule_based
- rule_plus_qwen
- rule_plus_gpt4o_mini
- bert_distilbert
- rule_plus_distilbert
"""

from pathlib import Path
from time import perf_counter
import logging

import pandas as pd

from classification.config import (
    get_strategy_config,
)

from classification.review.llm_reviewer import run_llm
from classification.infrastructure.metadata import (
    compute_final_prediction,
    add_output_metadata,
)
from classification.infrastructure.outputs import (
    build_output_file,
)
from classification.infrastructure.runtime import (
    compute_strategy_usage_summary,
)

logger = logging.getLogger(__name__)

def run_strategy_pipeline(
    base_df: pd.DataFrame,
    strategy: str,
    run_dir: Path,
    run_id: str,
) -> tuple[Path, float, dict]:
    """
    Execute a single classification strategy and save results.

    Args:
        base_df: DataFrame after Sweep 1.
        strategy: LLM strategy to use.
        run_dir: Directory for this pipeline run.
        run_id: Shared run identifier.

    Returns:
        Tuple of saved output path and strategy runtime in seconds.
    """

    strategy = strategy.lower().strip()

    strategy_config = get_strategy_config(
        strategy
    )

    #runner = strategy_config["runner"] (cna be used for dispatch later)

    logger.info("Starting strategy %s", strategy)

    strategy_start = perf_counter()

    df_strategy = base_df.copy(deep=True)

    if strategy == "rule_based":

        df_strategy = compute_final_prediction(
            df_strategy
        )
    elif strategy in {
        "rule_plus_qwen",
        "rule_plus_gpt4o_mini",
    }:
        model_id = strategy_config["model"]

        df_strategy = run_llm(
            df_strategy,
            model_id=model_id,
        )

        df_strategy = compute_final_prediction(
            df_strategy
        )
    #add future strategies before this block
    else:
        raise NotImplementedError(
            f"Strategy '{strategy}' is not yet implemented."
        )

    
    df_strategy = add_output_metadata(
        df=df_strategy,
        strategy=strategy,
        run_id=run_id,
    )

    output_file = build_output_file(
        strategy=strategy,
        run_dir=run_dir,
    )

    df_strategy.to_csv(
        output_file,
        index=False,
    )

    strategy_runtime_seconds = round(perf_counter() - strategy_start, 4)

    logger.info(
        "Completed strategy '%s' in %.4f seconds. Results saved to %s",
        strategy,
        strategy_runtime_seconds,
        output_file,
    )

    usage_summary = compute_strategy_usage_summary(df_strategy)

    return output_file, strategy_runtime_seconds, usage_summary