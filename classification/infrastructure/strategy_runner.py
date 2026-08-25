"""
Strategy execution layer.

Responsibilities:
- Execute one classification strategy
- Apply final prediction logic
- Add output metadata
- Save strategy-specific results
- Return runtime and operational usage

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
from classification.infrastructure.metadata import (
    add_output_metadata,
    compute_final_prediction,
)
from classification.infrastructure.outputs import (
    build_output_file,
)
from classification.infrastructure.runtime import (
    compute_strategy_usage_summary,
)
from classification.review.llm_reviewer import (
    run_llm,
)


logger = logging.getLogger(__name__)


def run_strategy_pipeline(
    base_df: pd.DataFrame,
    strategy: str,
    run_dir: Path,
    run_id: str,
    dataset_version: str,
    prompt_version: str,
) -> tuple[Path, float, dict]:
    """
    Execute one classification strategy and save its output.

    Args:
        base_df:
            DataFrame produced by Sweep 1.

        strategy:
            Registered classification strategy to execute.

        run_dir:
            Directory for the current classification run.

        run_id:
            Shared classification run identifier.

        dataset_version:
            Version of the dataset used for the run.

        prompt_version:
            Prompt version used by prompt-based strategies.

    Returns:
        Tuple containing:
        - saved strategy output path
        - strategy runtime in seconds
        - operational usage summary
    """

    strategy = strategy.lower().strip()

    strategy_config = get_strategy_config(
        strategy
    )

    runner = strategy_config["runner"]

    logger.info(
        "Starting strategy '%s' with runner '%s'.",
        strategy,
        runner,
    )

    strategy_start = perf_counter()

    df_strategy = base_df.copy(deep=True)

    # Ensure the explicit run-level dataset version is present before
    # metadata is resolved and attached.
    df_strategy["dataset_version"] = dataset_version

    if runner == "rule_based":
        df_strategy = compute_final_prediction(
            df_strategy
        )

    elif runner == "rule_plus_llm":
        model_id = strategy_config["model"]

        df_strategy = run_llm(
            df=df_strategy,
            model_id=model_id,
            prompt_version=prompt_version,
        )

        df_strategy = compute_final_prediction(
            df_strategy
        )

    elif runner == "bert":
        raise NotImplementedError(
            f"BERT strategy '{strategy}' is not yet implemented."
        )

    elif runner == "rule_plus_bert":
        raise NotImplementedError(
            f"Rule plus BERT strategy '{strategy}' "
            "is not yet implemented."
        )

    elif runner == "hybrid":
        raise NotImplementedError(
            f"Hybrid strategy '{strategy}' is not yet implemented."
        )

    else:
        raise ValueError(
            f"Unsupported runner '{runner}' configured "
            f"for strategy '{strategy}'."
        )

    df_strategy = add_output_metadata(
        df=df_strategy,
        strategy=strategy,
        run_id=run_id,
        prompt_version=prompt_version,
    )

    usage_summary = compute_strategy_usage_summary(
        df_strategy
    )

    output_file = build_output_file(
        strategy=strategy,
        run_dir=run_dir,
    )

    df_strategy.to_csv(
        output_file,
        index=False,
    )

    strategy_runtime_seconds = round(
        perf_counter() - strategy_start,
        4,
    )

    logger.info(
        "Completed strategy '%s' in %.4f seconds. "
        "Results saved to %s",
        strategy,
        strategy_runtime_seconds,
        output_file,
    )

    return (
        output_file,
        strategy_runtime_seconds,
        usage_summary,
    )