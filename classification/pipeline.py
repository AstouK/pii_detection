"""
Production classification pipeline.

Responsibilities:
1. Load input dataset
2. Execute Sweep 1
3. Execute strategy-specific classification runs
4. Save outputs
5. Save run metadata

Evaluation, benchmarking, error analysis, and MLflow tracking
belong in classification/evaluation/.
"""
import argparse
from pathlib import Path

from datetime import datetime
from time import perf_counter
import logging

from config.logging_config import setup_logging

from classification.config import (
    STRATEGIES_TO_RUN,
    validate_strategies,
    DEFAULT_INPUT_FILE,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_DATASET_VERSION,
)

from classification.infrastructure.metadata import (
    resolve_dataset_version,
)

from classification.infrastructure.io import load_input_data

from classification.sweep1 import run_sweep1

from classification.infrastructure.outputs import (
    create_run_dir,
    save_sweep1_results,
    save_run_metadata,
)

from classification.infrastructure.runtime import (
    compute_routing_metrics,
)

from classification.infrastructure.strategy_runner import (
    run_strategy_pipeline,
)



setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse classification runtime arguments.
    """

    parser = argparse.ArgumentParser(
        description="Run GDPR PII classification strategies."
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Dataset CSV to classify.",
    )

    parser.add_argument(
        "--prompt-version",
        type=str,
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt template version used by LLM strategies.",
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=STRATEGIES_TO_RUN,
        help="Classification strategies to execute.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategies = validate_strategies(args.strategies)

    df = load_input_data(input_file=args.input_file,)

    dataset_version = resolve_dataset_version(
        df=df,
        default_version=DEFAULT_DATASET_VERSION,
    )

    logger.info(
        "Dataset version: %s",
        dataset_version,
    )

    logger.info(
        "Prompt version: %s",
        args.prompt_version,
    )

    pipeline_start = perf_counter()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = create_run_dir(run_id)

    logger.info(
        "Classification pipeline started"
    )

    logger.info(
        "Run ID: %s",
        run_id,
    )

    logger.info(
        "Strategies selected: %s",
        strategies,
    )

    # --------------------------------------------------
    # Sweep 1
    # --------------------------------------------------

    logger.info(
        "Running Sweep 1"
    )

    sweep1_start = perf_counter()

    df_sweep1 = run_sweep1(df)

    sweep1_runtime_seconds = round(
        perf_counter() - sweep1_start,
        4,
    )

    logger.info(
        "Sweep 1 completed in %.4f seconds",
        sweep1_runtime_seconds,
    )

    saved_files = []

    sweep1_file = save_sweep1_results(
        df_sweep1=df_sweep1,
        run_dir=run_dir,
        run_id=run_id,
    )

    saved_files.append(
        sweep1_file
    )

    # --------------------------------------------------
    # Strategy execution
    # --------------------------------------------------

    strategy_runtime_seconds = {}

    strategy_usage = {}

    for strategy in strategies:

        (
            output_file,
            runtime_seconds,
            usage_summary,
        ) = run_strategy_pipeline(
            base_df=df_sweep1,
            strategy=strategy,
            run_dir=run_dir,
            run_id=run_id,
            dataset_version=dataset_version,
            prompt_version=args.prompt_version,
        )

        saved_files.append(
            output_file
        )

        strategy_runtime_seconds[
            strategy
        ] = runtime_seconds

        strategy_usage[
            strategy
        ] = usage_summary

    # --------------------------------------------------
    # Run Metadata
    # --------------------------------------------------

    routing_metrics = (
        compute_routing_metrics(
            df_sweep1
        )
    )

    sweep2_runtime_seconds = round(
        sum(
            strategy_runtime_seconds.values()
        ),
        4,
    )

    pipeline_runtime_seconds = round(
        perf_counter() - pipeline_start,
        4,
    )

    runtime_metrics = {
        "sweep1_runtime_seconds":
            sweep1_runtime_seconds,
        "sweep2_runtime_seconds":
            sweep2_runtime_seconds,
        "pipeline_runtime_seconds":
            pipeline_runtime_seconds,
        "strategy_runtime_seconds":
            strategy_runtime_seconds,
    }

    metadata_file = save_run_metadata(
        run_dir=run_dir,
        run_id=run_id,
        strategies=strategies,
        saved_files=saved_files,
        routing_metrics=routing_metrics,
        runtime_metrics=runtime_metrics,
        strategy_usage=strategy_usage,
        dataset_version=dataset_version,
        prompt_version=args.prompt_version,
    )

    saved_files.append(
        metadata_file
    )

    logger.info(
        "Classification pipeline completed"
    )

    logger.info(
        "Run duration: %.4f seconds",
        pipeline_runtime_seconds,
    )

    for file_path in saved_files:

        logger.info(
            "Created: %s",
            file_path,
        )


if __name__ == "__main__":
    main()