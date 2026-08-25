"""
Evaluation pipeline for GDPR PII detection outputs.

This pipeline evaluates saved classification run outputs.

It does not rerun production classification code. Instead, it:
1. Loads a classification run from classification/results/runs/
2. Loads prediction outputs such as rule_based.csv, rule_plus_qwen.csv
3. Computes metrics for each output
4. Runs row-level error analysis for each output
5. Saves evaluation artifacts under classification/evaluation/results/runs/
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.logging_config import setup_logging

from classification.evaluation.error_analysis import run_error_analysis

from classification.evaluation.metrics import compute_all_metrics

from classification.evaluation.reporting import (
    print_metric_report,
    save_metrics,
)

from classification.evaluation.io import (
    get_classification_run_dir,
    load_prediction_outputs,
    load_run_metadata,
    create_evaluation_run_dir,
    create_output_eval_dir,
    save_error_analysis_outputs,
    save_evaluation_metadata,
)

from classification.evaluation.benchmarking import (
    build_benchmark_summary_from_metric_files,
    save_benchmark_summary,
)

from classification.evaluation.config import (
    DEFAULT_PREDICTION_COL,
    ENABLE_MLFLOW_LOGGING,
    get_evaluation_metadata,
)

from classification.evaluation.mlflow_logger import log_benchmark_summary_to_mlflow

from classification.evaluation.cost_analysis import (
    create_cost_summary,
)


# ── Logging ─────────────────────────────────────────

setup_logging()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────

def evaluate_prediction_output(
    output_name: str,
    df: pd.DataFrame,
    evaluation_run_dir: Path,
) -> list:
    """
    Compute metrics and run error analysis for one prediction output.

    Example output_name values:
        sweep1
        rule_based
        rule_plus_qwen
        rule_plus_gpt4o_mini
    """

    logger.info(
        "Evaluating output '%s'",
        output_name,
    )

    output_eval_dir = create_output_eval_dir(
        evaluation_run_dir=evaluation_run_dir,
        output_name=output_name,
    )

    saved_files = []

    # ── Metrics ─────────────────────────────────────
    metrics = compute_all_metrics(df)

    print_metric_report(
        metrics=metrics,
        title=f"Evaluation metrics: {output_name}",
    )

    metrics_file = save_metrics(
        metrics=metrics,
        output_dir=output_eval_dir,
    )

    saved_files.append(metrics_file)

    logger.info(
        "Saved metrics for output '%s' to %s",
        output_name,
        metrics_file,
    )

    # ── Error analysis ──────────────────────────────
    error_outputs = run_error_analysis(
        df=df,
    )

    error_files = save_error_analysis_outputs(
        outputs=error_outputs,
        output_dir=output_eval_dir,
    )

    saved_files.extend(error_files)

    logger.info(
        "Saved %s error-analysis artifacts for output '%s' to %s",
        len(error_files),
        output_name,
        output_eval_dir,
    )

    return saved_files


def run_evaluation(
    run_id: str | None = None,
    log_mlflow: bool = ENABLE_MLFLOW_LOGGING,
) -> dict:
    """
    Evaluate a saved classification run.

    Args:
        run_id:
            Classification run ID to evaluate.
            If None, the latest classification run is evaluated.

            prediction_col controls error analysis.

    Metrics are computed for all available prediction columns present
    in the output file.

    Returns:
        Evaluation metadata dictionary.
    """

    evaluation_started_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    classification_run_dir = get_classification_run_dir(run_id)
    classification_run_id = classification_run_dir.name

    logger.info("Evaluation pipeline started")
    logger.info("Classification run selected: %s", classification_run_id)
    logger.info("Classification run directory: %s", classification_run_dir)

    classification_metadata = load_run_metadata(classification_run_dir)

    prediction_outputs = load_prediction_outputs(classification_run_dir)

    logger.info(
        "Loaded prediction outputs: %s",
        list(prediction_outputs.keys()),
    )

    evaluation_run_dir = create_evaluation_run_dir(
        classification_run_id=classification_run_id,
    )

    saved_files = []

    for output_name, df in prediction_outputs.items():
        output_saved_files = evaluate_prediction_output(
            output_name=output_name,
            df=df,
            evaluation_run_dir=evaluation_run_dir,
        )

        saved_files.extend(output_saved_files)

    benchmark_summary = build_benchmark_summary_from_metric_files(
        evaluation_run_dir=evaluation_run_dir,
    )

    benchmark_summary = create_cost_summary(
        benchmark_df=benchmark_summary,
        classification_metadata=classification_metadata,
    )

    benchmark_file = save_benchmark_summary(
        benchmark_df=benchmark_summary,
        evaluation_run_dir=evaluation_run_dir,
    )

    saved_files.append(benchmark_file)

    logger.info("Benchmark summary saved to %s", benchmark_file)

    evaluation_metadata = {
        **get_evaluation_metadata(),
        "evaluation_started_at": evaluation_started_at,
        "classification_run_id": classification_run_id,
        "classification_run_dir": str(classification_run_dir),
        "evaluation_run_dir": str(evaluation_run_dir),
        "evaluated_outputs": list(prediction_outputs.keys()),
        "benchmark_summary_file": str(benchmark_file),
        "classification_metadata": classification_metadata,
        "saved_files": [str(path) for path in saved_files],
    }

    metadata_file = save_evaluation_metadata(
        evaluation_run_dir=evaluation_run_dir,
        metadata=evaluation_metadata,
    )

    if log_mlflow:
        mlflow_run_ids = log_benchmark_summary_to_mlflow(
            benchmark_summary_file=benchmark_file,
            evaluation_run_dir=evaluation_run_dir,
            evaluation_metadata=evaluation_metadata,
        )

        evaluation_metadata["mlflow_run_ids"] = mlflow_run_ids

        metadata_file = save_evaluation_metadata(
            evaluation_run_dir=evaluation_run_dir,
            metadata=evaluation_metadata,
        )

        logger.info("Logged evaluation results to MLflow runs: %s", mlflow_run_ids)

    saved_files.append(metadata_file)

    logger.info("Evaluation metadata saved to %s", metadata_file)
    logger.info("Evaluation pipeline completed")

    for file_path in saved_files:
        logger.info("Evaluation artifact created: %s", file_path)

    return evaluation_metadata


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate saved GDPR PII classification run outputs."
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Classification run ID to evaluate. "
            "If omitted, the latest classification run is used."
        ),
    )

    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log evaluation results to MLflow.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the evaluation pipeline.
    """

    args = parse_args()

    run_evaluation(
        run_id=args.run_id,
        log_mlflow=args.log_mlflow,
    )


if __name__ == "__main__":
    main()