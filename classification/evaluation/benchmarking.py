"""
Benchmark summary helpers for GDPR PII detection evaluation.

This module builds a compact benchmark table from per-output metrics.csv files.
It is used to compare Sweep 1, LLM providers, and future local models.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_BENCHMARK_METRIC_NAME = "standardized"
METRICS_FILE_NAME = "metrics.csv"
BENCHMARK_FILE_NAME = "benchmark_summary.csv"


def load_metrics_file(metrics_file: Path) -> pd.DataFrame:
    """
    Load one metrics.csv file.
    """

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    return pd.read_csv(metrics_file)


def extract_benchmark_row(
    output_name: str,
    metrics_df: pd.DataFrame,
    metric_name: str = DEFAULT_BENCHMARK_METRIC_NAME,
) -> dict | None:
    """
    Extract the document-level benchmark row from a metrics dataframe.

    Default benchmark row:
        metric_group = document
        metric_name = standardized

    This means:
        sweep1.csv       -> predicted_pii = detected_pii
        qwen.csv         -> predicted_pii = final_pii
        openrouter.csv   -> predicted_pii = final_pii
        future bert.csv  -> predicted_pii = model prediction
    """

    required_cols = {"metric_group", "metric_name"}

    if not required_cols.issubset(metrics_df.columns):
        return None

    benchmark_rows = metrics_df[
        (metrics_df["metric_group"] == "document")
        & (metrics_df["metric_name"] == metric_name)
    ]

    if benchmark_rows.empty:
        return None

    row = benchmark_rows.iloc[0].to_dict()
    row["output_name"] = output_name

    return row


def build_benchmark_summary_from_metric_files(
    evaluation_run_dir: Path,
    metric_name: str = DEFAULT_BENCHMARK_METRIC_NAME,
) -> pd.DataFrame:
    """
    Build benchmark_summary.csv from all output-level metrics.csv files.

    Example input structure:
        evaluation/results/runs/<run_id>/
        ├── sweep1/metrics.csv
        ├── qwen/metrics.csv
        └── openrouter/metrics.csv
    """

    rows = []

    output_dirs = [
        path
        for path in evaluation_run_dir.iterdir()
        if path.is_dir()
    ]

    for output_dir in sorted(output_dirs, key=lambda path: path.name):
        metrics_file = output_dir / METRICS_FILE_NAME

        if not metrics_file.exists():
            continue

        metrics_df = load_metrics_file(metrics_file)

        row = extract_benchmark_row(
            output_name=output_dir.name,
            metrics_df=metrics_df,
            metric_name=metric_name,
        )

        if row is not None:
            rows.append(row)

    benchmark_df = pd.DataFrame(rows)

    if benchmark_df.empty:
        return benchmark_df

    preferred_cols = [
        "output_name",
        "run_id",
        "provider",
        "model_family",
        "model_name",
        "prediction_source",
        "prediction_stage",
        "pipeline_name",
        "metric_group",
        "metric_name",
        "label",
        "n",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "TP",
        "TN",
        "FP",
        "FN",
    ]

    existing_preferred_cols = [
        col for col in preferred_cols if col in benchmark_df.columns
    ]

    remaining_cols = [
        col for col in benchmark_df.columns if col not in existing_preferred_cols
    ]

    benchmark_df = benchmark_df[existing_preferred_cols + remaining_cols]

    return benchmark_df.sort_values(
        by=["f1", "recall", "precision"],
        ascending=False,
    ).reset_index(drop=True)


def save_benchmark_summary(
    benchmark_df: pd.DataFrame,
    evaluation_run_dir: Path,
    file_name: str = BENCHMARK_FILE_NAME,
) -> Path:
    """
    Save benchmark summary to the evaluation run directory.
    """

    output_file = evaluation_run_dir / file_name

    benchmark_df.to_csv(
        output_file,
        index=False,
    )

    return output_file