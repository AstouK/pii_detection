"""
MLflow logging helpers for GDPR PII detection evaluation.

This module logs evaluation benchmark results and artifacts to MLflow.
It is intentionally kept outside production classification code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from classification.evaluation.config import MLFLOW_EXPERIMENT_NAME

from classification.schemas.experiment_schema import (
    MLFLOW_METRIC_FIELDS,
    MLFLOW_PARAM_FIELDS,
)


def _clean_value(value: Any) -> str:
    """
    Convert a value to a safe MLflow parameter string.
    """

    if pd.isna(value):
        return ""

    return str(value)


def _extract_params(
    row: pd.Series,
    extra_params: dict | None = None,
) -> dict:
    """
    Extract configured MLflow parameters from one benchmark row.
    """

    params = {}

    for col in MLFLOW_PARAM_FIELDS:
        if col not in row.index:
            continue

        value = row[col]

        if pd.isna(value):
            continue

        params[col] = str(value)

    if extra_params:
        for key, value in extra_params.items():
            if value is None or pd.isna(value):
                continue

            params[key] = str(value)

    return params


def _extract_metrics(
    row: pd.Series,
) -> dict:
    """
    Extract configured numeric MLflow metrics from one benchmark row.
    """

    metrics = {}

    for col in MLFLOW_METRIC_FIELDS:
        if col not in row.index:
            continue

        value = row[col]

        if pd.isna(value):
            continue

        try:
            metrics[col] = float(value)
        except (TypeError, ValueError):
            continue

    return metrics


def log_benchmark_summary_to_mlflow(
    benchmark_summary_file: Path,
    evaluation_run_dir: Path,
    evaluation_metadata: dict,
) -> list:
    """
    Log benchmark rows to MLflow.

    One MLflow run is created per evaluated output (strategy based):
        rule_based
        rule_based_plus_qwen
        ...

    Each run logs:
        params: provider/model/run metadata
        metrics: accuracy, precision, recall, f1, TP/TN/FP/FN
        artifacts: output-specific evaluation files
    """

    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow is not installed. Install it with: pip install mlflow"
        ) from exc

    if not benchmark_summary_file.exists():
        raise FileNotFoundError(
            f"Benchmark summary file not found: {benchmark_summary_file}"
        )

    benchmark_df = pd.read_csv(benchmark_summary_file)

    if benchmark_df.empty:
        return []

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    mlflow_run_ids = []

    classification_run_id = evaluation_metadata.get(
        "classification_run_id",
        "",
    )

    common_params = {
        "classification_run_id": classification_run_id,
        "evaluation_version": evaluation_metadata.get("evaluation_version", ""),
    }

    for _, row in benchmark_df.iterrows():
        output_name = _clean_value(row.get("output_name", "unknown"))

        run_name = f"{classification_run_id}_{output_name}"

        output_artifact_dir = evaluation_run_dir / output_name

        with mlflow.start_run(run_name=run_name) as run:
            params = _extract_params(
                row=row,
                extra_params=common_params,
            )

            metrics = _extract_metrics(row)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            if output_artifact_dir.exists():
                mlflow.log_artifacts(
                    str(output_artifact_dir),
                    artifact_path=output_name,
                )

            mlflow.log_artifact(str(benchmark_summary_file))

            metadata_file = evaluation_run_dir / "evaluation_metadata.json"

            if metadata_file.exists():
                mlflow.log_artifact(str(metadata_file))

            mlflow_run_ids.append(run.info.run_id)

    return mlflow_run_ids