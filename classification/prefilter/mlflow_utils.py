"""
MLflow logging for pre-filter training runs.

The experiment name matches the evaluation module's
``MLFLOW_EXPERIMENT_NAME``, so a training run and the ``evaluate`` run that
scores its output land in the same experiment and can be compared directly.

Logging is opt-in (``--log-mlflow``) and never fatal: a broken tracking backend
must not lose a finished training run.
"""

from __future__ import annotations

import logging
from pathlib import Path

from classification.prefilter.config import MLFLOW_EXPERIMENT_NAME

logger = logging.getLogger(__name__)

#: Config fields worth logging as MLflow params. The rest stay in config.json,
#: which is logged as an artifact anyway.
PARAM_FIELDS = [
    "model_name",
    "pretrained_dir",
    "data_file",
    "split_mode",
    "max_length",
    "epochs",
    "batch_size",
    "learning_rate",
    "head_learning_rate",
    "weight_decay",
    "warmup_ratio",
    "multilabel_loss_weight",
    "max_pos_weight",
    "recall_target",
    "precision_target",
    "standalone_threshold",
    "entity_threshold",
    "seed",
]


def _flatten_metrics(summary: dict) -> dict:
    """
    Pull the numeric metrics worth tracking out of a training summary.
    """

    metrics: dict[str, float] = {}

    plain = summary.get("validation_binary_at_0_5", {})
    for key in ("accuracy", "precision", "recall", "f1", "TP", "TN", "FP", "FN"):
        if key in plain:
            metrics[f"val_{key.lower()}"] = float(plain[key])

    if summary.get("validation_pr_auc") is not None:
        metrics["val_pr_auc"] = float(summary["validation_pr_auc"])

    calibration = summary.get("calibration", {})
    for key in (
        "t_low",
        "t_high",
        "routed_fraction",
        "llm_call_reduction",
        "llm_calls_avoided",
        "prefilter_recall",
        "missed_positives",
        "auto_yes_precision",
        "oracle_f1",
        "conservative_precision",
        "conservative_recall",
        "conservative_f1",
    ):
        value = calibration.get(key)
        if value is not None:
            metrics[f"route_{key}"] = float(value)

    for key in ("training_seconds", "parameters_total", "model_size_mb"):
        if summary.get(key) is not None:
            metrics[key] = float(summary[key])

    for row in summary.get("validation_entity_metrics", []):
        entity = row["entity"].lower()
        metrics[f"entity_{entity}_precision"] = float(row["precision"])
        metrics[f"entity_{entity}_recall"] = float(row["recall"])
        metrics[f"entity_{entity}_f1"] = float(row["f1"])

    return metrics


def log_training_run(
    summary: dict,
    artifacts_dir: Path,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> str | None:
    """
    Log one training run: params, metrics, per-epoch history and artifacts.

    Returns the MLflow run id, or ``None`` if logging was skipped.
    """

    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow is not installed; skipping training-run logging.")
        return None

    config = summary.get("config", {})

    try:
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(
            run_name=f"prefilter_{summary.get('run_name', 'run')}"
        ) as run:
            mlflow.log_params(
                {
                    field: str(config.get(field, ""))
                    for field in PARAM_FIELDS
                    if field in config
                }
            )
            mlflow.log_params(
                {
                    "resolved_split_mode": str(summary.get("split_mode", "")),
                    "selected_epoch": str(summary.get("selected_epoch", "")),
                    "device": str(summary.get("device", "")),
                    "routing_feasible": str(
                        summary.get("calibration", {}).get("feasible", "")
                    ),
                }
            )

            mlflow.log_metrics(_flatten_metrics(summary))

            history_file = Path(artifacts_dir) / "training_history.csv"

            if history_file.exists():
                import pandas as pd

                for _, row in pd.read_csv(history_file).iterrows():
                    step = int(row["epoch"])
                    for column in (
                        "train_loss",
                        "train_binary_loss",
                        "train_entity_loss",
                        "val_f1",
                        "val_pr_auc",
                        "val_routed_fraction",
                    ):
                        if column in row and row[column] == row[column]:
                            mlflow.log_metric(column, float(row[column]), step=step)

            # Weights are excluded deliberately: a 268 MB blob per run fills the
            # tracking store fast and the checkpoint already lives in artifacts/.
            for pattern in ("*.csv", "*.json", "*.png"):
                for path in Path(artifacts_dir).glob(pattern):
                    mlflow.log_artifact(str(path))

            return run.info.run_id

    except Exception as error:  # noqa: BLE001 - logging must never kill a run
        logger.warning("MLflow logging failed (%s); continuing.", error)
        return None
