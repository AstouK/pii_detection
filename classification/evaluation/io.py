"""
Evaluation I/O helpers.

This module handles reading classification outputs and saving evaluation
artifacts. It keeps filesystem logic out of evaluation orchestration code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from classification.config1 import RESULTS_DIR as CLASSIFICATION_RESULTS_DIR
from classification.evaluation.config import (
    RUNS_DIR as EVALUATION_RUNS_DIR,
    ERROR_ANALYSIS_OUTPUT_FILES,
    ensure_evaluation_directories,
)


# ─────────────────────────────────────────────────────────────
# Classification run discovery
# ─────────────────────────────────────────────────────────────

def get_classification_runs_dir() -> Path:
    """
    Return the directory containing classification run outputs.
    """

    return CLASSIFICATION_RESULTS_DIR / "runs"


def list_classification_runs() -> list:
    """
    List available classification run directories sorted by name.

    Assumes run directories are named with sortable run IDs, e.g.
    20260810_153000.
    """

    runs_dir = get_classification_runs_dir()

    if not runs_dir.exists():
        return []

    run_dirs = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir()
    ]

    return sorted(run_dirs, key=lambda path: path.name)


def get_latest_classification_run_dir() -> Path:
    """
    Return the latest classification run directory.
    """

    run_dirs = list_classification_runs()

    if not run_dirs:
        raise FileNotFoundError(
            "No classification run directories found. "
            "Run the classification pipeline first."
        )

    return run_dirs[-1]


def get_classification_run_dir(run_id: str | None = None) -> Path:
    """
    Return a specific classification run directory, or the latest if run_id is None.
    """

    if run_id is None:
        return get_latest_classification_run_dir()

    run_dir = get_classification_runs_dir() / run_id

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Classification run directory not found: {run_dir}"
        )

    return run_dir


# ─────────────────────────────────────────────────────────────
# Loading classification outputs
# ─────────────────────────────────────────────────────────────

def load_run_metadata(run_dir: Path) -> dict:
    """
    Load run_metadata.json from a classification run directory.
    """

    metadata_file = run_dir / "run_metadata.json"

    if not metadata_file.exists():
        return {}

    return json.loads(metadata_file.read_text(encoding="utf-8"))


def list_prediction_files(run_dir: Path) -> list:
    """
    List prediction CSV files in a classification run directory.

    Excludes non-prediction files such as run_metadata.json.
    """

    prediction_files = [
        path
        for path in run_dir.glob("*.csv")
        if path.is_file()
    ]

    return sorted(prediction_files, key=lambda path: path.name)


def load_prediction_file(prediction_file: Path) -> pd.DataFrame:
    """
    Load one prediction CSV file.
    """

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file}"
        )

    return pd.read_csv(prediction_file)


def load_prediction_outputs(run_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load all prediction CSV files from a classification run directory.

    Returns a dict keyed by output name:
        sweep1
        qwen
        openrouter
        etc.
    """

    prediction_outputs = {}

    for prediction_file in list_prediction_files(run_dir):
        output_name = prediction_file.stem
        prediction_outputs[output_name] = load_prediction_file(prediction_file)

    if not prediction_outputs:
        raise FileNotFoundError(
            f"No prediction CSV files found in run directory: {run_dir}"
        )

    return prediction_outputs


# ─────────────────────────────────────────────────────────────
# Evaluation output directories
# ─────────────────────────────────────────────────────────────

def create_evaluation_run_dir(
    classification_run_id: str,
) -> Path:
    """
    Create an evaluation output directory for a classification run.

    Example:
        classification/evaluation/results/runs/20260810_153000/
    """

    ensure_evaluation_directories()

    evaluation_run_dir = EVALUATION_RUNS_DIR / classification_run_id
    evaluation_run_dir.mkdir(parents=True, exist_ok=True)

    return evaluation_run_dir


def create_output_eval_dir(
    evaluation_run_dir: Path,
    output_name: str,
) -> Path:
    """
    Create an evaluation subdirectory for one prediction output.

    Example:
        classification/evaluation/results/runs/20260810_153000/qwen/
    """

    output_eval_dir = evaluation_run_dir / output_name
    output_eval_dir.mkdir(parents=True, exist_ok=True)

    return output_eval_dir


# ─────────────────────────────────────────────────────────────
# Saving evaluation outputs
# ─────────────────────────────────────────────────────────────

def save_dataframe(
    df: pd.DataFrame,
    output_file: Path,
) -> Path:
    """
    Save a dataframe to CSV.
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_file,
        index=False,
    )

    return output_file


def save_error_analysis_outputs(
    outputs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list:
    """
    Save error-analysis output dataframes.

    Uses ERROR_ANALYSIS_OUTPUT_FILES from evaluation config.
    Unknown output keys are saved using '<key>.csv'.
    Empty dataframes are still saved to preserve artifact consistency.
    """

    saved_files = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for output_name, df in outputs.items():
        if not isinstance(df, pd.DataFrame):
            continue

        file_name = ERROR_ANALYSIS_OUTPUT_FILES.get(
            output_name,
            f"{output_name}.csv",
        )

        output_file = output_dir / file_name

        save_dataframe(
            df=df,
            output_file=output_file,
        )

        saved_files.append(output_file)

    return saved_files


def save_evaluation_metadata(
    evaluation_run_dir: Path,
    metadata: dict,
) -> Path:
    """
    Save evaluation metadata as JSON.
    """

    metadata_file = evaluation_run_dir / "evaluation_metadata.json"

    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata_file