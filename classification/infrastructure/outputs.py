"""
Result export utilities.

Responsibilities:
- Create run directories
- Save classification outputs
- Save Sweep 1 outputs
- Save run metadata

Evaluation reports should be saved by the
evaluation module, not here.
"""

from pathlib import Path
import json
import logging

import pandas as pd

from classification.infrastructure.metadata import(
    add_sweep1_metadata,
)

from classification.config import (
    RESULTS_DIR,
    DEFAULT_PIPELINE_NAME,
)

logger = logging.getLogger(__name__)

def create_run_dir(timestamp: str) -> Path:
    """
    Create a run-specific output directory.

    Example:
        classification/results/runs/20260810_153000/
    """

    run_dir = RESULTS_DIR / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir

def build_output_file(
    provider: str,
    run_dir: Path,
) -> Path:
    """
    Build provider-specific output file inside a run directory.
    """

    return run_dir / f"{provider}.csv"


def save_sweep1_results(
    df_sweep1: pd.DataFrame,
    run_dir: Path,
    run_id: str,
) -> Path:
    """
    Save Sweep 1 baseline results.
    """

    output_file = run_dir / "sweep1.csv"

    df_sweep1_output = add_sweep1_metadata(
        df=df_sweep1,
        run_id=run_id,
    )

    df_sweep1_output.to_csv(
        output_file,
        index=False,
    )

    logger.info("Sweep 1 results saved to %s", output_file)

    return output_file


def save_run_metadata(
    run_dir: Path,
    run_id: str,
    providers: list[str],
    saved_files: list[Path],
    routing_metrics: dict,
    runtime_metrics: dict,
    provider_usage: dict,
) -> Path:
    """
    Save run-level classification metadata.
    """

    metadata_file = run_dir / "run_metadata.json"

    metadata = {
        "run_id": run_id,
        "pipeline_name": DEFAULT_PIPELINE_NAME,
        "providers": providers,
        "saved_files": [str(path) for path in saved_files],
        **routing_metrics,
        **runtime_metrics,
        "provider_usage": provider_usage,
    }

    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    logger.info("Run metadata saved to %s", metadata_file)

    return metadata_file