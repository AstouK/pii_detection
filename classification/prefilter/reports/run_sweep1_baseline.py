"""
Measure the Sweep 1 baseline on the same dataset and split as the pre-filter.

Why this exists: the 0.9833 recall the whole work package is measured against
came from the project brief, not from a run anyone could point at. It is not
known which dataset or which Sweep 1 configuration produced it, so "the
pre-filter's test recall is 0.9643, below the 0.9833 baseline" compares two
numbers that were never measured on the same data.

This script runs Sweep 1 (regex + Presidio + spaCy) over Sonja's 1,400-row
dataset and writes an evaluation-compatible `rule_based.csv` into the same
classification run directory the pre-filter already wrote to. `evaluate` then
scores all strategies side by side, on identical documents and an identical
split, and the benchmark summary carries the real comparison.

Run:

    python classification/prefilter/reports/run_sweep1_baseline.py --run-id 20260823_013600
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from config.logging_config import setup_logging

from classification.infrastructure.metadata import add_sweep1_metadata
from classification.prefilter.config import (
    BINARY_LABEL_COL,
    CLASSIFICATION_RUNS_DIR,
    CONTEXT_COLS,
    DOCUMENT_ID_COL,
    PreFilterConfig,
    entity_label_columns,
)
from classification.prefilter.data import load_dataset, resolve_splits
from classification.sweep1 import run_sweep1

setup_logging()
logger = logging.getLogger(__name__)

DEFAULT_DATA_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "external" /
    "synthetic_dataset_1400.csv"
)

#: Sweep 1 fields worth carrying into the output. `per_type_conf` is the one the
#: evaluation actually needs — it derives per-entity metrics from it.
SWEEP1_COLUMNS = [
    "detected_pii",
    "detected_any_pii",
    "detected_categories",
    "strong_pii_categories",
    "potential_pii_categories",
    "has_person_hint",
    "needs_llm_review",
    "route",
    "routing_reason",
    "per_type_conf",
]


def build_output(
    source: pd.DataFrame,
    sweep1: pd.DataFrame,
    run_id: str,
    inference_ms: float,
) -> pd.DataFrame:
    """
    Assemble the evaluation-compatible CSV for Sweep 1.

    Ground truth is carried through for the same reason the pre-filter carries
    it: the evaluation reads it out of the prediction file rather than
    re-joining the dataset.
    """

    output = pd.DataFrame()

    output[DOCUMENT_ID_COL] = source[DOCUMENT_ID_COL].values
    output[BINARY_LABEL_COL] = source[BINARY_LABEL_COL].values

    for column in entity_label_columns():
        output[column] = source[column].values

    for column in CONTEXT_COLS:
        if column in source.columns:
            output[column] = source[column].values

    for column in SWEEP1_COLUMNS:
        if column in sweep1.columns:
            output[column] = sweep1[column].values

    output["inference_ms"] = round(inference_ms, 4)

    # add_sweep1_metadata sets run_id, strategy, provider, model_family,
    # model_name, prediction_stage, pipeline_name, prediction_source, and
    # derives predicted_pii from detected_pii.
    return add_sweep1_metadata(df=output, run_id=run_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Sweep 1 and write an evaluation-compatible baseline."
    )
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument(
        "--run-id",
        required=True,
        help="Classification run to write into, so evaluate compares strategies.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to score. 'all' scores every row. Defaults to test.",
    )
    parser.add_argument(
        "--strategy-name",
        default="rule_based",
        help="Output filename stem, i.e. the strategy the evaluation reports.",
    )

    args = parser.parse_args(argv)

    # The same config the pre-filter used, so the split resolves identically.
    config = PreFilterConfig(data_file=args.data_file)

    df = load_dataset(args.data_file)
    df, resolved_mode = resolve_splits(df, config)

    logger.info("Split mode resolved to: %s", resolved_mode)

    if args.split == "all":
        frame = df.reset_index(drop=True)
    else:
        frame = df[df["split"] == args.split].reset_index(drop=True)

    if frame.empty:
        raise ValueError(f"Split '{args.split}' is empty.")

    logger.info(
        "Running Sweep 1 over %s documents (split=%s)", len(frame), args.split
    )

    started = time.perf_counter()
    sweep1 = run_sweep1(frame)
    elapsed = time.perf_counter() - started

    inference_ms = 1000.0 * elapsed / max(len(frame), 1)

    logger.info(
        "Sweep 1 finished in %.1fs (%.1f ms/document)", elapsed, inference_ms
    )

    output = build_output(
        source=frame,
        sweep1=sweep1,
        run_id=args.run_id,
        inference_ms=inference_ms,
    )

    run_dir = CLASSIFICATION_RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    output_file = run_dir / f"{args.strategy_name}.csv"
    output.to_csv(output_file, index=False)

    logger.info("Wrote %s", output_file)

    _print_summary(output, args.split, inference_ms, elapsed)


def _print_summary(
    output: pd.DataFrame,
    split: str,
    inference_ms: float,
    elapsed: float,
) -> None:
    """
    The numbers that matter, computed here so the run is readable without
    waiting for `evaluate`.
    """

    from classification.prefilter.data import to_bool_series

    truth = to_bool_series(output[BINARY_LABEL_COL])
    predicted = to_bool_series(output["predicted_pii"])
    routed = to_bool_series(output["needs_llm_review"])

    tp = int((truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    fp = int((~truth & predicted).sum())
    tn = int((~truth & ~predicted).sum())

    n = len(output)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    print("\n" + "=" * 64)
    print(f"SWEEP 1 BASELINE — split={split}")
    print("=" * 64)
    print(f"  documents          : {n} ({int(truth.sum())} positive)")
    print(f"  accuracy           : {(tp + tn) / n:.4f}")
    print(f"  precision          : {precision:.4f}")
    print(f"  recall             : {recall:.4f}")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  routed to LLM      : {int(routed.sum())} "
          f"({100 * routed.mean():.1f}%)")
    print(f"  runtime            : {elapsed:.1f}s "
          f"({inference_ms:.1f} ms/document)")

    if "route" in output.columns:
        print("\n  route distribution:")
        for route, count in output["route"].value_counts().items():
            print(f"    {route:<20} {count:>5}  ({100 * count / n:.1f}%)")

    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
