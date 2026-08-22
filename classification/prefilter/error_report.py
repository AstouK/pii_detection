"""
Error analysis sliced by document type, difficulty and challenge category.

The evaluation module already produces TP/FP/FN/TN splits and grouped error
summaries. This adds the two things it does not: the slices are aggregated into
rates rather than raw counts, and every slice reports the routing cost next to
the error rate.

That pairing is the point. A slice where the pre-filter is accurate but routes
90% of documents is as much of a problem as a slice where it is wrong — the
first costs money, the second costs compliance — and only one number tells you
which. The output is direct input for Sonja on which document types and
challenge categories to over-sample when the dataset is scaled up.

Run:

    python -m classification.prefilter.error_report --run-id 20260822_231500
    python -m classification.prefilter.error_report          # latest run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from config.logging_config import setup_logging

from classification.prefilter.config import (
    BINARY_LABEL_COL,
    CLASSIFICATION_RUNS_DIR,
    ENTITY_LABELS,
    REPORTS_DIR,
    ROUTED_STRATEGY,
)
from classification.prefilter.data import to_bool_series

setup_logging()
logger = logging.getLogger(__name__)

#: Slices requested by the brief, plus two that turned out to matter on the
#: pilot: `edge_case` and `primary_pii_type`.
SLICE_COLUMNS = [
    "document_type",
    "difficulty",
    "challenge_category",
    "scenario_type",
    "edge_case",
    "primary_pii_type",
]


def latest_run_dir() -> Path:
    """
    Most recent classification run directory.
    """

    if not CLASSIFICATION_RUNS_DIR.exists():
        raise FileNotFoundError(
            f"No classification runs found under {CLASSIFICATION_RUNS_DIR}. "
            "Run: python -m classification.prefilter.predict"
        )

    run_dirs = sorted(
        (path for path in CLASSIFICATION_RUNS_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )

    if not run_dirs:
        raise FileNotFoundError(
            f"No classification runs found under {CLASSIFICATION_RUNS_DIR}."
        )

    return run_dirs[-1]


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the boolean columns the slicing works on.
    """

    annotated = df.copy()

    annotated["_truth"] = to_bool_series(annotated[BINARY_LABEL_COL])
    annotated["_pred"] = to_bool_series(annotated["predicted_pii"])

    if "routed_to_llm" in annotated.columns:
        annotated["_routed"] = to_bool_series(annotated["routed_to_llm"])
    else:
        annotated["_routed"] = False

    # A dropped positive is the only error this stage cannot recover from: it
    # is a positive document the router decided locally and got wrong.
    annotated["_missed_positive"] = (
        annotated["_truth"] & ~annotated["_pred"] & ~annotated["_routed"]
    )
    annotated["_unreviewed_fp"] = (
        ~annotated["_truth"] & annotated["_pred"] & ~annotated["_routed"]
    )

    annotated["_error_type"] = "true_negative"
    annotated.loc[annotated["_truth"] & annotated["_pred"], "_error_type"] = (
        "true_positive"
    )
    annotated.loc[~annotated["_truth"] & annotated["_pred"], "_error_type"] = (
        "false_positive"
    )
    annotated.loc[annotated["_truth"] & ~annotated["_pred"], "_error_type"] = (
        "false_negative"
    )

    return annotated


def slice_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Per-value error and routing rates for one slicing column.
    """

    if column not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby(column)
        .agg(
            n=("_truth", "size"),
            positives=("_truth", "sum"),
            correct=("_error_type", lambda s: s.isin(
                ["true_positive", "true_negative"]).sum()),
            false_positives=("_error_type", lambda s: (s == "false_positive").sum()),
            false_negatives=("_error_type", lambda s: (s == "false_negative").sum()),
            missed_positives=("_missed_positive", "sum"),
            unreviewed_fp=("_unreviewed_fp", "sum"),
            routed=("_routed", "sum"),
            mean_probability=("pii_probability", "mean"),
        )
        .reset_index()
    )

    grouped["accuracy"] = (grouped["correct"] / grouped["n"]).round(4)
    grouped["routing_rate"] = (grouped["routed"] / grouped["n"]).round(4)
    grouped["recall"] = (
        (grouped["positives"] - grouped["missed_positives"])
        / grouped["positives"].where(grouped["positives"] > 0)
    ).round(4)
    grouped["mean_probability"] = grouped["mean_probability"].round(4)

    return grouped.sort_values("n", ascending=False).reset_index(drop=True)


def entity_slice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-entity agreement between the multi-label head and ground truth.

    Compares ``predicted_<ENTITY>_yes_no`` against ``<ENTITY>_yes_no``.
    """

    rows = []

    for label in ENTITY_LABELS:
        truth_col = f"{label}_yes_no"
        pred_col = f"predicted_{label}_yes_no"

        if truth_col not in df.columns or pred_col not in df.columns:
            continue

        truth = to_bool_series(df[truth_col])
        pred = to_bool_series(df[pred_col])

        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        tn = int((~truth & ~pred).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        rows.append(
            {
                "entity": label,
                "support": tp + fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            }
        )

    return pd.DataFrame(rows)


def build_report(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    All slices for one prediction file.
    """

    annotated = annotate(df)

    report = {
        column: slice_summary(annotated, column)
        for column in SLICE_COLUMNS
        if column in annotated.columns
    }

    report["entity"] = entity_slice(annotated)

    report["errors"] = annotated[
        annotated["_error_type"].isin(["false_positive", "false_negative"])
    ][
        [
            column
            for column in [
                "document_id",
                "document_type",
                "difficulty",
                "challenge_category",
                "primary_pii_type",
                BINARY_LABEL_COL,
                "predicted_pii",
                "pii_probability",
                "routed_to_llm",
                "_error_type",
            ]
            if column in annotated.columns
        ]
    ].rename(columns={"_error_type": "error_type"})

    return {name: frame for name, frame in report.items() if not frame.empty}


def print_report(report: dict[str, pd.DataFrame], source: Path) -> None:
    print("\n" + "=" * 78)
    print(f"PRE-FILTER ERROR ANALYSIS — {source}")
    print("=" * 78)

    for name, frame in report.items():
        if name == "errors":
            continue

        print(f"\n▶ by {name}")

        if name == "entity":
            print(frame.to_string(index=False))
            continue

        columns = [
            name,
            "n",
            "positives",
            "accuracy",
            "recall",
            "routing_rate",
            "missed_positives",
            "unreviewed_fp",
            "mean_probability",
        ]
        print(frame[[c for c in columns if c in frame.columns]].to_string(index=False))

    errors = report.get("errors")

    if errors is not None and not errors.empty:
        print(f"\n▶ individual errors ({len(errors)})")
        print(errors.to_string(index=False))
    else:
        print("\n▶ no false positives or false negatives on this split")

    print("\n" + "=" * 78 + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Slice pre-filter errors by document type and difficulty."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Classification run id. Defaults to the latest run.",
    )
    parser.add_argument(
        "--strategy",
        default=ROUTED_STRATEGY,
        help=f"Prediction file to analyse. Defaults to {ROUTED_STRATEGY}.",
    )
    parser.add_argument("--output-dir", default=str(REPORTS_DIR))

    args = parser.parse_args(argv)

    run_dir = (
        CLASSIFICATION_RUNS_DIR / args.run_id if args.run_id else latest_run_dir()
    )

    prediction_file = run_dir / f"{args.strategy}.csv"

    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

    df = pd.read_csv(prediction_file)
    report = build_report(df)

    print_report(report, prediction_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in report.items():
        frame.to_csv(output_dir / f"error_by_{name}.csv", index=False)

    (output_dir / "error_report_source.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "strategy": args.strategy,
                "prediction_file": str(prediction_file),
                "n_documents": int(len(df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Error-analysis slices written to %s", output_dir)


if __name__ == "__main__":
    main()
