"""
Reporting helpers for GDPR PII detection evaluation.

This module prints and saves evaluation metrics.
It does not compute metrics directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from classification.evaluation.metrics import metrics_to_dataframe


# ─────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────

def print_metric_block(metric: dict) -> None:
    """
    Print one metric dictionary.
    """

    print(f"  Accuracy : {metric['accuracy']:.4f}")
    print(f"  Precision: {metric['precision']:.4f}")
    print(f"  Recall   : {metric['recall']:.4f}")
    print(f"  F1 Score : {metric['f1']:.4f}")
    print(
        f"  TP={metric['TP']}  "
        f"TN={metric['TN']}  "
        f"FP={metric['FP']}  "
        f"FN={metric['FN']}  "
        f"n={metric['n']}"
    )


def print_document_metrics(document_metrics: dict[str, dict]) -> None:
    """
    Print document-level metrics.
    """

    if not document_metrics:
        print("  No document-level metrics found.")
        return

    for metric_name, metric in document_metrics.items():
        print(f"\n▶ {metric_name}")
        print_metric_block(metric)


def print_entity_metrics(entity_metrics: dict[str, dict]) -> None:
    """
    Print entity-level metrics.
    """

    if not entity_metrics:
        print("\n  No entity-level metrics found.")
        return

    print("\n▶ Entity-level metrics")

    header = (
        f"  {'Entity Type':<24} "
        f"{'Acc':>6} "
        f"{'Prec':>6} "
        f"{'Rec':>6} "
        f"{'F1':>6} "
        f"{'TP':>5} "
        f"{'TN':>5} "
        f"{'FP':>5} "
        f"{'FN':>5} "
        f"{'n':>5}"
    )

    print(header)
    print("  " + "-" * (len(header) - 2))

    for entity_type, metric in entity_metrics.items():
        print(
            f"  {entity_type:<24} "
            f"{metric['accuracy']:>6.4f} "
            f"{metric['precision']:>6.4f} "
            f"{metric['recall']:>6.4f} "
            f"{metric['f1']:>6.4f} "
            f"{metric['TP']:>5} "
            f"{metric['TN']:>5} "
            f"{metric['FP']:>5} "
            f"{metric['FN']:>5} "
            f"{metric['n']:>5}"
        )


def print_metric_report(
    metrics: dict,
    title: str | None = None,
) -> None:
    """
    Print a full metric report.
    """

    sep = "=" * 60

    print(f"\n{sep}")

    if title:
        print(title)
        print(sep)

    document_metrics = metrics.get("document", {})
    entity_metrics = metrics.get("entity", {})

    print_document_metrics(document_metrics)
    print_entity_metrics(entity_metrics)

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────

def save_metrics_dataframe(
    metrics_df: pd.DataFrame,
    output_file: Path,
) -> Path:
    """
    Save a metrics dataframe to CSV.
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(
        output_file,
        index=False,
    )

    return output_file


def save_metrics(
    metrics: dict,
    output_dir: Path,
    file_name: str = "metrics.csv",
) -> Path:
    """
    Save nested metrics as a tidy CSV file.
    """

    metrics_df = metrics_to_dataframe(metrics)

    output_file = output_dir / file_name

    return save_metrics_dataframe(
        metrics_df=metrics_df,
        output_file=output_file,
    )