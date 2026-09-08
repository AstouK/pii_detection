"""
Train / validation / test generalisation check for a fitted pre-filter.

    python -m classification.prefilter.overfit_check
    python -m classification.prefilter.overfit_check --run-name distilbert_prefilter

The check scores the **finished checkpoint** on all three splits with the
routing thresholds frozen at the validation operating point. It does not
re-fit ``t_low`` / ``t_high`` on train or test — that would hide the gap
the test is meant to measure.

How to read the gaps
--------------------
``train − val`` large and positive
    the model memorised the training set.

``val − test`` large and positive
    the router calibration did not generalise; thresholds overfit validation.

all three splits near 1.0 with gaps near 0
    no measurable overfitting. On the 500-row DistilBERT pilot this is the
    expected finding: the data is linearly separable, not a proof that the
    model cannot overfit.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.logging_config import setup_logging

from classification.prefilter.config import (
    REPORTS_DIR,
    SPLIT_NAMES,
    run_artifacts_dir,
)
from classification.prefilter.thresholds import (
    average_precision,
    binary_metrics,
    evaluate_routing,
)

setup_logging()
logger = logging.getLogger(__name__)

#: Absolute gap above which a split difference is flagged rather than noise.
GAP_FLAG_THRESHOLD = 0.05

#: F1 at or above this on every split, with gaps below the flag threshold,
#: is read as "the dataset is too easy to measure overfitting".
EASY_DATASET_F1 = 0.99

SPLIT_GAP_COLUMNS = [
    "split",
    "n",
    "positives",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "routed_fraction",
    "prefilter_recall",
    "missed_positives",
    "auto_yes_precision",
    "t_low",
    "t_high",
]

# Palette matches compare_runs.py (categorical slots 1–3).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#c9c8c3"
SURFACE = "#fcfcfb"

PLOT_METRICS = (
    ("f1", "F1 @ 0.5"),
    ("pr_auc", "PR-AUC"),
    ("prefilter_recall", "Pre-filter recall"),
    ("routed_fraction", "Routed fraction"),
)


# ─────────────────────────────────────────────────────────────
# Scoring (no model required)
# ─────────────────────────────────────────────────────────────

def score_split_probs(
    split: str,
    probs: np.ndarray,
    y_true: np.ndarray,
    t_low: float,
    t_high: float,
    standalone_threshold: float = 0.5,
) -> dict:
    """
    Score one split at a **frozen** ``(t_low, t_high)``.

    The thresholds are used as given. This function never calls
    ``calibrate_thresholds``.
    """

    y_bool = np.asarray(y_true).astype(bool)
    probs = np.asarray(probs, dtype=float)

    plain = binary_metrics(y_bool, probs >= standalone_threshold)
    routing = evaluate_routing(probs, y_bool, t_low=t_low, t_high=t_high)

    return {
        "split": split,
        "n": int(plain["n"]),
        "positives": int(y_bool.sum()),
        "accuracy": plain["accuracy"],
        "precision": plain["precision"],
        "recall": plain["recall"],
        "f1": plain["f1"],
        "pr_auc": round(average_precision(y_bool, probs), 6),
        "routed_fraction": routing["routed_fraction"],
        "prefilter_recall": routing["prefilter_recall"],
        "missed_positives": routing["missed_positives"],
        "auto_yes_precision": routing["auto_yes_precision"],
        "t_low": routing["t_low"],
        "t_high": routing["t_high"],
    }


def compute_gaps(rows: list[dict]) -> dict:
    """
    ``train − val`` and ``val − test`` for F1, PR-AUC and pre-filter recall.
    """

    by_split = {row["split"]: row for row in rows}

    def _gap(left: str, right: str, key: str) -> float | None:
        if left not in by_split or right not in by_split:
            return None

        left_value = by_split[left].get(key)
        right_value = by_split[right].get(key)

        if left_value is None or right_value is None:
            return None

        return round(float(left_value) - float(right_value), 6)

    return {
        "train_minus_val_f1": _gap("train", "validation", "f1"),
        "val_minus_test_f1": _gap("validation", "test", "f1"),
        "train_minus_val_pr_auc": _gap("train", "validation", "pr_auc"),
        "val_minus_test_pr_auc": _gap("validation", "test", "pr_auc"),
        "train_minus_val_prefilter_recall": _gap(
            "train", "validation", "prefilter_recall"
        ),
        "val_minus_test_prefilter_recall": _gap(
            "validation", "test", "prefilter_recall"
        ),
    }


def interpret_gaps(
    rows: list[dict],
    gaps: dict,
    gap_threshold: float = GAP_FLAG_THRESHOLD,
    easy_f1: float = EASY_DATASET_F1,
) -> dict:
    """
    Turn numeric gaps into the three readings the report needs.
    """

    flags: list[str] = []

    train_val_f1 = gaps.get("train_minus_val_f1")
    val_test_f1 = gaps.get("val_minus_test_f1")
    val_test_recall = gaps.get("val_minus_test_prefilter_recall")

    if train_val_f1 is not None and train_val_f1 >= gap_threshold:
        flags.append("train_val_overfit")

    if (val_test_f1 is not None and val_test_f1 >= gap_threshold) or (
        val_test_recall is not None and val_test_recall >= gap_threshold
    ):
        flags.append("val_test_calibration_gap")

    f1_values = [row.get("f1") for row in rows if row.get("f1") is not None]
    gap_magnitudes = [
        abs(value)
        for value in (train_val_f1, val_test_f1)
        if value is not None
    ]

    if (
        f1_values
        and all(value >= easy_f1 for value in f1_values)
        and gap_magnitudes
        and all(value < gap_threshold for value in gap_magnitudes)
    ):
        flags.append("dataset_too_easy")

    return {
        "flags": flags,
        "gap_threshold": gap_threshold,
        "reading": _reading(flags),
    }


def _reading(flags: list[str]) -> str:
    if "train_val_overfit" in flags:
        return (
            "Train is substantially above validation: the model memorised the "
            "training set."
        )

    if "val_test_calibration_gap" in flags:
        return (
            "Validation is substantially above test: the frozen routing "
            "thresholds did not generalise."
        )

    if "dataset_too_easy" in flags:
        return (
            "Train, validation and test are all near-perfect with no gap. "
            "Overfitting is unmeasurable on this dataset, not disproven."
        )

    return "No material generalisation gap on the scored splits."


def split_gap_payload(
    rows: list[dict],
    t_low: float,
    t_high: float,
    calibrated_on: str = "validation",
) -> dict:
    """
    Bundle rows, gaps and the interpretation into the JSON artifact shape.
    """

    gaps = compute_gaps(rows)

    return {
        "t_low": round(float(t_low), 6),
        "t_high": round(float(t_high), 6),
        "calibrated_on": calibrated_on,
        "splits": rows,
        "gaps": gaps,
        "interpretation": interpret_gaps(rows, gaps),
        "table": pd.DataFrame(rows, columns=SPLIT_GAP_COLUMNS),
    }


# ─────────────────────────────────────────────────────────────
# Scoring with a live model
# ─────────────────────────────────────────────────────────────

def score_splits(
    model,
    frames: dict[str, pd.DataFrame],
    loaders: dict,
    calibration: dict,
    config,
    device,
) -> dict:
    """
    Run the frozen-threshold check on every split present in ``frames``.
    """

    from classification.prefilter.data import extract_labels
    from classification.prefilter.train import predict_probabilities

    t_low = float(calibration["t_low"])
    t_high = float(calibration["t_high"])
    standalone_threshold = float(config.standalone_threshold)

    rows: list[dict] = []

    for split in SPLIT_NAMES:
        frame = frames.get(split)

        if frame is None or frame.empty:
            raise ValueError(
                f"Split '{split}' is empty; the overfit check needs train, "
                "validation and test."
            )

        if split not in loaders:
            raise ValueError(f"No dataloader for split '{split}'.")

        probs, _entity_probs = predict_probabilities(model, loaders[split], device)
        y_true, _entity_true = extract_labels(frame)

        rows.append(
            score_split_probs(
                split=split,
                probs=probs,
                y_true=y_true,
                t_low=t_low,
                t_high=t_high,
                standalone_threshold=standalone_threshold,
            )
        )

        logger.info(
            "split=%s n=%s f1=%.4f pr_auc=%.4f prefilter_recall=%s routed=%.1f%%",
            split,
            rows[-1]["n"],
            rows[-1]["f1"],
            rows[-1]["pr_auc"],
            rows[-1]["prefilter_recall"],
            100 * rows[-1]["routed_fraction"],
        )

    return split_gap_payload(
        rows,
        t_low=t_low,
        t_high=t_high,
        calibrated_on=str(calibration.get("calibrated_on", "validation")),
    )


# ─────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────

def persist_split_gap(result: dict, output_dir: Path) -> dict[str, Path]:
    """
    Write ``split_gap.csv``, ``split_gap.json`` and ``split_gap.png``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "split_gap.csv"
    json_path = output_dir / "split_gap.json"
    png_path = output_dir / "split_gap.png"

    table = result["table"]
    table.to_csv(csv_path, index=False)

    payload = {
        key: value
        for key, value in result.items()
        if key != "table"
    }
    json_path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    plot_split_gap(table, png_path)

    logger.info("Wrote split-gap artifacts to %s", output_dir)

    return {"csv": csv_path, "json": json_path, "png": png_path}


def plot_split_gap(table: pd.DataFrame, output_file: Path) -> Path:
    """
    Grouped bars: F1, PR-AUC, pre-filter recall and routed fraction by split.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(10.2, 4.6), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, alpha=0.45, linewidth=0.7)
    axes.set_axisbelow(True)

    for side in ("top", "right"):
        axes.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)

    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    metric_keys = [key for key, _label in PLOT_METRICS]
    metric_labels = [label for _key, label in PLOT_METRICS]
    splits = [name for name in SPLIT_NAMES if name in set(table["split"])]

    positions = np.arange(len(metric_keys))
    width = 0.8 / max(len(splits), 1)

    indexed = table.set_index("split")

    for index, split in enumerate(splits):
        offset = (index - (len(splits) - 1) / 2) * width
        values = [
            float(indexed.loc[split, key]) if pd.notna(indexed.loc[split, key]) else 0.0
            for key in metric_keys
        ]

        bars = axes.bar(
            positions + offset,
            values,
            width=width * 0.92,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            label=split,
            edgecolor=SURFACE,
            linewidth=1.2,
        )

        for bar, value in zip(bars, values):
            axes.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=TEXT_SECONDARY,
            )

    axes.set_xticks(positions)
    axes.set_xticklabels(metric_labels)
    axes.set_ylim(0, 1.12)
    axes.set_title(
        "Generalisation gap (frozen validation thresholds)",
        color=TEXT_PRIMARY,
        fontsize=12,
        pad=14,
        loc="left",
    )
    axes.set_ylabel("Score", color=TEXT_SECONDARY, fontsize=10)
    axes.legend(
        fontsize=9,
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
        labelcolor=TEXT_PRIMARY,
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_file, dpi=170, facecolor=SURFACE)
    plt.close(figure)

    return output_file


def print_split_gap_summary(run_name: str, result: dict) -> None:
    gaps = result["gaps"]
    interpretation = result["interpretation"]

    print("\n" + "=" * 64)
    print(f"OVERFIT CHECK — {run_name}")
    print("=" * 64)
    print(
        f"  frozen thresholds : t_low={result['t_low']:.4f}  "
        f"t_high={result['t_high']:.4f}  "
        f"(calibrated on {result['calibrated_on']})"
    )
    print()
    print(
        f"  {'split':<12} {'n':>5} {'pos':>4} {'f1':>7} {'pr_auc':>8} "
        f"{'prefilter_recall':>16} {'routed':>8}"
    )

    for row in result["splits"]:
        recall = row["prefilter_recall"]
        recall_text = f"{recall:.4f}" if recall is not None else "   n/a"
        print(
            f"  {row['split']:<12} {row['n']:>5} {row['positives']:>4} "
            f"{row['f1']:>7.4f} {row['pr_auc']:>8.4f} "
            f"{recall_text:>16} {100 * row['routed_fraction']:>7.1f}%"
        )

    def _fmt(value: float | None) -> str:
        return f"{value:+.4f}" if value is not None else "  n/a"

    print()
    print("  gaps (train − val / val − test)")
    print(
        f"    f1               : {_fmt(gaps['train_minus_val_f1'])} / "
        f"{_fmt(gaps['val_minus_test_f1'])}"
    )
    print(
        f"    pr_auc           : {_fmt(gaps['train_minus_val_pr_auc'])} / "
        f"{_fmt(gaps['val_minus_test_pr_auc'])}"
    )
    print(
        f"    prefilter_recall : {_fmt(gaps['train_minus_val_prefilter_recall'])} / "
        f"{_fmt(gaps['val_minus_test_prefilter_recall'])}"
    )
    print()
    flags = interpretation["flags"] or ["none"]
    print(f"  flags   : {', '.join(flags)}")
    print(f"  reading : {interpretation['reading']}")
    print("=" * 64 + "\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def run_overfit_check(
    run_name: str,
    data_file: str | None = None,
    device: str = "",
) -> dict:
    """
    Load a checkpoint, score all three splits, persist the gap table.
    """

    from classification.prefilter.data import (
        build_dataloader,
        build_dataset,
        load_dataset,
        resolve_splits,
        split_frames,
    )
    from classification.prefilter.model import load_checkpoint, load_tokenizer
    from classification.prefilter.thresholds import load_calibration
    from classification.prefilter.train import resolve_device

    artifacts_dir = run_artifacts_dir(run_name)
    torch_device = resolve_device(device)

    model, config = load_checkpoint(artifacts_dir, device=torch_device)
    model.to(torch_device)

    calibration_file = artifacts_dir / "calibration.json"

    if not calibration_file.exists():
        raise FileNotFoundError(
            f"Calibration not found: {calibration_file}. Run training first."
        )

    calibration = load_calibration(calibration_file)

    logger.info(
        "Loaded %s (t_low=%.4f, t_high=%.4f, calibrated on %s)",
        run_name,
        float(calibration["t_low"]),
        float(calibration["t_high"]),
        calibration.get("calibrated_on", "validation"),
    )

    df = load_dataset(data_file or config.data_file)
    df, resolved_mode = resolve_splits(df, config)

    if resolved_mode != config.resolved_split_mode:
        logger.warning(
            "Split mode resolved to '%s' but the checkpoint was trained with "
            "'%s'. Splits may not match the ones used during training.",
            resolved_mode,
            config.resolved_split_mode,
        )

    frames = split_frames(df)
    tokenizer = load_tokenizer(config.pretrained_dir)

    loaders = {
        name: build_dataloader(
            build_dataset(frame, tokenizer, config.max_length),
            batch_size=config.eval_batch_size,
            shuffle=False,
        )
        for name, frame in frames.items()
    }

    result = score_splits(
        model=model,
        frames=frames,
        loaders=loaders,
        calibration=calibration,
        config=config,
        device=torch_device,
    )

    persist_split_gap(result, artifacts_dir)
    persist_split_gap(result, REPORTS_DIR / run_name)
    print_split_gap_summary(run_name, result)

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score train, validation and test with frozen validation "
            "thresholds and report the generalisation gap."
        )
    )
    parser.add_argument("--run-name", default="distilbert_prefilter")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--device", default="")

    args = parser.parse_args(argv)

    run_overfit_check(
        run_name=args.run_name,
        data_file=args.data_file,
        device=args.device,
    )


if __name__ == "__main__":
    main()
