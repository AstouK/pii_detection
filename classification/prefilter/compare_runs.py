"""
Compare two or more pre-filter runs side by side.

    python -m classification.prefilter.compare_runs distilbert_prefilter mbert_prefilter_1400

Writes PNGs to ``classification/prefilter/reports/comparison/``. Reads each run
from ``artifacts/<run_name>/``, which holds the complete record; the copies
under ``reports/<run_name>/`` are a subset published for the repository.

Why a module and not a notebook: the comparison is the deliverable. "How far
does the LLM call volume drop, and does recall hold" is only answerable against
a baseline, so producing that comparison has to be repeatable and versioned
rather than reconstructed by hand each time the dataset changes.

The five figures, in the order they answer questions:

    routing_frontier   the headline — LLM calls vs. guaranteed recall
    score_distribution why the frontier looks the way it does
    training_curves    did it converge, and did routing cost settle
    entity_f1          did more data help the 12-label head
    headline_metrics   binary accuracy/precision/recall/F1 side by side
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
    ARTIFACTS_DIR,
    BINARY_LABEL_COL,
    ENTITY_LABELS,
    REPORTS_DIR,
)
from classification.prefilter.data import to_bool_series

setup_logging()
logger = logging.getLogger(__name__)

COMPARISON_DIR = REPORTS_DIR / "comparison"

# ── Palette ─────────────────────────────────────────────────
# Categorical slots 1 and 2 of the project's validated palette. Verified with
# the palette validator for this two-series case: worst adjacent CVD ΔE 24.7
# (protan), normal-vision ΔE 33.6, both well clear of the 8 / 15 floors, and
# both clear 3:1 against the light surface.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#c9c8c3"
SURFACE = "#fcfcfb"

#: The rule-based Sweep 1 baseline the router must not fall below.
BASELINE_RECALL = 0.9833


# ─────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_run(run_name: str) -> dict:
    """
    Load everything the figures need for one run.

    Missing pieces come back as ``None`` rather than raising: a run interrupted
    before calibration should still contribute its training curve.
    """

    run_dir = ARTIFACTS_DIR / run_name
    published_dir = REPORTS_DIR / run_name

    def _pick(file_name: str):
        """Prefer the authoritative artifact, fall back to the published copy."""
        primary = run_dir / file_name
        return primary if primary.exists() else published_dir / file_name

    if not run_dir.exists() and not published_dir.exists():
        raise FileNotFoundError(
            f"No artifacts for run '{run_name}'. Looked in {run_dir} and "
            f"{published_dir}."
        )

    summary = _read_json(_pick("metrics_summary.json")) or {}
    config = summary.get("config", {})

    return {
        "name": run_name,
        "label": _run_label(run_name, summary, config),
        "summary": summary,
        "config": config,
        "calibration": _read_json(_pick("calibration.json")) or {},
        "frontier": _read_csv(_pick("routing_frontier.csv")),
        "history": _read_csv(_pick("training_history.csv")),
        "entity": _read_csv(_pick("validation_entity_metrics.csv")),
        "scores": _read_csv(_pick("validation_scores.csv")),
    }


def _run_label(run_name: str, summary: dict, config: dict) -> str:
    """
    Legend label: what actually distinguishes the runs, not the directory name.
    """

    n_documents = sum(
        row.get("n", 0) for row in summary.get("split_summary", [])
    )

    # Keep the model family in the label. Stripping it down to "uncased" vs
    # "multilingual-cased" saves space at the cost of a legend nobody can read
    # without the surrounding text.
    model = str(config.get("model_name", run_name))
    model = model.replace("-base-", "-").replace("-v1", "")

    if n_documents:
        return f"{model} · {n_documents} docs"

    return run_name


# ─────────────────────────────────────────────────────────────
# Figure scaffolding
# ─────────────────────────────────────────────────────────────

def _new_figure(width: float = 8.0, height: float = 5.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    _style(axes)

    return plt, figure, axes


def _style(axes) -> None:
    """
    Recessive grid and axes; text in ink colours, never a series colour.
    """

    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, alpha=0.45, linewidth=0.7)
    axes.set_axisbelow(True)

    for side in ("top", "right"):
        axes.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)

    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    for label in (*axes.get_xticklabels(), *axes.get_yticklabels()):
        label.set_color(TEXT_SECONDARY)


def _titles(axes, title: str, xlabel: str, ylabel: str, subtitle: str = "") -> None:
    axes.set_title(title, color=TEXT_PRIMARY, fontsize=12, pad=14, loc="left")
    axes.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    axes.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)

    if subtitle:
        axes.annotate(
            subtitle,
            xy=(0, 1.02),
            xycoords="axes fraction",
            color=TEXT_SECONDARY,
            fontsize=9,
        )


def _save(plt, figure, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_file, dpi=170, facecolor=SURFACE)
    plt.close(figure)

    logger.info("Wrote %s", output_file)

    return output_file


# ─────────────────────────────────────────────────────────────
# 1. Routing frontier — the headline
# ─────────────────────────────────────────────────────────────

def plot_frontier_comparison(runs: list[dict], output_file: Path) -> Path | None:
    """
    LLM-call share against the recall the router still guarantees.

    A vertical line at 0% means the classes separate so cleanly that no recall
    target costs a call — true of the pilot, and a sign the dataset is too easy
    rather than that the router is good.
    """

    usable = [run for run in runs if run["frontier"] is not None]

    if not usable:
        return None

    plt, figure, axes = _new_figure(8.2, 5.2)

    for index, run in enumerate(usable):
        frontier = run["frontier"]
        frontier = frontier[frontier["feasible"]]

        if frontier.empty:
            continue

        axes.plot(
            frontier["routed_fraction"] * 100,
            frontier["recall_target"],
            marker="o",
            markersize=4,
            linewidth=2,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            label=run["label"],
        )

        operating = run["calibration"]

        if operating.get("feasible") and operating.get("prefilter_recall") is not None:
            axes.scatter(
                [operating["routed_fraction"] * 100],
                [operating["prefilter_recall"]],
                s=160,
                marker="*",
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=5,
            )

    axes.axhline(
        BASELINE_RECALL,
        color=TEXT_SECONDARY,
        linestyle="--",
        linewidth=1.2,
        label=f"rule-based baseline ({BASELINE_RECALL:.4f})",
    )

    # Same reason as the training panels: a frontier pinned at 0% must not get
    # a hairline axis that hides what it is saying.
    axes.set_xlim(left=-1.0)

    _titles(
        axes,
        "LLM-call volume vs. guaranteed recall",
        "Documents routed to the LLM (%)",
        "Pre-filter recall (positives not silently dropped)",
        "★ = calibrated operating point · lower-right is better",
    )

    axes.legend(
        loc="lower right",
        fontsize=9,
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
        labelcolor=TEXT_PRIMARY,
    )

    return _save(plt, figure, output_file)


# ─────────────────────────────────────────────────────────────
# 2. Score distribution — why the frontier looks like that
# ─────────────────────────────────────────────────────────────

def plot_score_distributions(runs: list[dict], output_file: Path) -> Path | None:
    """
    One panel per run: predicted probability by true class, with the routing
    band shaded.

    This is the diagnostic. A wide empty gap between the classes means there is
    nothing to route; an overlap means the router has real work to do.
    """

    usable = [run for run in runs if run["scores"] is not None]

    if not usable:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes_row = plt.subplots(
        1,
        len(usable),
        figsize=(6.2 * len(usable), 4.6),
        facecolor=SURFACE,
        squeeze=False,
    )

    bins = np.linspace(0, 1, 41)

    for index, (run, axes) in enumerate(zip(usable, axes_row[0])):
        _style(axes)

        scores = run["scores"]
        positive = to_bool_series(scores[BINARY_LABEL_COL])
        probs = scores["pii_probability"].to_numpy()

        axes.hist(
            probs[~positive],
            bins=bins,
            color=SERIES_COLORS[0],
            alpha=0.75,
            label=f"no personal data (n={int((~positive).sum())})",
        )
        axes.hist(
            probs[positive],
            bins=bins,
            color=SERIES_COLORS[1],
            alpha=0.75,
            label=f"contains personal data (n={int(positive.sum())})",
        )

        operating = run["calibration"]
        t_low = operating.get("t_low")
        t_high = operating.get("t_high")

        if t_low is not None and t_high is not None:
            routed = operating.get("routed_fraction", 0.0)
            if t_high > t_low:
                axes.axvspan(
                    t_low,
                    t_high,
                    color="#eda100",
                    alpha=0.22,
                    label=f"routed to LLM ({100 * routed:.1f}%)",
                )
            for cut in (t_low, t_high):
                axes.axvline(cut, color=TEXT_PRIMARY, linestyle="--", linewidth=1.1)

        axes.set_yscale("symlog")
        _titles(
            axes,
            run["label"],
            "Predicted probability of personal data",
            "Documents" if index == 0 else "",
        )
        axes.legend(
            fontsize=8,
            frameon=True,
            facecolor=SURFACE,
            edgecolor=GRID,
            labelcolor=TEXT_PRIMARY,
        )

    figure.suptitle(
        "Score distribution and routing zone",
        color=TEXT_PRIMARY,
        fontsize=13,
        x=0.01,
        ha="left",
    )

    return _save(plt, figure, output_file)


# ─────────────────────────────────────────────────────────────
# 3. Training curves
# ─────────────────────────────────────────────────────────────

def plot_training_curves(runs: list[dict], output_file: Path) -> Path | None:
    """
    Validation F1 and routing cost per epoch.

    Two panels rather than two y-axes on one: a dual-axis chart invites the
    reader to compare two scales that have no common meaning.
    """

    usable = [run for run in runs if run["history"] is not None]

    if not usable:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 4.4), facecolor=SURFACE
    )

    for axes in (left, right):
        _style(axes)

    for index, run in enumerate(usable):
        history = run["history"]
        color = SERIES_COLORS[index % len(SERIES_COLORS)]

        left.plot(
            history["epoch"],
            history["val_f1"],
            marker="o",
            markersize=6,
            linewidth=2,
            color=color,
            label=run["label"],
        )

        right.plot(
            history["epoch"],
            history["val_routed_fraction"] * 100,
            marker="o",
            markersize=6,
            linewidth=2,
            color=color,
            label=run["label"],
        )

        selected = run["summary"].get("selected_epoch")

        if selected:
            for axes, column, scale in (
                (left, "val_f1", 1),
                (right, "val_routed_fraction", 100),
            ):
                row = history[history["epoch"] == selected]
                if not row.empty:
                    axes.scatter(
                        [selected],
                        [row[column].iloc[0] * scale],
                        s=170,
                        marker="*",
                        color=color,
                        edgecolor=SURFACE,
                        linewidth=1.5,
                        zorder=5,
                    )

    # Anchor both axes rather than letting them autoscale. A run whose routing
    # cost is 0 in every epoch otherwise gets an axis spanning ±0.04%, which
    # reads as a broken plot instead of as the finding it is.
    left.set_ylim(top=1.03)
    right.set_ylim(bottom=0)

    _titles(left, "Validation F1 per epoch", "Epoch", "F1", "★ = selected checkpoint")
    _titles(
        right,
        "Routing cost per epoch",
        "Epoch",
        "Documents routed to the LLM (%)",
        "lower is cheaper",
    )

    for axes in (left, right):
        axes.legend(
            fontsize=9,
            frameon=True,
            facecolor=SURFACE,
            edgecolor=GRID,
            labelcolor=TEXT_PRIMARY,
        )

    return _save(plt, figure, output_file)


# ─────────────────────────────────────────────────────────────
# 4. Entity head
# ─────────────────────────────────────────────────────────────

def plot_entity_comparison(runs: list[dict], output_file: Path) -> Path | None:
    """
    Per-label F1 for the 12-label head, with validation support annotated.

    Support is the point of the figure: an F1 fitted on one validation example
    is noise, and the label needs to say so next to the bar.
    """

    usable = [run for run in runs if run["entity"] is not None]

    if not usable:
        return None

    plt, figure, axes = _new_figure(10.5, 5.6)

    positions = np.arange(len(ENTITY_LABELS))
    width = 0.8 / len(usable)

    for index, run in enumerate(usable):
        entity = run["entity"].set_index("entity").reindex(ENTITY_LABELS)
        offset = (index - (len(usable) - 1) / 2) * width

        bars = axes.bar(
            positions + offset,
            entity["f1"].fillna(0),
            width=width * 0.92,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            label=run["label"],
            edgecolor=SURFACE,
            linewidth=1.2,
        )

        for bar, support in zip(bars, entity["support"].fillna(0)):
            axes.annotate(
                f"n={int(support)}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color=TEXT_SECONDARY,
                rotation=90,
            )

    axes.set_xticks(positions)
    axes.set_xticklabels(ENTITY_LABELS, rotation=35, ha="right", fontsize=8)
    axes.set_ylim(0, 1.15)

    _titles(
        axes,
        "Entity head: per-label F1 on validation",
        "",
        "F1",
        "n = positive validation documents for that label; a label with n≈1 is noise",
    )

    axes.legend(
        fontsize=9,
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
        labelcolor=TEXT_PRIMARY,
    )

    return _save(plt, figure, output_file)


# ─────────────────────────────────────────────────────────────
# 5. Headline metrics
# ─────────────────────────────────────────────────────────────

HEADLINE_METRICS = ["accuracy", "precision", "recall", "f1"]


def plot_headline_metrics(runs: list[dict], output_file: Path) -> Path | None:
    """
    Binary metrics at the 0.5 cut, on validation.

    Deliberately the unrouted numbers: they say how good the classifier is,
    where the frontier says what the router costs.
    """

    usable = [
        run for run in runs if run["summary"].get("validation_binary_at_0_5")
    ]

    if not usable:
        return None

    plt, figure, axes = _new_figure(8.0, 4.6)

    positions = np.arange(len(HEADLINE_METRICS))
    width = 0.8 / len(usable)

    for index, run in enumerate(usable):
        metrics = run["summary"]["validation_binary_at_0_5"]
        offset = (index - (len(usable) - 1) / 2) * width

        values = [metrics.get(name, 0.0) for name in HEADLINE_METRICS]

        bars = axes.bar(
            positions + offset,
            values,
            width=width * 0.92,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            label=run["label"],
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
    axes.set_xticklabels([name.capitalize() for name in HEADLINE_METRICS])
    axes.set_ylim(0, 1.12)

    _titles(
        axes,
        "Binary classification on validation (threshold 0.5)",
        "",
        "Score",
        "unrouted — how good the classifier is, before any routing",
    )

    axes.legend(
        fontsize=9,
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
        labelcolor=TEXT_PRIMARY,
    )

    return _save(plt, figure, output_file)


# ─────────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────────

def comparison_table(runs: list[dict]) -> pd.DataFrame:
    """
    The figures as numbers — every chart needs a table view.
    """

    rows = []

    for run in runs:
        summary = run["summary"]
        binary = summary.get("validation_binary_at_0_5", {})
        operating = run["calibration"]

        rows.append(
            {
                "run": run["name"],
                "model": run["config"].get("model_name", ""),
                "documents": sum(
                    row.get("n", 0) for row in summary.get("split_summary", [])
                ),
                "split_mode": summary.get("split_mode", ""),
                "max_length": run["config"].get("max_length", ""),
                "selected_epoch": summary.get("selected_epoch", ""),
                "val_accuracy": binary.get("accuracy"),
                "val_precision": binary.get("precision"),
                "val_recall": binary.get("recall"),
                "val_f1": binary.get("f1"),
                "val_pr_auc": summary.get("validation_pr_auc"),
                "routed_fraction": operating.get("routed_fraction"),
                "prefilter_recall": operating.get("prefilter_recall"),
                "missed_positives": operating.get("missed_positives"),
                "t_low": operating.get("t_low"),
                "t_high": operating.get("t_high"),
                "training_seconds": summary.get("training_seconds"),
                "model_size_mb": summary.get("model_size_mb"),
            }
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def compare(run_names: list[str], output_dir: Path) -> list[Path]:
    """
    Build every figure plus the table for the given runs.
    """

    runs = [load_run(name) for name in run_names]

    output_dir.mkdir(parents=True, exist_ok=True)

    written = [
        plot_frontier_comparison(runs, output_dir / "routing_frontier.png"),
        plot_score_distributions(runs, output_dir / "score_distribution.png"),
        plot_training_curves(runs, output_dir / "training_curves.png"),
        plot_entity_comparison(runs, output_dir / "entity_f1.png"),
        plot_headline_metrics(runs, output_dir / "headline_metrics.png"),
    ]

    table = comparison_table(runs)
    table_file = output_dir / "comparison.csv"
    table.to_csv(table_file, index=False)
    written.append(table_file)

    print("\n" + "=" * 78)
    print("RUN COMPARISON")
    print("=" * 78)
    print(table.to_string(index=False))
    print("=" * 78 + "\n")

    return [path for path in written if path is not None]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare pre-filter runs and write comparison figures."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run names, i.e. the directory names under artifacts/.",
    )
    parser.add_argument("--output-dir", default=str(COMPARISON_DIR))

    args = parser.parse_args(argv)

    compare(args.runs, Path(args.output_dir))


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────
# 6. Pre-filter vs. the rule-based baseline
# ─────────────────────────────────────────────────────────────

def plot_baseline_comparison(
    predictions_dir: Path,
    output_file: Path,
    strategies: tuple[str, ...] = ("rule_based", "bert_prefilter"),
    labels: tuple[str, ...] = ("Sweep 1 (Presidio + spaCy + regex)", "Pre-filter (DistilBERT)"),
) -> Path | None:
    """
    The two numbers that decide whether the pre-filter is worth having.

    Recall is measured the recoverable way for both: the share of positive
    documents a stage does not *irrecoverably* drop. A document Sweep 1 escalates
    to the LLM is not lost, so scoring it by strict `detected_pii` understates it
    badly (0.32 rather than 0.93) and would make the comparison dishonest in the
    pre-filter's favour.

    Two panels rather than two axes on one chart: recall is a rate on positives,
    LLM share is a rate on all documents, and they have no common scale.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []

    for strategy, label in zip(strategies, labels):
        path = Path(predictions_dir) / f"{strategy}.csv"

        if not path.exists():
            continue

        frame = pd.read_csv(path)

        truth = to_bool_series(frame[BINARY_LABEL_COL])

        routed = (
            to_bool_series(frame["needs_llm_review"])
            if "needs_llm_review" in frame.columns
            else pd.Series([False] * len(frame))
        )

        decided = (
            to_bool_series(frame["detected_pii"])
            if "detected_pii" in frame.columns
            else to_bool_series(frame["predicted_pii"])
        )

        lost = truth & ~decided & ~routed

        rows.append(
            {
                "label": label,
                "recall": 1.0 - lost.sum() / max(truth.sum(), 1),
                "routed": float(routed.mean()),
                "lost": int(lost.sum()),
                "n_positive": int(truth.sum()),
            }
        )

    if len(rows) < 2:
        return None

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), facecolor=SURFACE
    )

    for axes in (left, right):
        _style(axes)

    positions = np.arange(len(rows))
    colors = [SERIES_COLORS[index % len(SERIES_COLORS)] for index in range(len(rows))]

    recall_bars = left.bar(
        positions,
        [row["recall"] for row in rows],
        width=0.55,
        color=colors,
        edgecolor=SURFACE,
        linewidth=1.2,
    )

    for bar, row in zip(recall_bars, rows):
        left.annotate(
            f"{row['recall']:.4f}\n({row['lost']} of {row['n_positive']} lost)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )

    left.set_ylim(0, 1.15)
    left.set_xticks(positions)
    left.set_xticklabels([row["label"] for row in rows], fontsize=8.5)

    _titles(
        left,
        "Recall — positives not irrecoverably dropped",
        "",
        "Recall",
        "higher is safer",
    )

    routed_bars = right.bar(
        positions,
        [100 * row["routed"] for row in rows],
        width=0.55,
        color=colors,
        edgecolor=SURFACE,
        linewidth=1.2,
    )

    for bar, row in zip(routed_bars, rows):
        right.annotate(
            f"{100 * row['routed']:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=TEXT_SECONDARY,
        )

    right.set_ylim(0, 100)
    right.set_xticks(positions)
    right.set_xticklabels([row["label"] for row in rows], fontsize=8.5)

    _titles(
        right,
        "Documents routed to the LLM",
        "",
        "Share of all documents (%)",
        "lower is cheaper",
    )

    return _save(plt, figure, output_file)
