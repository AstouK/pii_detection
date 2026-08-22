"""
Three-zone routing and threshold calibration.

Instead of a single 0.5 cut, the pre-filter splits the probability axis into
three zones:

    p <  t_low    confident non-PII   -> decided locally, no LLM call
    p >  t_high   confident PII       -> decided locally, no LLM call
    otherwise     uncertain           -> routed to the LLM

Why recall is measured the way it is
------------------------------------
A routed document is not a mistake — it goes on to a stronger model. The only
irrecoverable errors this stage can make are:

    * a positive document dropped into the ``p < t_low`` zone (a false negative
      that nothing downstream will ever revisit), and
    * a negative document auto-approved as PII in the ``p > t_high`` zone (a
      false positive that is never reviewed).

So the recall that must stay at or above the baseline is
:func:`prefilter_recall`: the share of positive documents that are *not*
silently dropped. It counts routed positives as saved, because they are.

Two constraints, not one
------------------------
Minimising LLM calls under the recall constraint alone is degenerate: the
optimiser would set ``t_high = t_low``, route nothing, and let every
false positive through unreviewed. The upper zone therefore carries its own
constraint — ``precision`` of the auto-approved-PII zone must clear
``precision_target``.

End-to-end numbers are reported under two explicit assumptions, because the LLM
is not run here:

    ``oracle``        the LLM answers routed documents correctly. Upper bound.
    ``conservative``  routed documents count as flagged-PII. This is what the
                      prediction CSV writes, and it is the GDPR-safe reading:
                      escalated means "treated as potential personal data until
                      a reviewer says otherwise".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Basic metrics
# ─────────────────────────────────────────────────────────────

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Accuracy / precision / recall / F1 plus the confusion counts.
    """

    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())

    n = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n": n,
        "accuracy": round((tp + tn) / n, 6) if n else 0.0,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def average_precision(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Area under the precision-recall curve (PR-AUC), computed as the step-wise
    average precision. Used as the tie-break model-selection metric when no
    threshold pair meets the recall target.
    """

    y_true = np.asarray(y_true).astype(bool)
    probs = np.asarray(probs, dtype=float)

    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return 0.0

    order = np.argsort(-probs, kind="mergesort")
    sorted_true = y_true[order]

    tp = np.cumsum(sorted_true)
    fp = np.cumsum(~sorted_true)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    recall_gain = np.diff(np.concatenate([[0.0], recall]))

    return float(np.sum(precision * recall_gain))


# ─────────────────────────────────────────────────────────────
# Routing evaluation
# ─────────────────────────────────────────────────────────────

def zone_masks(
    probs: np.ndarray,
    t_low: float,
    t_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(auto_no, routed, auto_yes)`` boolean masks.
    """

    probs = np.asarray(probs, dtype=float)

    auto_no = probs < t_low
    auto_yes = probs > t_high
    routed = ~(auto_no | auto_yes)

    return auto_no, routed, auto_yes


def evaluate_routing(
    probs: np.ndarray,
    y_true: np.ndarray,
    t_low: float,
    t_high: float,
) -> dict:
    """
    Full metric set for one ``(t_low, t_high)`` operating point.

    The headline pair is ``routed_fraction`` (the LLM bill) and
    ``prefilter_recall`` (the safety guarantee).
    """

    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true).astype(bool)

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    auto_no, routed, auto_yes = zone_masks(probs, t_low, t_high)

    missed_positives = int((y_true & auto_no).sum())
    prefilter_recall = (
        1.0 - missed_positives / n_pos if n_pos else float("nan")
    )

    auto_yes_tp = int((y_true & auto_yes).sum())
    auto_yes_fp = int((~y_true & auto_yes).sum())
    auto_yes_n = auto_yes_tp + auto_yes_fp

    auto_no_n = int(auto_no.sum())
    auto_no_correct = int((~y_true & auto_no).sum())

    routed_n = int(routed.sum())
    auto_n = n - routed_n

    # Two readings of the end-to-end decision, see the module docstring.
    oracle_pred = auto_yes | (routed & y_true)
    conservative_pred = auto_yes | routed

    return {
        "t_low": round(float(t_low), 6),
        "t_high": round(float(t_high), 6),
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,

        # ── Core routing metrics ────────────────────────────
        "routed_n": routed_n,
        "routed_fraction": round(routed_n / n, 6) if n else 0.0,
        "auto_decided_n": auto_n,
        "llm_calls_avoided": auto_n,
        "llm_call_reduction": round(auto_n / n, 6) if n else 0.0,

        # ── Safety ──────────────────────────────────────────
        "prefilter_recall": round(prefilter_recall, 6) if n_pos else None,
        "missed_positives": missed_positives,

        # ── Zone quality ────────────────────────────────────
        "auto_yes_n": auto_yes_n,
        "auto_yes_precision": (
            round(auto_yes_tp / auto_yes_n, 6) if auto_yes_n else None
        ),
        "auto_yes_fp": auto_yes_fp,
        "auto_no_n": auto_no_n,
        "auto_no_npv": (
            round(auto_no_correct / auto_no_n, 6) if auto_no_n else None
        ),
        "routed_positives": int((y_true & routed).sum()),
        "routed_negatives": int((~y_true & routed).sum()),

        # ── End-to-end under the two assumptions ────────────
        **{
            f"oracle_{key}": value
            for key, value in binary_metrics(y_true, oracle_pred).items()
        },
        **{
            f"conservative_{key}": value
            for key, value in binary_metrics(y_true, conservative_pred).items()
        },
    }


# ─────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────

def _candidate_thresholds(probs: np.ndarray, grid_steps: int) -> np.ndarray:
    """
    Candidate cut points: a uniform grid plus the midpoints between observed
    probabilities.

    The midpoints matter on a small validation split, where the useful cut
    almost always sits in a gap between two adjacent scores and a uniform grid
    can step straight over it.
    """

    probs = np.asarray(probs, dtype=float)

    grid = np.linspace(0.0, 1.0, max(int(grid_steps), 2))

    unique = np.unique(probs)
    midpoints = (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else np.array([])

    candidates = np.concatenate([grid, unique, midpoints, [0.0, 1.0]])

    return np.unique(np.clip(candidates, 0.0, 1.0))


def calibrate_thresholds(
    probs: np.ndarray,
    y_true: np.ndarray,
    recall_target: float,
    precision_target: float,
    grid_steps: int = 201,
) -> dict:
    """
    Search ``(t_low, t_high)`` minimising LLM calls under both constraints.

    Objective:
        minimise ``routed_fraction``
    subject to:
        ``prefilter_recall  >= recall_target``
        ``auto_yes_precision >= precision_target`` (vacuous if the upper zone
        is empty, which is itself a valid — if expensive — solution)

    Ties are broken towards the point with the fewest unreviewed false
    positives, then towards the wider confident-negative zone.

    Returns a dict with ``feasible``, the chosen thresholds and their full
    metric set. When no pair satisfies the constraints, the most conservative
    fallback is returned (``t_low = 0``: nothing is ever silently dropped) and
    ``feasible`` is ``False``.
    """

    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true).astype(bool)

    if y_true.sum() == 0:
        raise ValueError(
            "Cannot calibrate a recall constraint on a split with no positive "
            "documents. Check the split configuration (see prefilter README)."
        )

    candidates = _candidate_thresholds(probs, grid_steps)

    best: dict | None = None
    feasible_rows: list[dict] = []

    for t_low in candidates:
        # The recall constraint depends on t_low only, so reject early and skip
        # the whole inner loop for an infeasible lower cut.
        missed = int((y_true & (probs < t_low)).sum())
        recall = 1.0 - missed / y_true.sum()

        if recall < recall_target:
            continue

        for t_high in candidates:
            if t_high < t_low:
                continue

            metrics = evaluate_routing(probs, y_true, t_low, t_high)

            precision = metrics["auto_yes_precision"]

            # An empty upper zone has no precision to violate.
            if precision is not None and precision < precision_target:
                continue

            feasible_rows.append(metrics)

            key = (
                metrics["routed_fraction"],
                metrics["auto_yes_fp"],
                -metrics["auto_no_n"],
            )

            if best is None or key < best["_key"]:
                best = {"_key": key, **metrics}

    if best is None:
        logger.warning(
            "No (t_low, t_high) satisfies recall >= %.4f and precision >= %.4f. "
            "Falling back to t_low=0.0 (never drop a document silently).",
            recall_target,
            precision_target,
        )

        fallback = evaluate_routing(probs, y_true, 0.0, 1.0)

        return {
            "feasible": False,
            "recall_target": recall_target,
            "precision_target": precision_target,
            "n_feasible_points": 0,
            **fallback,
        }

    best.pop("_key", None)

    return {
        "feasible": True,
        "recall_target": recall_target,
        "precision_target": precision_target,
        "n_feasible_points": len(feasible_rows),
        **best,
    }


def routing_frontier(
    probs: np.ndarray,
    y_true: np.ndarray,
    precision_target: float,
    recall_targets: np.ndarray | None = None,
    grid_steps: int = 201,
) -> pd.DataFrame:
    """
    Trade-off curve: minimum LLM-call share achievable at each recall target.

    This is the figure for the meeting and the paper — x = share of documents
    routed to the LLM, y = guaranteed document-level recall. Every point is the
    cheapest router that still clears that recall.
    """

    if recall_targets is None:
        recall_targets = np.round(np.arange(0.80, 1.0001, 0.01), 4)

    rows = []

    for target in recall_targets:
        result = calibrate_thresholds(
            probs=probs,
            y_true=y_true,
            recall_target=float(target),
            precision_target=precision_target,
            grid_steps=grid_steps,
        )

        rows.append(
            {
                "recall_target": float(target),
                "feasible": result["feasible"],
                "t_low": result["t_low"],
                "t_high": result["t_high"],
                "routed_fraction": result["routed_fraction"],
                "llm_call_reduction": result["llm_call_reduction"],
                "prefilter_recall": result["prefilter_recall"],
                "auto_yes_precision": result["auto_yes_precision"],
                "missed_positives": result["missed_positives"],
                "oracle_f1": result["oracle_f1"],
                "conservative_precision": result["conservative_precision"],
                "conservative_f1": result["conservative_f1"],
            }
        )

    return pd.DataFrame(rows)


def single_threshold_sweep(
    probs: np.ndarray,
    y_true: np.ndarray,
    grid_steps: int = 201,
) -> pd.DataFrame:
    """
    Plain single-threshold sweep, for comparison against the three-zone router.
    """

    rows = []

    for threshold in _candidate_thresholds(probs, grid_steps):
        metrics = binary_metrics(y_true, np.asarray(probs) >= threshold)
        rows.append({"threshold": round(float(threshold), 6), **metrics})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_routing_frontier(
    frontier: pd.DataFrame,
    output_file: Path,
    title: str = "LLM-call volume vs. guaranteed recall",
    baseline_recall: float = 0.9833,
    operating_point: dict | None = None,
) -> Path:
    """
    Save the trade-off curve as a PNG.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    usable = frontier[frontier["feasible"]].copy()

    figure, axes = plt.subplots(figsize=(7.5, 5.0))

    if not usable.empty:
        axes.plot(
            usable["routed_fraction"] * 100,
            usable["recall_target"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color="#1f4e79",
            label="Cheapest router at each recall target",
        )

    axes.axhline(
        baseline_recall,
        color="#c00000",
        linestyle="--",
        linewidth=1.2,
        label=f"Rule-based baseline recall ({baseline_recall:.4f})",
    )

    if operating_point and operating_point.get("feasible"):
        axes.scatter(
            [operating_point["routed_fraction"] * 100],
            [operating_point["prefilter_recall"]],
            s=130,
            marker="*",
            color="#e69f00",
            edgecolor="black",
            zorder=5,
            label=(
                "Selected operating point "
                f"(t_low={operating_point['t_low']:.3f}, "
                f"t_high={operating_point['t_high']:.3f})"
            ),
        )

    axes.set_xlabel("Documents routed to the LLM (%)")
    axes.set_ylabel("Pre-filter recall (positives not silently dropped)")
    axes.set_title(title)
    axes.grid(alpha=0.3)
    axes.legend(loc="lower right", fontsize=8)

    figure.tight_layout()
    figure.savefig(output_file, dpi=160)
    plt.close(figure)

    return output_file


def plot_probability_distribution(
    probs: np.ndarray,
    y_true: np.ndarray,
    t_low: float,
    t_high: float,
    output_file: Path,
    title: str = "Pre-filter score distribution and routing zones",
) -> Path:
    """
    Save a class-conditional score histogram with the two cuts drawn on it.

    This is the diagnostic that explains *why* a given routing rate came out:
    a wide uncertain band means the classes are not separated at that recall.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true).astype(bool)

    figure, axes = plt.subplots(figsize=(7.5, 4.5))

    bins = np.linspace(0, 1, 41)

    axes.hist(
        probs[~y_true],
        bins=bins,
        alpha=0.65,
        label=f"no personal data (n={int((~y_true).sum())})",
        color="#4c72b0",
    )
    axes.hist(
        probs[y_true],
        bins=bins,
        alpha=0.65,
        label=f"contains personal data (n={int(y_true.sum())})",
        color="#c44e52",
    )

    axes.axvspan(t_low, t_high, color="#f0e442", alpha=0.25, label="routed to LLM")
    axes.axvline(t_low, color="black", linestyle="--", linewidth=1.1)
    axes.axvline(t_high, color="black", linestyle="--", linewidth=1.1)

    axes.set_xlabel("Predicted probability of personal data")
    axes.set_ylabel("Documents")
    axes.set_yscale("symlog")
    axes.set_title(title)
    axes.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(output_file, dpi=160)
    plt.close(figure)

    return output_file


def save_calibration(
    calibration: dict,
    output_file: Path,
) -> Path:
    """
    Persist the chosen operating point as JSON.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    return output_file


def load_calibration(input_file: Path) -> dict:
    """
    Load a previously saved operating point.
    """
    return json.loads(Path(input_file).read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────
# Entity-label thresholds
# ─────────────────────────────────────────────────────────────

def calibrate_entity_thresholds(
    entity_probs: np.ndarray,
    entity_true: np.ndarray,
    labels: list[str],
    default_threshold: float = 0.5,
    grid_steps: int = 201,
) -> dict[str, float]:
    """
    Fit one decision threshold per entity label on the validation split.

    A fixed 0.5 cut is wrong for these heads for a mechanical reason: the entity
    labels are rare enough (3 of 500 documents for IBAN_CODE in the pilot) that
    the head never becomes confident in absolute terms, even where its *ranking*
    is good. On the pilot the highest score assigned to a true PERSON document
    was 0.49 — a hair under the cut — so a 0.5 threshold reported precision and
    recall of exactly zero for a head with a validation PR-AUC of 0.80.

    Each threshold maximises F1 for its label. Labels with no positive
    validation document keep ``default_threshold``: there is nothing to fit
    against, and inventing a threshold from negatives alone would just be noise.

    Calibrating on validation and applying to test is the same discipline the
    binary router follows; the test split stays untouched.
    """

    entity_probs = np.asarray(entity_probs, dtype=float)
    entity_true = np.asarray(entity_true).astype(bool)

    thresholds: dict[str, float] = {}

    for index, label in enumerate(labels):
        truth = entity_true[:, index]

        if truth.sum() == 0:
            thresholds[label] = float(default_threshold)
            continue

        probs = entity_probs[:, index]

        best_threshold = float(default_threshold)
        best_f1 = -1.0

        for threshold in _candidate_thresholds(probs, grid_steps):
            metrics = binary_metrics(truth, probs >= threshold)

            # Prefer the higher threshold among ties: it is the more
            # conservative of two cuts that score identically here, and so the
            # less likely to have been chosen by noise.
            if metrics["f1"] >= best_f1:
                best_f1 = metrics["f1"]
                best_threshold = float(threshold)

        thresholds[label] = round(best_threshold, 6)

    return thresholds


def entity_metrics_at_thresholds(
    entity_probs: np.ndarray,
    entity_true: np.ndarray,
    labels: list[str],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """
    Per-label metrics using the calibrated thresholds.
    """

    rows = []

    for index, label in enumerate(labels):
        threshold = thresholds.get(label, 0.5)

        rows.append(
            {
                "entity": label,
                "threshold": round(float(threshold), 6),
                "support": int(entity_true[:, index].sum()),
                "pr_auc": round(
                    average_precision(
                        entity_true[:, index].astype(bool), entity_probs[:, index]
                    ),
                    6,
                ),
                **binary_metrics(
                    entity_true[:, index].astype(bool),
                    entity_probs[:, index] >= threshold,
                ),
            }
        )

    return pd.DataFrame(rows)
