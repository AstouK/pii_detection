"""
Checks for the train / validation / test generalisation gap.

These cases use synthetic probabilities so the right answer is known without a
model. The load-bearing claim is that the overfit check scores splits at a
**frozen** ``(t_low, t_high)`` and never re-fits the router on train or test.

Run:

    python -m pytest classification/prefilter/tests/test_overfit_check.py -q
"""

from __future__ import annotations

import numpy as np

from classification.prefilter.overfit_check import (
    compute_gaps,
    interpret_gaps,
    score_split_probs,
    split_gap_payload,
)
from classification.prefilter.thresholds import (
    calibrate_thresholds,
    evaluate_routing,
)


def _row(split: str, f1: float, pr_auc: float, recall: float) -> dict:
    return {
        "split": split,
        "f1": f1,
        "pr_auc": pr_auc,
        "prefilter_recall": recall,
    }


# ─────────────────────────────────────────────────────────────
# Frozen thresholds
# ─────────────────────────────────────────────────────────────

def test_score_split_probs_keeps_the_thresholds_it_was_given():
    """
    Scores that would collapse to a near-zero band if re-calibrated still
    report the frozen (t_low, t_high) that was passed in.
    """

    probs = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    y_true = np.array([False, False, False, True, True, True])

    fitted = calibrate_thresholds(
        probs=probs,
        y_true=y_true,
        recall_target=0.98,
        precision_target=0.90,
    )

    frozen_low, frozen_high = 0.3, 0.7
    row = score_split_probs(
        "train",
        probs,
        y_true,
        t_low=frozen_low,
        t_high=frozen_high,
    )

    assert row["t_low"] == frozen_low
    assert row["t_high"] == frozen_high
    assert (row["t_low"], row["t_high"]) != (fitted["t_low"], fitted["t_high"])


def test_score_split_probs_matches_evaluate_routing_at_the_given_cut():
    probs = np.array([0.05, 0.40, 0.55, 0.95])
    y_true = np.array([False, True, False, True])
    t_low, t_high = 0.2, 0.8

    row = score_split_probs("validation", probs, y_true, t_low, t_high)
    expected = evaluate_routing(probs, y_true, t_low, t_high)

    assert row["routed_fraction"] == expected["routed_fraction"]
    assert row["prefilter_recall"] == expected["prefilter_recall"]
    assert row["missed_positives"] == expected["missed_positives"]
    assert row["auto_yes_precision"] == expected["auto_yes_precision"]
    assert row["t_low"] == expected["t_low"]
    assert row["t_high"] == expected["t_high"]


def test_score_split_probs_does_not_call_calibrate_thresholds(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("calibrate_thresholds must not run during scoring")

    monkeypatch.setattr(
        "classification.prefilter.thresholds.calibrate_thresholds",
        _boom,
    )

    score_split_probs(
        "test",
        np.array([0.1, 0.9]),
        np.array([False, True]),
        t_low=0.25,
        t_high=0.75,
    )


# ─────────────────────────────────────────────────────────────
# Gaps
# ─────────────────────────────────────────────────────────────

def test_compute_gaps_are_train_minus_val_and_val_minus_test():
    rows = [
        _row("train", 1.0, 1.0, 1.0),
        _row("validation", 0.8, 0.9, 0.95),
        _row("test", 0.6, 0.7, 0.85),
    ]

    gaps = compute_gaps(rows)

    assert gaps["train_minus_val_f1"] == 0.2
    assert gaps["val_minus_test_f1"] == 0.2
    assert gaps["train_minus_val_pr_auc"] == 0.1
    assert gaps["val_minus_test_pr_auc"] == 0.2
    assert gaps["train_minus_val_prefilter_recall"] == 0.05
    assert gaps["val_minus_test_prefilter_recall"] == 0.1


def test_compute_gaps_return_none_when_a_split_is_missing():
    gaps = compute_gaps([_row("train", 1.0, 1.0, 1.0)])

    assert gaps["train_minus_val_f1"] is None
    assert gaps["val_minus_test_f1"] is None


# ─────────────────────────────────────────────────────────────
# Interpretation
# ─────────────────────────────────────────────────────────────

def test_interpret_gaps_flags_train_val_overfit():
    rows = [
        _row("train", 1.0, 1.0, 1.0),
        _row("validation", 0.80, 0.82, 0.90),
        _row("test", 0.80, 0.81, 0.90),
    ]
    reading = interpret_gaps(rows, compute_gaps(rows))

    assert "train_val_overfit" in reading["flags"]
    assert "dataset_too_easy" not in reading["flags"]
    assert "memorised" in reading["reading"]


def test_interpret_gaps_flags_val_test_calibration_gap():
    rows = [
        _row("train", 0.90, 0.91, 0.98),
        _row("validation", 0.90, 0.91, 0.98),
        _row("test", 0.70, 0.72, 0.80),
    ]
    reading = interpret_gaps(rows, compute_gaps(rows))

    assert "val_test_calibration_gap" in reading["flags"]
    assert "did not generalise" in reading["reading"]


def test_interpret_gaps_flags_dataset_too_easy():
    rows = [
        _row("train", 1.0, 1.0, 1.0),
        _row("validation", 1.0, 1.0, 1.0),
        _row("test", 1.0, 1.0, 1.0),
    ]
    reading = interpret_gaps(rows, compute_gaps(rows))

    assert reading["flags"] == ["dataset_too_easy"]
    assert "unmeasurable" in reading["reading"]


def test_split_gap_payload_carries_frozen_cuts_into_the_artifact():
    probs = np.array([0.05, 0.95])
    y_true = np.array([False, True])
    rows = [
        score_split_probs(split, probs, y_true, t_low=0.2, t_high=0.8)
        for split in ("train", "validation", "test")
    ]

    payload = split_gap_payload(rows, t_low=0.2, t_high=0.8)

    assert payload["t_low"] == 0.2
    assert payload["t_high"] == 0.8
    assert payload["calibrated_on"] == "validation"
    assert payload["interpretation"]["flags"] == ["dataset_too_easy"]
    assert list(payload["table"]["split"]) == ["train", "validation", "test"]
