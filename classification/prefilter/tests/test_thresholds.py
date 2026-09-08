"""
Checks for the routing logic.

The three-zone router is the actual contribution of this work package, and its
metrics are easy to get subtly wrong in a way that still looks plausible on real
data — an off-by-one in a zone boundary, or a recall definition that quietly
counts routed positives as losses. These cases use synthetic scores where the
right answer is known by construction.

Run:

    python -m pytest classification/prefilter/tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

from classification.prefilter.data import (
    make_stratified_split,
    to_bool_series,
    validate_split,
)
from classification.prefilter.thresholds import (
    average_precision,
    binary_metrics,
    calibrate_entity_thresholds,
    calibrate_thresholds,
    evaluate_routing,
    routing_frontier,
    zone_masks,
)

import pandas as pd


# ─────────────────────────────────────────────────────────────
# Zones
# ─────────────────────────────────────────────────────────────

def test_zones_partition_every_document():
    probs = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

    auto_no, routed, auto_yes = zone_masks(probs, 0.3, 0.7)

    # Every document lands in exactly one zone.
    assert (auto_no.astype(int) + routed + auto_yes == 1).all()


def test_zone_boundaries_are_inclusive_for_routing():
    # A score sitting exactly on a cut is routed, not auto-decided: the
    # boundaries are the uncertain band's edges, so ties escalate.
    probs = np.array([0.3, 0.7])

    auto_no, routed, auto_yes = zone_masks(probs, 0.3, 0.7)

    assert routed.all()
    assert not auto_no.any()
    assert not auto_yes.any()


# ─────────────────────────────────────────────────────────────
# Routing metrics
# ─────────────────────────────────────────────────────────────

def test_routed_positive_does_not_count_against_recall():
    # One positive scored 0.5, routed. It reaches the LLM, so the pre-filter
    # has not lost it: recall must stay 1.0.
    probs = np.array([0.5, 0.9, 0.01])
    y_true = np.array([True, True, False])

    metrics = evaluate_routing(probs, y_true, t_low=0.2, t_high=0.8)

    assert metrics["prefilter_recall"] == 1.0
    assert metrics["missed_positives"] == 0
    assert metrics["routed_n"] == 1


def test_positive_below_t_low_is_an_irrecoverable_miss():
    # Same positive, now below t_low. Nothing downstream revisits it.
    probs = np.array([0.5, 0.9, 0.01])
    y_true = np.array([True, True, False])

    metrics = evaluate_routing(probs, y_true, t_low=0.6, t_high=0.8)

    assert metrics["missed_positives"] == 1
    assert metrics["prefilter_recall"] == 0.5


def test_llm_calls_avoided_is_the_complement_of_routing():
    probs = np.linspace(0, 1, 10)
    y_true = probs > 0.5

    metrics = evaluate_routing(probs, y_true, t_low=0.3, t_high=0.7)

    assert metrics["routed_n"] + metrics["llm_calls_avoided"] == metrics["n"]
    assert metrics["routed_fraction"] + metrics["llm_call_reduction"] == 1.0


def test_oracle_beats_conservative_on_precision():
    # The two end-to-end readings differ exactly on the routed negatives:
    # the oracle resolves them correctly, the conservative reading flags them.
    probs = np.array([0.5, 0.5, 0.95, 0.02])
    y_true = np.array([True, False, True, False])

    metrics = evaluate_routing(probs, y_true, t_low=0.1, t_high=0.9)

    assert metrics["oracle_precision"] == 1.0
    assert metrics["conservative_precision"] < 1.0
    assert metrics["conservative_recall"] == 1.0


# ─────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────

def _separable_scores(n: int = 200, overlap: float = 0.0):
    """
    Half positives near 1, half negatives near 0, with a controllable overlap.
    """
    rng = np.random.default_rng(0)

    positives = rng.uniform(0.6 - overlap, 1.0, n // 2)
    negatives = rng.uniform(0.0, 0.4 + overlap, n // 2)

    probs = np.concatenate([positives, negatives])
    y_true = np.concatenate([np.ones(n // 2, bool), np.zeros(n // 2, bool)])

    return probs, y_true


def test_separable_scores_need_no_llm_calls():
    probs, y_true = _separable_scores()

    result = calibrate_thresholds(
        probs, y_true, recall_target=0.98, precision_target=0.9
    )

    assert result["feasible"]
    assert result["routed_fraction"] == 0.0
    assert result["prefilter_recall"] == 1.0


def test_overlapping_scores_force_routing():
    # With the classes genuinely overlapping, holding recall at 1.0 is only
    # possible by escalating part of the overlap.
    probs, y_true = _separable_scores(overlap=0.35)

    result = calibrate_thresholds(
        probs, y_true, recall_target=1.0, precision_target=0.95
    )

    assert result["feasible"]
    assert result["routed_fraction"] > 0.0
    assert result["prefilter_recall"] == 1.0


def test_calibration_always_meets_its_recall_target():
    probs, y_true = _separable_scores(overlap=0.3)

    for target in (0.9, 0.95, 0.98, 1.0):
        result = calibrate_thresholds(
            probs, y_true, recall_target=target, precision_target=0.8
        )
        assert result["prefilter_recall"] >= target - 1e-9, target


def test_unreachable_precision_target_empties_the_confident_pii_zone():
    # No upper zone can reach this precision, so the router must not auto-approve
    # anything: everything it does not confidently reject is escalated. It may
    # still carve out a confident-negative zone where that costs no positive —
    # cheaper than routing everything, and just as safe.
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.4, 0.6, 100)
    y_true = rng.random(100) < 0.5

    result = calibrate_thresholds(
        probs, y_true, recall_target=1.0, precision_target=1.01
    )

    assert result["auto_yes_n"] == 0
    assert result["missed_positives"] == 0
    assert result["prefilter_recall"] == 1.0


def test_ties_on_cost_are_broken_towards_keeping_positives():
    # Two operating points cost zero LLM calls. One drops a positive to widen
    # the confident-negative zone; the recall target permits it. The safer one
    # must still win — a dropped positive is never worth a wider zone.
    probs = np.array([0.90, 0.85, 0.30, 0.05, 0.04, 0.03, 0.02, 0.01])
    y_true = np.array([True, True, True, False, False, False, False, False])

    result = calibrate_thresholds(
        probs, y_true, recall_target=0.66, precision_target=0.9
    )

    assert result["routed_fraction"] == 0.0
    assert result["missed_positives"] == 0
    assert result["prefilter_recall"] == 1.0


def test_frontier_is_monotone_in_recall():
    # Demanding more recall can never get cheaper.
    probs, y_true = _separable_scores(overlap=0.3)

    frontier = routing_frontier(
        probs, y_true, precision_target=0.9,
        recall_targets=np.array([0.90, 0.95, 0.98, 1.0]),
    )

    routed = frontier[frontier["feasible"]]["routed_fraction"].to_numpy()

    assert (np.diff(routed) >= -1e-9).all()


def test_calibration_refuses_a_split_without_positives():
    with pytest.raises(ValueError, match="no positive documents"):
        calibrate_thresholds(
            np.array([0.1, 0.2]),
            np.array([False, False]),
            recall_target=0.98,
            precision_target=0.9,
        )


# ─────────────────────────────────────────────────────────────
# Entity thresholds
# ─────────────────────────────────────────────────────────────

def test_entity_threshold_recovers_a_head_that_never_reaches_0_5():
    # The pilot's failure mode: correct ranking, scores compressed below 0.5.
    entity_probs = np.array([[0.49], [0.48], [0.10], [0.11]])
    entity_true = np.array([[1], [1], [0], [0]])

    thresholds = calibrate_entity_thresholds(
        entity_probs, entity_true, labels=["PERSON"]
    )

    assert thresholds["PERSON"] <= 0.48
    predicted = entity_probs[:, 0] >= thresholds["PERSON"]
    assert binary_metrics(entity_true[:, 0].astype(bool), predicted)["f1"] == 1.0


def test_entity_threshold_without_positives_keeps_the_default():
    entity_probs = np.array([[0.9], [0.8]])
    entity_true = np.array([[0], [0]])

    thresholds = calibrate_entity_thresholds(
        entity_probs, entity_true, labels=["URL"], default_threshold=0.5
    )

    assert thresholds["URL"] == 0.5


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def test_average_precision_is_one_for_perfect_ranking():
    assert average_precision(
        np.array([True, True, False, False]),
        np.array([0.9, 0.8, 0.2, 0.1]),
    ) == pytest.approx(1.0)


def test_binary_metrics_match_hand_computed_values():
    metrics = binary_metrics(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
    )

    assert (metrics["TP"], metrics["FN"], metrics["FP"], metrics["TN"]) == (1, 1, 1, 1)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["accuracy"] == 0.5


# ─────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────

def _frame(n_positive: int, n_negative: int, duplicate_every: int = 0):
    rows = []

    for index in range(n_positive + n_negative):
        is_positive = index < n_positive
        text = (
            f"shared document {index % duplicate_every}"
            if duplicate_every
            else f"document {index}"
        )
        rows.append(
            {
                "document_id": f"DOC-{index:04d}",
                "full_text": text,
                "contains_personal_data": "yes" if is_positive else "no",
            }
        )

    return pd.DataFrame(rows)


def test_validate_split_rejects_an_all_negative_validation_split():
    # This is the pilot's actual shape: every positive in train.
    df = _frame(20, 80)
    df["recommended_split"] = ["train"] * 60 + ["validation"] * 20 + ["test"] * 20

    problems = validate_split(df)

    assert any("validation" in problem for problem in problems)
    assert any("0 positive" in problem for problem in problems)


def test_validate_split_accepts_a_healthy_split():
    df = _frame(30, 70)
    rng = np.random.default_rng(3)
    df["recommended_split"] = rng.choice(
        ["train", "validation", "test"], size=len(df), p=[0.6, 0.2, 0.2]
    )

    # Force at least one positive and negative into each split.
    for offset, name in enumerate(["train", "validation", "test"]):
        df.loc[offset, "recommended_split"] = name           # positive
        df.loc[30 + offset, "recommended_split"] = name      # negative

    assert validate_split(df) == []


def test_stratified_split_puts_positives_in_every_split():
    df = _frame(30, 170)

    splits = make_stratified_split(
        df, validation_fraction=0.15, test_fraction=0.15, seed=42
    )
    df = df.assign(split=splits)

    for name in ("train", "validation", "test"):
        subset = df[df["split"] == name]
        assert len(subset) > 0, name
        assert to_bool_series(subset["contains_personal_data"]).sum() > 0, name


def test_duplicate_documents_never_span_splits():
    # 200 rows over 20 distinct texts: without grouping, near-certain leakage.
    df = _frame(40, 160, duplicate_every=20)

    splits = make_stratified_split(
        df, validation_fraction=0.15, test_fraction=0.15, seed=42,
        group_duplicate_texts=True,
    )
    df = df.assign(split=splits)

    assert (df.groupby("full_text")["split"].nunique() == 1).all()
