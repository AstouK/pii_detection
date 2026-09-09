"""
Routing and final-prediction checks for the transformer strategies.

Two strategies are exercised here:

    rule_plus_distilbert            Sweep 1, then DistilBERT on the ambiguous
                                    subset, no LLM stage.
    rule_plus_distilbert_plus_qwen  Sweep 1, then DistilBERT, then Qwen on the
                                    documents DistilBERT is uncertain about.

The BERT and Qwen calls are mocked, so these tests are hermetic: they check the
*routing contract* (who sees BERT, who reaches the LLM, how the final decision
is fused) rather than model quality. The real inference has its own coverage in
``classification/prefilter/tests``.

Run:

    python -m pytest classification/infrastructure/tests/ -q
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from classification.infrastructure import strategy_runner
from classification.infrastructure.metadata import (
    compute_bert_final_prediction,
    compute_hybrid_final_prediction,
)


ENTITY_YES_NO_COLS = [
    "PERSON_yes_no",
    "EMAIL_ADDRESS_yes_no",
]


def _base_sweep1_frame() -> pd.DataFrame:
    """
    A tiny Sweep 1 output with one document per routing outcome.

    Routing states, as Sweep 1 leaves them:
        DOC-LOCAL-PII      strong detection, detected_pii=True, not ambiguous
        DOC-LOCAL-NON-PII  no signal, detected_pii=False, not ambiguous
        DOC-AMB-*          ambiguous, needs_llm_review=True -> goes to BERT
    """

    rows = [
        # document_id, detected_pii, needs_llm_review(=ambiguous), truth
        ("DOC-LOCAL-PII", True, False, True),
        ("DOC-LOCAL-NON-PII", False, False, False),
        ("DOC-AMB-CONF-PII", False, True, True),
        ("DOC-AMB-CONF-NON", False, True, False),
        ("DOC-AMB-ROUTED-POS", False, True, True),
        ("DOC-AMB-ROUTED-NEG", False, True, False),
    ]

    frame = pd.DataFrame(
        {
            "document_id": [r[0] for r in rows],
            "full_text": [f"text for {r[0]}" for r in rows],
            "detected_pii": [r[1] for r in rows],
            "needs_llm_review": [r[2] for r in rows],
            "contains_personal_data": [r[3] for r in rows],
            "entities": [[] for _ in rows],
            "per_type_conf": [{} for _ in rows],
        }
    )

    for column in ENTITY_YES_NO_COLS:
        frame[column] = "no"

    return frame


def _make_bundle(t_low: float = 0.3, t_high: float = 0.7,
                 standalone_threshold: float = 0.5) -> dict:
    """A stand-in for :func:`load_prefilter_bundle` with no real model."""

    config = types.SimpleNamespace(standalone_threshold=standalone_threshold)

    return {
        "run_name": "fake_checkpoint",
        "config": config,
        "t_low": t_low,
        "t_high": t_high,
    }


def _fake_score_factory(prob_map: dict[str, float]):
    """
    Build a replacement for :func:`score_ambiguous_documents`.

    It reproduces the real column contract for the masked rows using the given
    per-document probabilities, so the runner and final-prediction logic can be
    tested without loading a checkpoint.
    """

    def _fake_score(df: pd.DataFrame, mask: pd.Series, bundle: dict):
        result = df.copy()

        # Neutral defaults, matching the real implementation.
        for column, default in {
            "pii_probability": np.nan,
            "routing_zone": "",
            "routed_to_llm": False,
            "needs_bert_review": False,
            "bert_request_success": False,
            "bert_runtime_seconds": 0.0,
        }.items():
            if column not in result.columns:
                result[column] = default

        mask = mask.fillna(False).astype(bool)
        t_low = bundle["t_low"]
        t_high = bundle["t_high"]

        for idx in result.index[mask]:
            prob = float(prob_map[result.at[idx, "document_id"]])

            if prob < t_low:
                zone = "confident_non_pii"
                routed = False
            elif prob > t_high:
                zone = "confident_pii"
                routed = False
            else:
                zone = "routed_to_llm"
                routed = True

            result.at[idx, "pii_probability"] = prob
            result.at[idx, "routing_zone"] = zone
            result.at[idx, "routed_to_llm"] = routed
            result.at[idx, "needs_bert_review"] = True
            result.at[idx, "bert_request_success"] = True
            result.at[idx, "bert_runtime_seconds"] = 0.001

        runtime_info = {"documents_scored": int(mask.sum())}
        return result, runtime_info

    return _fake_score


def _fake_run_llm_factory(llm_map: dict[str, bool], succeed: bool = True):
    """Replacement for ``run_llm`` that only touches ``needs_llm_review`` rows."""

    def _fake_run_llm(df: pd.DataFrame, model_id: str, prompt_version: str):
        df["llm_pii"] = False
        df["llm_request_success"] = False
        df["llm_prompt_tokens"] = 0

        flagged = df[df["needs_llm_review"].fillna(False).astype(bool)]

        # Record which documents the LLM was actually asked about; the test
        # asserts on this to prove BERT-confident rows never reach the LLM.
        _fake_run_llm.seen_documents = list(flagged["document_id"])

        for idx in flagged.index:
            doc_id = df.at[idx, "document_id"]
            df.at[idx, "llm_pii"] = bool(llm_map.get(doc_id, False))
            df.at[idx, "llm_request_success"] = succeed

        return df

    _fake_run_llm.seen_documents = []
    return _fake_run_llm


# ─────────────────────────────────────────────────────────────
# Pure final-prediction logic
# ─────────────────────────────────────────────────────────────

def test_bert_final_prediction_zones():
    """Rule + BERT: local PII/non-PII plus the standalone threshold cut."""

    frame = pd.DataFrame(
        {
            "document_id": ["a", "b", "c", "d"],
            "detected_pii": [True, False, False, False],
            "needs_bert_review": [False, False, True, True],
            "pii_probability": [np.nan, np.nan, 0.9, 0.1],
        }
    )

    result = compute_bert_final_prediction(frame, standalone_threshold=0.5)

    # a: local PII -> positive; b: local non-PII, never scored -> negative
    # c: scored, prob >= 0.5 -> positive; d: scored, prob < 0.5 -> negative
    assert result["predicted_pii"].tolist() == [True, False, True, False]
    assert result["final_pii"].tolist() == [True, False, True, False]


def test_hybrid_final_prediction_confident_zones_ignore_llm():
    """BERT-confident decisions must not be overturned by the LLM flag."""

    frame = pd.DataFrame(
        {
            "document_id": ["conf_pii", "conf_non", "detected"],
            "detected_pii": [False, False, True],
            "needs_bert_review": [True, True, False],
            "routing_zone": ["confident_pii", "confident_non_pii", ""],
            "routed_to_llm": [False, False, False],
            # An LLM decision exists on these rows but must be ignored, because
            # none of them was routed to the LLM.
            "llm_pii": [False, True, False],
        }
    )

    result = compute_hybrid_final_prediction(frame)

    assert result["predicted_pii"].tolist() == [True, False, True]


def test_hybrid_final_prediction_defers_to_llm_when_routed():
    """Only routed rows take the LLM decision, in both directions."""

    frame = pd.DataFrame(
        {
            "document_id": ["routed_pos", "routed_neg"],
            "detected_pii": [False, False],
            "needs_bert_review": [True, True],
            "routing_zone": ["routed_to_llm", "routed_to_llm"],
            "routed_to_llm": [True, True],
            "llm_pii": [True, False],
        }
    )

    result = compute_hybrid_final_prediction(frame)

    assert result["predicted_pii"].tolist() == [True, False]


# ─────────────────────────────────────────────────────────────
# Runner: rule + DistilBERT (no LLM)
# ─────────────────────────────────────────────────────────────

def test_rule_plus_bert_runner(tmp_path, monkeypatch):
    from classification.prefilter import predict as predict_module

    prob_map = {
        "DOC-AMB-CONF-PII": 0.95,
        "DOC-AMB-CONF-NON": 0.02,
        "DOC-AMB-ROUTED-POS": 0.5,
        "DOC-AMB-ROUTED-NEG": 0.5,
    }

    monkeypatch.setattr(
        predict_module, "load_prefilter_bundle",
        lambda *a, **k: _make_bundle(),
    )
    monkeypatch.setattr(
        predict_module, "score_ambiguous_documents",
        _fake_score_factory(prob_map),
    )

    output_file, runtime, usage = strategy_runner.run_strategy_pipeline(
        base_df=_base_sweep1_frame(),
        strategy="rule_plus_distilbert",
        run_dir=tmp_path,
        run_id="test_run",
        dataset_version="v1",
        prompt_version="pii_review_v1",
    )

    df = pd.read_csv(output_file)
    by_id = df.set_index("document_id")

    # BERT saw exactly the four ambiguous documents.
    assert int(df["needs_bert_review"].fillna(False).astype(bool).sum()) == 4

    # No LLM stage in this strategy.
    assert not df["needs_llm_review"].fillna(False).astype(bool).any()
    assert usage["llm_requests_attempted"] == 0
    assert usage["bert_requests_attempted"] == 4

    # Final decisions: local PII and the high-probability ambiguous doc are
    # positive; local non-PII and the low/borderline ambiguous docs at the
    # 0.5 standalone cut behave as expected.
    assert bool(by_id.loc["DOC-LOCAL-PII", "predicted_pii"])
    assert not bool(by_id.loc["DOC-LOCAL-NON-PII", "predicted_pii"])
    assert bool(by_id.loc["DOC-AMB-CONF-PII", "predicted_pii"])
    assert not bool(by_id.loc["DOC-AMB-CONF-NON", "predicted_pii"])

    # Output contract: ground truth, strategy label and metadata are present.
    assert "contains_personal_data" in df.columns
    assert set(df["strategy"]) == {"rule_plus_distilbert"}
    assert set(df["model_family"]) == {"bert"}


# ─────────────────────────────────────────────────────────────
# Runner: rule + DistilBERT + Qwen (hybrid)
# ─────────────────────────────────────────────────────────────

def test_hybrid_runner_routes_only_uncertain_to_llm(tmp_path, monkeypatch):
    from classification.prefilter import predict as predict_module

    prob_map = {
        "DOC-AMB-CONF-PII": 0.95,   # confident_pii -> no LLM
        "DOC-AMB-CONF-NON": 0.02,   # confident_non_pii -> no LLM
        "DOC-AMB-ROUTED-POS": 0.5,  # routed -> LLM says PII
        "DOC-AMB-ROUTED-NEG": 0.5,  # routed -> LLM says non-PII
    }
    llm_map = {
        "DOC-AMB-ROUTED-POS": True,
        "DOC-AMB-ROUTED-NEG": False,
    }

    fake_run_llm = _fake_run_llm_factory(llm_map, succeed=True)

    monkeypatch.setattr(
        predict_module, "load_prefilter_bundle",
        lambda *a, **k: _make_bundle(),
    )
    monkeypatch.setattr(
        predict_module, "score_ambiguous_documents",
        _fake_score_factory(prob_map),
    )
    monkeypatch.setattr(strategy_runner, "run_llm", fake_run_llm)

    output_file, runtime, usage = strategy_runner.run_strategy_pipeline(
        base_df=_base_sweep1_frame(),
        strategy="rule_plus_distilbert_plus_qwen",
        run_dir=tmp_path,
        run_id="test_run",
        dataset_version="v1",
        prompt_version="pii_review_v1",
    )

    # Qwen was asked about exactly the two routed documents.
    assert set(fake_run_llm.seen_documents) == {
        "DOC-AMB-ROUTED-POS",
        "DOC-AMB-ROUTED-NEG",
    }
    assert usage["llm_requests_attempted"] == 2
    assert usage["llm_requests_successful"] == 2
    assert usage["bert_requests_attempted"] == 4

    df = pd.read_csv(output_file)
    by_id = df.set_index("document_id")

    assert bool(by_id.loc["DOC-LOCAL-PII", "predicted_pii"])
    assert not bool(by_id.loc["DOC-LOCAL-NON-PII", "predicted_pii"])
    assert bool(by_id.loc["DOC-AMB-CONF-PII", "predicted_pii"])
    assert not bool(by_id.loc["DOC-AMB-CONF-NON", "predicted_pii"])
    assert bool(by_id.loc["DOC-AMB-ROUTED-POS", "predicted_pii"])
    assert not bool(by_id.loc["DOC-AMB-ROUTED-NEG", "predicted_pii"])

    # Hybrid metadata: provider comes from the LLM stage, family is combined.
    assert set(df["strategy"]) == {"rule_plus_distilbert_plus_qwen"}
    assert set(df["model_family"]) == {"bert+llm"}
    assert set(df["prediction_source"]) == {"hybrid"}


def test_hybrid_runner_survives_failed_llm_request(tmp_path, monkeypatch):
    """A failed LLM request must not crash the run; the row stays non-PII."""

    from classification.prefilter import predict as predict_module

    prob_map = {
        "DOC-AMB-CONF-PII": 0.95,
        "DOC-AMB-CONF-NON": 0.02,
        "DOC-AMB-ROUTED-POS": 0.5,
        "DOC-AMB-ROUTED-NEG": 0.5,
    }
    # LLM "fails": returns non-PII and success=False for every routed row.
    fake_run_llm = _fake_run_llm_factory({}, succeed=False)

    monkeypatch.setattr(
        predict_module, "load_prefilter_bundle",
        lambda *a, **k: _make_bundle(),
    )
    monkeypatch.setattr(
        predict_module, "score_ambiguous_documents",
        _fake_score_factory(prob_map),
    )
    monkeypatch.setattr(strategy_runner, "run_llm", fake_run_llm)

    output_file, _runtime, usage = strategy_runner.run_strategy_pipeline(
        base_df=_base_sweep1_frame(),
        strategy="rule_plus_distilbert_plus_qwen",
        run_dir=tmp_path,
        run_id="test_run",
        dataset_version="v1",
        prompt_version="pii_review_v1",
    )

    df = pd.read_csv(output_file)
    by_id = df.set_index("document_id")

    # Two attempts, none successful.
    assert usage["llm_requests_attempted"] == 2
    assert usage["llm_requests_successful"] == 0

    # BERT-confident decisions still hold despite the LLM failure.
    assert bool(by_id.loc["DOC-AMB-CONF-PII", "predicted_pii"])
    assert not bool(by_id.loc["DOC-AMB-ROUTED-POS", "predicted_pii"])
