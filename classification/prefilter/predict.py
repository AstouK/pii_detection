"""
Run the pre-filter and write evaluation-compatible prediction output.

    python -m classification.prefilter.predict
    python -m classification.prefilter.predict --split all
    python -m classification.prefilter.predict --run-name distilbert_prefilter

Writes to ``classification/results/runs/<run_id>/``, the same place the rest of
the classification pipeline writes, so ``evaluate --run-id <run_id>`` picks it
up with no further wiring:

    rule_plus_bert.csv   the routed pipeline (registered strategy)
    bert_prefilter.csv   the standalone model, no routing applied
    run_metadata.json    routing and runtime metrics

Two outputs, because they answer different questions. ``bert_prefilter.csv`` is
"how good is the model"; ``rule_plus_bert.csv`` is "what does the pipeline do
with it", and only the second one's ``routed_to_llm`` column carries the number
this work package is judged on.

What ``predicted_pii`` means in each file
-----------------------------------------
``bert_prefilter.csv``  ``p >= standalone_threshold``. The model's own call.

``rule_plus_bert.csv``  confident zones as decided; **routed documents count as
                        predicted-PII**. The LLM is not run here, so the routed
                        rows need some value, and escalated-means-flagged is
                        the GDPR-safe reading: a document under review is
                        treated as potential personal data until a reviewer
                        says otherwise. It also keeps the file honest about
                        cost — those rows are precision debt this stage has not
                        paid off. ``routed_to_llm`` marks every one of them, so
                        any other assumption can be recomputed downstream.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.logging_config import setup_logging

from classification.prefilter.config import (
    BINARY_LABEL_COL,
    CLASSIFICATION_RUNS_DIR,
    CONTEXT_COLS,
    DOCUMENT_ID_COL,
    ENTITY_LABELS,
    MODEL_FAMILY,
    PIPELINE_NAME,
    PREDICTION_SOURCE,
    PREDICTION_STAGE,
    PROVIDER,
    ROUTED_STRATEGY,
    STANDALONE_STRATEGY,
    entity_label_columns,
    run_artifacts_dir,
)
from classification.prefilter.data import (
    build_dataloader,
    build_dataset,
    extract_labels,
    load_dataset,
    resolve_splits,
)
from classification.prefilter.model import load_checkpoint, load_tokenizer
from classification.prefilter.thresholds import (
    evaluate_routing,
    load_calibration,
    zone_masks,
)
from classification.prefilter.train import predict_probabilities, resolve_device

setup_logging()
logger = logging.getLogger(__name__)


def make_run_id() -> str:
    """
    Run id in the pipeline's format: ``YYYYMMDD_HHMMSS``.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_per_type_conf(
    entity_probs: np.ndarray,
    thresholds: dict[str, float],
) -> list[str]:
    """
    Build the ``per_type_conf`` column the evaluation reads for entity metrics.

    ``classification.evaluation.metrics.entity_detected`` treats an entity as
    detected when its type is a *key* of this dict, so only labels above their
    own calibrated threshold are included. The confidence values ride along for
    error analysis; the evaluation ignores them.
    """

    rows = []

    for probabilities in entity_probs:
        detected = {
            label: round(float(probability), 4)
            for label, probability in zip(ENTITY_LABELS, probabilities)
            if probability >= thresholds.get(label, 0.5)
        }
        rows.append(json.dumps(detected))

    return rows


def build_entity_predictions(
    entity_probs: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, list[str]]:
    """
    Per-entity predictions from the multi-label head, as ``<ENTITY>_predicted``.

    The suffix matters. ``<ENTITY>_yes_no`` is the *ground-truth* spelling, and
    ``evaluation/metrics.py::get_entity_types_from_columns`` derives the entity
    vocabulary by stripping ``_yes_no`` off every column that ends with it — so
    a predicted column named ``predicted_PERSON_yes_no`` is read back as an
    entity type literally called ``predicted_PERSON`` and produces twelve
    phantom entity rows in metrics.csv. ``_predicted`` keeps the head's output
    in the file without colliding with that scan.

    The evaluation's own per-entity metrics come from ``per_type_conf``; these
    columns exist for error analysis and for anyone comparing the two heads.
    """

    return {
        f"{label}_predicted": [
            "yes" if probability >= thresholds.get(label, 0.5) else "no"
            for probability in entity_probs[:, index]
        ]
        for index, label in enumerate(ENTITY_LABELS)
    }


def build_output_frame(
    source: pd.DataFrame,
    probs: np.ndarray,
    entity_probs: np.ndarray,
    strategy: str,
    run_id: str,
    model_name: str,
    inference_ms: float,
    entity_thresholds: dict[str, float],
    predicted_pii: np.ndarray,
    routing: dict | None = None,
) -> pd.DataFrame:
    """
    Assemble one prediction CSV.

    Column contract, from ``infrastructure/metadata.py::add_sweep1_metadata``
    and ``evaluation/metrics.py``:

        document_id, predicted_pii, run_id, strategy, provider, model_family,
        model_name, prediction_stage, pipeline_name, prediction_source

    plus the ground-truth columns (``contains_personal_data`` and the twelve
    ``<ENTITY>_yes_no``), because the evaluation reads ground truth out of the
    prediction file rather than re-joining the dataset.
    """

    output = pd.DataFrame()

    # ── Identity and ground truth ───────────────────────────
    output[DOCUMENT_ID_COL] = source[DOCUMENT_ID_COL].values
    output[BINARY_LABEL_COL] = source[BINARY_LABEL_COL].values

    for column in entity_label_columns():
        output[column] = source[column].values

    for column in CONTEXT_COLS:
        if column in source.columns:
            output[column] = source[column].values

    # ── Prediction ──────────────────────────────────────────
    output["predicted_pii"] = np.asarray(predicted_pii).astype(bool)
    output["pii_probability"] = np.round(probs, 6)
    output["per_type_conf"] = build_per_type_conf(entity_probs, entity_thresholds)

    for column, values in build_entity_predictions(
        entity_probs, entity_thresholds
    ).items():
        output[column] = values

    # ── Routing ─────────────────────────────────────────────
    if routing is not None:
        output["routing_zone"] = routing["zone"]
        output["routed_to_llm"] = routing["routed"]
        # The runtime aggregator counts LLM attempts from `needs_llm_review`
        # and review routing from `needs_review`
        # (infrastructure/runtime.py), so both are written with the same value.
        output["needs_llm_review"] = routing["routed"]
        output["needs_review"] = routing["routed"]
        output["t_low"] = routing["t_low"]
        output["t_high"] = routing["t_high"]
    else:
        output["routed_to_llm"] = False

    # ── Runtime, for the cost analysis ──────────────────────
    output["inference_ms"] = round(inference_ms, 4)
    output["needs_bert_review"] = True
    output["bert_request_success"] = True
    output["bert_runtime_seconds"] = round(inference_ms / 1000.0, 6)

    # ── Standard metadata ───────────────────────────────────
    output["run_id"] = run_id
    output["strategy"] = strategy
    output["provider"] = PROVIDER
    output["model_family"] = MODEL_FAMILY
    output["model_name"] = model_name
    output["prediction_stage"] = PREDICTION_STAGE
    output["pipeline_name"] = PIPELINE_NAME
    output["prediction_source"] = PREDICTION_SOURCE

    return output


def build_run_metadata(
    run_id: str,
    strategies: list[str],
    saved_files: list[Path],
    routing_metrics: dict,
    runtime_metrics: dict,
    calibration: dict,
) -> dict:
    """
    Run metadata for ``classification/evaluation/cost_analysis.py``.

    That module reads ``documents_sent_to_llm`` and ``provider_usage``, while
    ``infrastructure/runtime.py`` and ``infrastructure/outputs.py`` produce
    ``documents_sent_to_review`` and ``strategy_usage``. The two halves of the
    pipeline do not agree on those names yet, so both spellings are written
    here — the cost columns come out as zeros otherwise. Flagged for the team;
    fixing it belongs in the evaluation module, not in this one.
    """

    return {
        "run_id": run_id,
        "pipeline_name": PIPELINE_NAME,
        "saved_files": [str(path) for path in saved_files],
        "strategies": strategies,
        **routing_metrics,
        **runtime_metrics,
        "calibration": calibration,
        "strategy_usage": {
            ROUTED_STRATEGY: {
                **runtime_metrics,
                "requests_attempted": routing_metrics["documents_sent_to_review"],
                "requests_successful": 0,
            }
        },
        # Keyed by provider, which is what get_provider_usage() looks up.
        "provider_usage": {
            PROVIDER: {
                "requests_attempted": routing_metrics["documents_sent_to_review"],
                "requests_successful": routing_metrics["documents_total"],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
                "provider_reported_cost": 0.0,
            }
        },
    }


def run_prediction(
    run_name: str,
    split: str = "test",
    run_id: str | None = None,
    data_file: str | None = None,
    device: str = "",
) -> dict:
    """
    Score a split and write the run directory.

    Returns a dict with the run id, the written files and the routing summary.
    """

    artifacts_dir = run_artifacts_dir(run_name)

    model, config = load_checkpoint(artifacts_dir, device=resolve_device(device))
    torch_device = resolve_device(device)

    calibration_file = artifacts_dir / "calibration.json"

    if not calibration_file.exists():
        raise FileNotFoundError(
            f"Calibration not found: {calibration_file}. Run training first."
        )

    calibration = load_calibration(calibration_file)

    t_low = float(calibration["t_low"])
    t_high = float(calibration["t_high"])

    # Per-label entity thresholds, fitted on validation during training. Older
    # checkpoints predate them and fall back to the config's single value.
    entity_thresholds = {
        label: float(value)
        for label, value in calibration.get("entity_thresholds", {}).items()
    }

    if not entity_thresholds:
        logger.warning(
            "Checkpoint has no calibrated entity thresholds; falling back to a "
            "flat %.2f cut for all 12 labels.",
            config.entity_threshold,
        )
        entity_thresholds = {
            label: config.entity_threshold for label in ENTITY_LABELS
        }

    logger.info(
        "Loaded %s (t_low=%.4f, t_high=%.4f, calibrated on %s)",
        run_name,
        t_low,
        t_high,
        calibration.get("calibrated_on", "validation"),
    )

    # ── Data ────────────────────────────────────────────────
    # The same config drives the split, so `--split test` here is the split the
    # calibration never saw.
    df = load_dataset(data_file or config.data_file)
    df, resolved_mode = resolve_splits(df, config)

    if resolved_mode != config.resolved_split_mode:
        logger.warning(
            "Split mode resolved to '%s' but the checkpoint was trained with "
            "'%s'. Splits may not match the ones used during training.",
            resolved_mode,
            config.resolved_split_mode,
        )

    if split == "all":
        frame = df.reset_index(drop=True)
    else:
        frame = df[df["split"] == split].reset_index(drop=True)

    if frame.empty:
        raise ValueError(f"Split '{split}' is empty; nothing to predict.")

    logger.info("Scoring %s documents from split '%s'", len(frame), split)

    tokenizer = load_tokenizer(config.pretrained_dir)

    dataset = build_dataset(frame, tokenizer, config.max_length)
    loader = build_dataloader(dataset, config.eval_batch_size, shuffle=False)

    # ── Inference ───────────────────────────────────────────
    # Timed over the forward passes only; tokenisation is excluded because in
    # production Sweep 1 has already read the document.
    started = time.perf_counter()
    probs, entity_probs = predict_probabilities(model, loader, torch_device)
    elapsed = time.perf_counter() - started

    inference_ms = 1000.0 * elapsed / max(len(frame), 1)

    logger.info(
        "Inference: %.2fs total, %.2f ms/document (device=%s)",
        elapsed,
        inference_ms,
        torch_device,
    )

    # ── Routing ─────────────────────────────────────────────
    auto_no, routed, auto_yes = zone_masks(probs, t_low, t_high)

    zones = np.where(auto_no, "confident_non_pii", np.where(auto_yes, "confident_pii", "routed_to_llm"))

    routing = {
        "zone": zones,
        "routed": routed,
        "t_low": t_low,
        "t_high": t_high,
    }

    run_id = run_id or make_run_id()
    run_dir = CLASSIFICATION_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []

    routed_output = build_output_frame(
        source=frame,
        probs=probs,
        entity_probs=entity_probs,
        strategy=ROUTED_STRATEGY,
        run_id=run_id,
        model_name=config.model_name,
        inference_ms=inference_ms,
        entity_thresholds=entity_thresholds,
        predicted_pii=auto_yes | routed,
        routing=routing,
    )

    routed_file = run_dir / f"{ROUTED_STRATEGY}.csv"
    routed_output.to_csv(routed_file, index=False)
    saved_files.append(routed_file)

    standalone_output = build_output_frame(
        source=frame,
        probs=probs,
        entity_probs=entity_probs,
        strategy=STANDALONE_STRATEGY,
        run_id=run_id,
        model_name=config.model_name,
        inference_ms=inference_ms,
        entity_thresholds=entity_thresholds,
        predicted_pii=probs >= config.standalone_threshold,
    )

    standalone_file = run_dir / f"{STANDALONE_STRATEGY}.csv"
    standalone_output.to_csv(standalone_file, index=False)
    saved_files.append(standalone_file)

    # ── Metadata ────────────────────────────────────────────
    binary_true, _ = extract_labels(frame)

    routing_summary = evaluate_routing(
        probs=probs,
        y_true=binary_true.astype(bool),
        t_low=t_low,
        t_high=t_high,
    )
    routing_summary["split"] = split
    routing_summary["calibrated_on"] = calibration.get("calibrated_on")

    documents_total = len(frame)
    documents_routed = int(routed.sum())

    routing_metrics = {
        "documents_total": documents_total,
        "documents_sent_to_review": documents_routed,
        # cost_analysis.py reads this spelling.
        "documents_sent_to_llm": documents_routed,
        "documents_resolved_locally": documents_total - documents_routed,
        "llm_calls_avoided": documents_total - documents_routed,
        "routing_rate": round(documents_routed / documents_total, 6),
        "local_processing_rate": round(
            (documents_total - documents_routed) / documents_total, 6
        ),
    }

    runtime_metrics = {
        "bert_requests_attempted": documents_total,
        "bert_requests_successful": documents_total,
        "bert_runtime_seconds": round(elapsed, 4),
        "bert_average_runtime_seconds": round(elapsed / max(documents_total, 1), 6),
        "inference_ms_per_document": round(inference_ms, 4),
    }

    metadata = build_run_metadata(
        run_id=run_id,
        strategies=[ROUTED_STRATEGY, STANDALONE_STRATEGY],
        saved_files=saved_files,
        routing_metrics=routing_metrics,
        runtime_metrics=runtime_metrics,
        calibration=calibration,
    )
    metadata["split"] = split
    metadata["split_mode"] = resolved_mode
    metadata["routing_summary"] = routing_summary
    metadata["prefilter_run_name"] = run_name

    metadata_file = run_dir / "run_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    logger.info("Run directory written: %s", run_dir)

    _print_summary(run_id, split, routing_summary, runtime_metrics, saved_files)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "saved_files": saved_files,
        "routing_summary": routing_summary,
        "runtime_metrics": runtime_metrics,
    }


def _print_summary(
    run_id: str,
    split: str,
    routing_summary: dict,
    runtime_metrics: dict,
    saved_files: list[Path],
) -> None:
    print("\n" + "=" * 64)
    print(f"PRE-FILTER PREDICTION — run_id={run_id}  split={split}")
    print("=" * 64)
    print(f"  documents            : {routing_summary['n']} "
          f"({routing_summary['n_positive']} positive)")
    print(f"  routed to LLM        : {routing_summary['routed_n']} "
          f"({100 * routing_summary['routed_fraction']:.1f}%)")
    print(f"  LLM calls avoided    : {routing_summary['llm_calls_avoided']} "
          f"({100 * routing_summary['llm_call_reduction']:.1f}%)")
    print(f"  pre-filter recall    : {routing_summary['prefilter_recall']}")
    print(f"  missed positives     : {routing_summary['missed_positives']}")
    print(f"  auto-PII precision   : {routing_summary['auto_yes_precision']}")
    print(f"  inference            : "
          f"{runtime_metrics['inference_ms_per_document']:.2f} ms/document")
    print("\n  files:")
    for path in saved_files:
        print(f"    {path}")
    print(f"\n  next: evaluate --run-id {run_id}")
    print("=" * 64 + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write evaluation-compatible pre-filter predictions."
    )
    parser.add_argument("--run-name", default="distilbert_prefilter")
    parser.add_argument(
        "--split",
        default="test",
        help="train / validation / test / all. Defaults to test.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Classification run id. Defaults to a fresh YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--device", default="")

    args = parser.parse_args(argv)

    run_prediction(
        run_name=args.run_name,
        split=args.split,
        run_id=args.run_id,
        data_file=args.data_file,
        device=args.device,
    )


if __name__ == "__main__":
    main()
