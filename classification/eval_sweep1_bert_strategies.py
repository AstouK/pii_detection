"""
Isolated evaluation runner for two Sweep 1 + DistilBERT strategies.

This file is intentionally standalone. It does not plug into
``strategy_runner.py`` and can be removed once the hybrid runners are
implemented in the main pipeline.

Strategy A — ``sweep1_ambiguous_bert``
    Take the documents Sweep 1 marks as ambiguous (``needs_llm_review``)
    and score them with DistilBERT only. Answers: how well does the model
    resolve the hard cases?

Strategy B — ``rule_plus_distilbert_plus_qwen``
    Full hybrid on the whole dataset:
        Sweep 1 short-circuits clear cases
        DistilBERT routes ambiguous documents
        the LLM reviews the uncertain band

Outputs are written to ``classification/results/runs/<run_id>/`` so the
existing ``evaluate`` command can score them without extra wiring.

Examples
--------
Run both strategies and evaluate the saved run:

    python -m classification.eval_sweep1_bert_strategies
    evaluate --run-id <run_id>

Run only the ambiguous DistilBERT slice (no LLM calls):

    python -m classification.eval_sweep1_bert_strategies --skip-hybrid
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from config.logging_config import setup_logging

from classification.config import (
    DEFAULT_INPUT_FILE,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_DATASET_VERSION,
    get_model_config,
)
from classification.evaluation.evaluate_pipeline import run_evaluation
from classification.evaluation.metrics import compute_all_metrics
from classification.evaluation.reporting import print_metric_report
from classification.infrastructure.io import load_input_data
from classification.infrastructure.metadata import (
    add_sweep1_metadata,
    resolve_dataset_version,
)
from classification.infrastructure.outputs import (
    build_output_file,
    create_run_dir,
    save_run_metadata,
    save_sweep1_results,
)
from classification.infrastructure.runtime import (
    compute_routing_metrics,
    compute_strategy_usage_summary,
)
from classification.prefilter.config import (
    BINARY_LABEL_COL,
    CONTEXT_COLS,
    DOCUMENT_ID_COL,
    ENTITY_LABELS,
    MODEL_FAMILY,
    PIPELINE_NAME,
    PREDICTION_SOURCE,
    PREDICTION_STAGE,
    PROVIDER,
    entity_label_columns,
    run_artifacts_dir,
)
from classification.prefilter.data import (
    build_dataloader,
    build_dataset,
    to_bool_series,
)
from classification.prefilter.model import load_checkpoint, load_tokenizer
from classification.prefilter.predict import (
    build_entity_predictions,
    build_per_type_conf,
)
from classification.prefilter.thresholds import load_calibration, zone_masks
from classification.prefilter.train import predict_probabilities, resolve_device
from classification.review.llm_reviewer import run_llm
from classification.schemas.experiment_schema import ExperimentMetadata
from classification.sweep1 import run_sweep1

setup_logging()
logger = logging.getLogger(__name__)

STRATEGY_AMBIGUOUS_BERT = "sweep1_ambiguous_bert"
STRATEGY_HYBRID = "rule_plus_distilbert_plus_qwen"

DEFAULT_PREFILTER_RUN_NAME = "distilbert_prefilter_3500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated Sweep 1 + DistilBERT evaluation strategies and "
            "save evaluation-compatible outputs."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Dataset CSV to classify.",
    )
    parser.add_argument(
        "--prefilter-run-name",
        default=DEFAULT_PREFILTER_RUN_NAME,
        help="Checkpoint directory name under classification/prefilter/artifacts/.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "validation", "test"],
        help="Optional dataset split filter. Defaults to all rows.",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen3_7_plus",
        help="LLM model id from MODEL_REGISTRY for the hybrid strategy.",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt template version for the hybrid LLM stage.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Torch device for DistilBERT inference. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--skip-hybrid",
        action="store_true",
        help="Skip the Sweep 1 + DistilBERT + LLM strategy.",
    )
    parser.add_argument(
        "--skip-ambiguous-bert",
        action="store_true",
        help="Skip the ambiguous-only DistilBERT strategy.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the evaluation pipeline immediately after saving outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick smoke tests.",
    )

    return parser.parse_args()


def filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all":
        return df.reset_index(drop=True)

    split_col = "recommended_split"
    if split_col not in df.columns:
        raise ValueError(
            f"Split '{split}' requested but column '{split_col}' is missing."
        )

    frame = df[df[split_col] == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Split '{split}' is empty.")

    return frame


def load_prefilter_bundle(
    prefilter_run_name: str,
    device: str,
) -> tuple:
    artifacts_dir = run_artifacts_dir(prefilter_run_name)
    calibration_file = artifacts_dir / "calibration.json"

    if not calibration_file.exists():
        raise FileNotFoundError(
            f"Calibration not found: {calibration_file}. "
            "Train the pre-filter first or pass --prefilter-run-name."
        )

    model, config = load_checkpoint(artifacts_dir, device=resolve_device(device))
    calibration = load_calibration(calibration_file)
    tokenizer = load_tokenizer(config.pretrained_dir)

    entity_thresholds = {
        label: float(value)
        for label, value in calibration.get("entity_thresholds", {}).items()
    }
    if not entity_thresholds:
        entity_thresholds = {
            label: config.entity_threshold for label in ENTITY_LABELS
        }

    return {
        "model": model,
        "config": config,
        "calibration": calibration,
        "tokenizer": tokenizer,
        "entity_thresholds": entity_thresholds,
        "t_low": float(calibration["t_low"]),
        "t_high": float(calibration["t_high"]),
        "device": resolve_device(device),
    }


def initialize_bert_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    defaults = {
        "pii_probability": np.nan,
        "routing_zone": "",
        "routed_to_llm": False,
        "needs_bert_review": False,
        "bert_request_success": False,
        "bert_runtime_seconds": 0.0,
        "inference_ms": 0.0,
        "t_low": np.nan,
        "t_high": np.nan,
        "per_type_conf": "{}",
    }

    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default

    for label in ENTITY_LABELS:
        column = f"{label}_predicted"
        if column not in result.columns:
            result[column] = "no"

    return result


def score_ambiguous_with_bert(
    df: pd.DataFrame,
    bundle: dict,
) -> tuple[pd.DataFrame, dict]:
    result = initialize_bert_columns(df)

    ambiguous_mask = (
        result["needs_llm_review"]
        .fillna(False)
        .astype(bool)
    )

    ambiguous_count = int(ambiguous_mask.sum())
    runtime_info = {
        "bert_runtime_seconds": 0.0,
        "bert_requests_attempted": ambiguous_count,
        "bert_requests_successful": 0,
        "inference_ms_per_document": 0.0,
    }

    if ambiguous_count == 0:
        logger.info("No ambiguous Sweep 1 documents; skipping DistilBERT inference.")
        return result, runtime_info

    ambiguous_df = result.loc[ambiguous_mask].reset_index(drop=True)
    config = bundle["config"]

    dataset = build_dataset(
        ambiguous_df,
        bundle["tokenizer"],
        config.max_length,
    )
    loader = build_dataloader(
        dataset,
        config.eval_batch_size,
        shuffle=False,
    )

    started = time.perf_counter()
    probs, entity_probs = predict_probabilities(
        bundle["model"],
        loader,
        bundle["device"],
    )
    elapsed = time.perf_counter() - started

    inference_ms = 1000.0 * elapsed / max(ambiguous_count, 1)
    auto_no, routed, auto_yes = zone_masks(
        probs,
        bundle["t_low"],
        bundle["t_high"],
    )
    zones = np.where(
        auto_no,
        "confident_non_pii",
        np.where(auto_yes, "confident_pii", "routed_to_llm"),
    )

    per_type_conf = build_per_type_conf(
        entity_probs,
        bundle["entity_thresholds"],
    )
    entity_predictions = build_entity_predictions(
        entity_probs,
        bundle["entity_thresholds"],
    )

    original_indices = result.index[ambiguous_mask]

    result.loc[original_indices, "pii_probability"] = np.round(probs, 6)
    result.loc[original_indices, "routing_zone"] = zones
    result.loc[original_indices, "routed_to_llm"] = routed
    result.loc[original_indices, "needs_bert_review"] = True
    result.loc[original_indices, "bert_request_success"] = True
    result.loc[original_indices, "bert_runtime_seconds"] = round(
        inference_ms / 1000.0,
        6,
    )
    result.loc[original_indices, "inference_ms"] = round(inference_ms, 4)
    result.loc[original_indices, "t_low"] = bundle["t_low"]
    result.loc[original_indices, "t_high"] = bundle["t_high"]
    result.loc[original_indices, "per_type_conf"] = per_type_conf

    for column, values in entity_predictions.items():
        result.loc[original_indices, column] = values

    runtime_info.update(
        {
            "bert_runtime_seconds": round(elapsed, 4),
            "bert_requests_successful": ambiguous_count,
            "inference_ms_per_document": round(inference_ms, 4),
        }
    )

    logger.info(
        "DistilBERT scored %s ambiguous documents in %.2fs (%.2f ms/doc)",
        ambiguous_count,
        elapsed,
        inference_ms,
    )
    logger.info(
        "Routing on ambiguous subset: confident_non_pii=%s, routed=%s, confident_pii=%s",
        int(auto_no.sum()),
        int(routed.sum()),
        int(auto_yes.sum()),
    )

    return result, runtime_info


def add_strategy_metadata(
    df: pd.DataFrame,
    *,
    strategy: str,
    run_id: str,
    dataset_version: str,
    prompt_version: str,
    provider: str,
    model_family: str,
    model_name: str,
    prediction_source: str,
) -> pd.DataFrame:
    result = df.copy()

    metadata = ExperimentMetadata(
        run_id=run_id,
        strategy=strategy,
        provider=provider,
        model_family=model_family,
        model_name=model_name,
        prompt_version=prompt_version,
        dataset_version=dataset_version,
    )

    for column, value in metadata.__dict__.items():
        result[column] = value

    result["prediction_source"] = prediction_source
    result["prediction_stage"] = PREDICTION_STAGE
    result["pipeline_name"] = PIPELINE_NAME

    return result


def build_ambiguous_bert_output(
    df: pd.DataFrame,
    *,
    run_id: str,
    dataset_version: str,
    model_name: str,
    standalone_threshold: float,
) -> pd.DataFrame:
    ambiguous = df[
        df["needs_llm_review"]
        .fillna(False)
        .astype(bool)
    ].copy()

    if ambiguous.empty:
        return pd.DataFrame()

    ambiguous["predicted_pii"] = (
        ambiguous["pii_probability"].fillna(0.0) >= standalone_threshold
    )

    output = ambiguous.copy()
    output = add_strategy_metadata(
        output,
        strategy=STRATEGY_AMBIGUOUS_BERT,
        run_id=run_id,
        dataset_version=dataset_version,
        prompt_version="not_applicable",
        provider=PROVIDER,
        model_family=MODEL_FAMILY,
        model_name=model_name,
        prediction_source=PREDICTION_SOURCE,
    )

    return output


def compute_hybrid_final_prediction(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    detected_pii = result["detected_pii"].fillna(False).astype(bool)
    needs_llm_review = result["needs_llm_review"].fillna(False).astype(bool)
    llm_pii = result.get("llm_pii", False)
    llm_pii = pd.Series(llm_pii, index=result.index).fillna(False).astype(bool)

    routing_zone = result.get("routing_zone", "")
    routing_zone = pd.Series(routing_zone, index=result.index).fillna("")

    bert_confident_pii = needs_llm_review & (routing_zone == "confident_pii")
    bert_routed = needs_llm_review & result["routed_to_llm"].fillna(False).astype(bool)

    result["final_pii"] = detected_pii | bert_confident_pii | (bert_routed & llm_pii)
    result["predicted_pii"] = result["final_pii"]

    return result


def build_hybrid_output(
    df: pd.DataFrame,
    *,
    run_id: str,
    dataset_version: str,
    prompt_version: str,
    bert_model_name: str,
    llm_model_id: str,
) -> pd.DataFrame:
    llm_config = get_model_config(llm_model_id)

    output = compute_hybrid_final_prediction(df)
    output = add_strategy_metadata(
        output,
        strategy=STRATEGY_HYBRID,
        run_id=run_id,
        dataset_version=dataset_version,
        prompt_version=prompt_version,
        provider=llm_config["provider"],
        model_family=f"{MODEL_FAMILY}+{llm_config['model_family']}",
        model_name=f"{bert_model_name}+{llm_config['model_name']}",
        prediction_source="hybrid",
    )

    return output


def save_strategy_output(
    df: pd.DataFrame,
    *,
    strategy: str,
    run_dir: Path,
) -> Path:
    output_file = build_output_file(strategy=strategy, run_dir=run_dir)
    df.to_csv(output_file, index=False)
    logger.info("Saved strategy output: %s", output_file)
    return output_file


def print_quick_metrics(
    df: pd.DataFrame,
    *,
    title: str,
    prediction_col: str = "predicted_pii",
) -> None:
    if BINARY_LABEL_COL not in df.columns:
        logger.warning("Ground-truth column missing; skipping quick metrics.")
        return

    metrics = compute_all_metrics(df)
    print_metric_report(metrics, title=title)

    ambiguous = df[
        df["needs_llm_review"].fillna(False).astype(bool)
    ] if "needs_llm_review" in df.columns else df

    if prediction_col in ambiguous.columns and not ambiguous.empty:
        truth = to_bool_series(ambiguous[BINARY_LABEL_COL])
        predicted = to_bool_series(ambiguous[prediction_col])
        tp = int((truth & predicted).sum())
        fn = int((truth & ~predicted).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        print(
            f"\n  Ambiguous-subset recall ({prediction_col}): "
            f"{recall:.4f}  (TP={tp}, FN={fn}, n={len(ambiguous)})"
        )


def run_strategies(args: argparse.Namespace) -> dict:
    pipeline_start = perf_counter()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = create_run_dir(run_id)

    df = load_input_data(input_file=args.input_file)
    if args.limit:
        df = df.head(args.limit)

    df = filter_split(df, args.split)
    dataset_version = resolve_dataset_version(
        df=df,
        default_version=DEFAULT_DATASET_VERSION,
    )

    logger.info("Run ID: %s", run_id)
    logger.info("Dataset: %s (%s rows, split=%s)", args.input_file, len(df), args.split)

    sweep1_start = perf_counter()
    df_sweep1 = run_sweep1(df)
    sweep1_runtime_seconds = round(perf_counter() - sweep1_start, 4)

    saved_files = [
        save_sweep1_results(
            df_sweep1=df_sweep1,
            run_dir=run_dir,
            run_id=run_id,
        )
    ]

    bundle = load_prefilter_bundle(
        prefilter_run_name=args.prefilter_run_name,
        device=args.device,
    )

    df_scored, bert_runtime = score_ambiguous_with_bert(df_sweep1, bundle)

    strategies: list[str] = []
    strategy_runtime_seconds: dict[str, float] = {}
    strategy_usage: dict[str, dict] = {}

    if not args.skip_ambiguous_bert:
        strategy_start = perf_counter()
        ambiguous_output = build_ambiguous_bert_output(
            df_scored,
            run_id=run_id,
            dataset_version=dataset_version,
            model_name=bundle["config"].model_name,
            standalone_threshold=bundle["config"].standalone_threshold,
        )

        if ambiguous_output.empty:
            logger.warning(
                "Skipping '%s' because Sweep 1 produced no ambiguous documents.",
                STRATEGY_AMBIGUOUS_BERT,
            )
        else:
            saved_files.append(
                save_strategy_output(
                    ambiguous_output,
                    strategy=STRATEGY_AMBIGUOUS_BERT,
                    run_dir=run_dir,
                )
            )
            strategies.append(STRATEGY_AMBIGUOUS_BERT)
            strategy_runtime_seconds[STRATEGY_AMBIGUOUS_BERT] = round(
                perf_counter() - strategy_start,
                4,
            )
            strategy_usage[STRATEGY_AMBIGUOUS_BERT] = compute_strategy_usage_summary(
                ambiguous_output
            )
            print_quick_metrics(
                ambiguous_output,
                title=f"Quick metrics: {STRATEGY_AMBIGUOUS_BERT}",
            )

    if not args.skip_hybrid:
        strategy_start = perf_counter()
        df_hybrid = df_scored.copy(deep=True)

        df_hybrid["needs_llm_review"] = (
            df_hybrid["needs_llm_review"].fillna(False).astype(bool)
            & df_hybrid["routed_to_llm"].fillna(False).astype(bool)
        )

        df_hybrid = run_llm(
            df=df_hybrid,
            model_id=args.llm_model,
            prompt_version=args.prompt_version,
        )
        hybrid_output = build_hybrid_output(
            df_hybrid,
            run_id=run_id,
            dataset_version=dataset_version,
            prompt_version=args.prompt_version,
            bert_model_name=bundle["config"].model_name,
            llm_model_id=args.llm_model,
        )
        saved_files.append(
            save_strategy_output(
                hybrid_output,
                strategy=STRATEGY_HYBRID,
                run_dir=run_dir,
            )
        )
        strategies.append(STRATEGY_HYBRID)
        strategy_runtime_seconds[STRATEGY_HYBRID] = round(
            perf_counter() - strategy_start,
            4,
        )
        strategy_usage[STRATEGY_HYBRID] = compute_strategy_usage_summary(
            hybrid_output
        )
        print_quick_metrics(
            hybrid_output,
            title=f"Quick metrics: {STRATEGY_HYBRID}",
        )

    routing_metrics = compute_routing_metrics(df_sweep1)
    routing_metrics["documents_sent_to_llm"] = int(
        df_scored["routed_to_llm"].fillna(False).astype(bool).sum()
    )

    runtime_metrics = {
        "sweep1_runtime_seconds": sweep1_runtime_seconds,
        "sweep2_runtime_seconds": round(
            sum(strategy_runtime_seconds.values()),
            4,
        ),
        "pipeline_runtime_seconds": round(
            perf_counter() - pipeline_start,
            4,
        ),
        "strategy_runtime_seconds": strategy_runtime_seconds,
        **bert_runtime,
    }

    metadata_file = save_run_metadata(
        run_dir=run_dir,
        run_id=run_id,
        strategies=strategies,
        saved_files=saved_files,
        routing_metrics=routing_metrics,
        runtime_metrics=runtime_metrics,
        strategy_usage=strategy_usage,
        dataset_version=dataset_version,
        prompt_version=args.prompt_version,
    )
    saved_files.append(metadata_file)

    summary = {
        "run_id": run_id,
        "run_dir": run_dir,
        "saved_files": saved_files,
        "strategies": strategies,
        "prefilter_run_name": args.prefilter_run_name,
        "calibration": {
            "t_low": bundle["t_low"],
            "t_high": bundle["t_high"],
        },
    }

    print("\n" + "=" * 72)
    print(f"ISOLATED SWEEP1 + BERT EVAL RUN — run_id={run_id}")
    print("=" * 72)
    print(f"  split                     : {args.split}")
    print(f"  documents                 : {len(df_sweep1)}")
    print(
        "  sweep1 ambiguous docs     : "
        f"{int(df_sweep1['needs_llm_review'].fillna(False).astype(bool).sum())}"
    )
    print(
        "  bert routed to LLM        : "
        f"{int(df_scored['routed_to_llm'].fillna(False).astype(bool).sum())}"
    )
    print(f"  prefilter checkpoint      : {args.prefilter_run_name}")
    print(f"  strategies written        : {', '.join(strategies) or '(none)'}")
    print("\n  files:")
    for path in saved_files:
        print(f"    {path}")
    print(f"\n  next: evaluate --run-id {run_id}")
    print("=" * 72 + "\n")

    return summary


def main() -> None:
    args = parse_args()
    summary = run_strategies(args)

    if args.evaluate:
        run_evaluation(run_id=summary["run_id"])


if __name__ == "__main__":
    main()
