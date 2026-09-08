"""
Fine-tune the dual-head pre-filter and calibrate its router.

Run:

    python -m classification.prefilter.train
    python -m classification.prefilter.train --epochs 10 --seed 7
    python -m classification.prefilter.train --split-mode recommended

Every run writes a self-describing directory under
``classification/prefilter/artifacts/<run_name>/``:

    config.json              the exact PreFilterConfig used
    prefilter_model.pt       weights + encoder config + label vocabulary
    training_history.csv     per-epoch losses and validation metrics
    calibration.json         the selected (t_low, t_high) and its metrics
    routing_frontier.csv     LLM-call share vs. recall, the full curve
    validation_scores.csv    per-document probabilities on validation
    metrics_summary.json     everything reported in the brief's section 6

Model selection happens on the validation split against ``routing_cost``: the
checkpoint kept is the one that routes the fewest documents to the LLM while
still clearing the recall target. That is the quantity the work package exists
to minimise, so selecting on it directly beats selecting on F1 and hoping.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from config.logging_config import setup_logging

from classification.prefilter.config import (
    BINARY_LABEL_COL,
    ENTITY_LABELS,
    PreFilterConfig,
    REPORTS_DIR,
    VALIDATION_SPLIT,
    run_artifacts_dir,
)
from classification.prefilter.data import (
    build_dataloader,
    build_dataset,
    compute_entity_pos_weights,
    compute_pos_weight,
    describe_split,
    entity_support,
    extract_labels,
    load_dataset,
    resolve_splits,
    split_frames,
    to_bool_series,
)
from classification.prefilter.model import (
    PiiPreFilterModel,
    build_optimizer_groups,
    count_parameters,
    describe_model,
    load_tokenizer,
)
from classification.prefilter.thresholds import (
    average_precision,
    binary_metrics,
    calibrate_entity_thresholds,
    calibrate_thresholds,
    entity_metrics_at_thresholds,
    plot_probability_distribution,
    plot_routing_frontier,
    routing_frontier,
    save_calibration,
    single_threshold_sweep,
)

setup_logging()
logger = logging.getLogger(__name__)

#: Recall of the rule-based Sweep 1 baseline the router must not fall below.
BASELINE_RECALL = 0.9833


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Pin every RNG the training loop touches.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def resolve_device(requested: str = "") -> torch.device:
    """
    Pick a device, honouring an explicit request.
    """

    if requested:
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_probabilities(
    model: PiiPreFilterModel,
    dataloader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over a dataloader.

    Returns ``(binary_probs, entity_probs)`` with shapes ``(n,)`` and
    ``(n, 12)``.
    """

    model.eval()

    binary_batches = []
    entity_batches = []

    for batch in dataloader:
        binary_logits, entity_logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )

        binary_batches.append(torch.sigmoid(binary_logits).cpu().numpy())
        entity_batches.append(torch.sigmoid(entity_logits).cpu().numpy())

    return (
        np.concatenate(binary_batches) if binary_batches else np.array([]),
        (
            np.concatenate(entity_batches)
            if entity_batches
            else np.zeros((0, len(ENTITY_LABELS)))
        ),
    )


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(config: PreFilterConfig) -> dict:
    """
    Train, calibrate and persist one pre-filter run.

    Returns the metrics summary dict that is also written to disk.
    """

    set_seed(config.seed)

    device = resolve_device(config.device)
    config.device = str(device)

    logger.info("Device: %s", device)
    logger.info("Seed: %s", config.seed)

    # ── Data ────────────────────────────────────────────────
    df = load_dataset(config.data_file)
    logger.info("Loaded %s documents from %s", len(df), config.data_file)

    df, resolved_mode = resolve_splits(df, config)
    config.resolved_split_mode = resolved_mode

    logger.info("Split mode resolved to: %s", resolved_mode)
    logger.info("Split summary:\n%s", describe_split(df, "split").to_string(index=False))

    frames = split_frames(df)

    for name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"Split '{name}' is empty; cannot train.")

    # Fail before training rather than after. The router is calibrated against
    # a recall constraint, which is undefined on a split with no positives, so
    # `--split-mode recommended` on the current pilot is guaranteed to die --
    # and dying two minutes later, after a wasted epoch, hides what went wrong
    # behind a stack trace.
    validation_positives = int(
        to_bool_series(frames[VALIDATION_SPLIT][BINARY_LABEL_COL]).sum()
    )

    if validation_positives == 0:
        raise ValueError(
            f"The '{VALIDATION_SPLIT}' split contains no positive documents, so "
            "the router cannot be calibrated against a recall target.\n"
            f"  split mode : {resolved_mode}\n"
            f"  dataset    : {config.data_file}\n"
            "Use --split-mode stratified (or auto) to build a label-stratified "
            "split, or fix the dataset's split column. "
            "See classification/prefilter/README.md -> Findings for the team."
        )

    tokenizer = load_tokenizer(config.pretrained_dir)

    datasets = {
        name: build_dataset(frame, tokenizer, config.max_length)
        for name, frame in frames.items()
    }

    train_loader = build_dataloader(
        datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )

    eval_loaders = {
        name: build_dataloader(
            datasets[name],
            batch_size=config.eval_batch_size,
            shuffle=False,
        )
        for name in frames
    }

    train_binary, train_entity = extract_labels(frames["train"])

    pos_weight = compute_pos_weight(train_binary, config.max_pos_weight)
    entity_pos_weights = compute_entity_pos_weights(
        train_entity, config.max_entity_pos_weight
    )

    logger.info(
        "Class imbalance: %.1f%% positive in train, binary pos_weight=%.3f",
        100 * train_binary.mean(),
        pos_weight,
    )

    # ── Model ───────────────────────────────────────────────
    model = PiiPreFilterModel(
        pretrained_dir=config.pretrained_dir,
        num_entity_labels=config.num_entity_labels,
        classifier_dropout=config.classifier_dropout,
    ).to(device)

    logger.info("Model: %s", describe_model(model))

    binary_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    entity_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(entity_pos_weights, device=device)
    )

    optimizer = torch.optim.AdamW(
        build_optimizer_groups(
            model=model,
            learning_rate=config.learning_rate,
            head_learning_rate=config.head_learning_rate,
            weight_decay=config.weight_decay,
        )
    )

    total_steps = max(len(train_loader) * config.epochs, 1)
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(step: int) -> float:
        """Linear warmup, then linear decay to zero."""
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        remaining = total_steps - warmup_steps
        return max(0.0, (total_steps - step) / max(remaining, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    validation_binary, validation_entity = extract_labels(frames[VALIDATION_SPLIT])

    output_dir = run_artifacts_dir(config.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_state: dict | None = None
    best_score: float | None = None
    best_epoch = -1

    training_started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()

        running = {"total": 0.0, "binary": 0.0, "entity": 0.0}

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            binary_logits, entity_logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            binary_loss = binary_loss_fn(
                binary_logits, batch["binary_label"].to(device)
            )
            entity_loss = entity_loss_fn(
                entity_logits, batch["entity_labels"].to(device)
            )

            loss = binary_loss + config.multilabel_loss_weight * entity_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running["total"] += float(loss.item())
            running["binary"] += float(binary_loss.item())
            running["entity"] += float(entity_loss.item())

        n_batches = max(len(train_loader), 1)

        validation_probs, validation_entity_probs = predict_probabilities(
            model, eval_loaders[VALIDATION_SPLIT], device
        )

        plain = binary_metrics(
            validation_binary.astype(bool),
            validation_probs >= config.standalone_threshold,
        )
        pr_auc = average_precision(validation_binary.astype(bool), validation_probs)

        calibration = calibrate_thresholds(
            probs=validation_probs,
            y_true=validation_binary.astype(bool),
            recall_target=config.recall_target,
            precision_target=config.precision_target,
            grid_steps=config.threshold_grid_steps,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(running["total"] / n_batches, 6),
            "train_binary_loss": round(running["binary"] / n_batches, 6),
            "train_entity_loss": round(running["entity"] / n_batches, 6),
            "val_accuracy": plain["accuracy"],
            "val_precision": plain["precision"],
            "val_recall": plain["recall"],
            "val_f1": plain["f1"],
            "val_pr_auc": round(pr_auc, 6),
            "val_routing_feasible": calibration["feasible"],
            "val_routed_fraction": calibration["routed_fraction"],
            "val_prefilter_recall": calibration["prefilter_recall"],
            "epoch_seconds": round(time.perf_counter() - epoch_started, 2),
        }

        history.append(epoch_record)

        logger.info(
            "epoch %s/%s loss=%.4f val_f1=%.4f val_pr_auc=%.4f "
            "routed=%.1f%% feasible=%s",
            epoch,
            config.epochs,
            epoch_record["train_loss"],
            plain["f1"],
            pr_auc,
            100 * calibration["routed_fraction"],
            calibration["feasible"],
        )

        # ── Model selection ─────────────────────────────────
        # Lower is better, compared as a tuple. A feasible router always beats
        # an infeasible one, so infeasible epochs are ranked in a strictly worse
        # band and ordered among themselves by the same tie-breaks.
        #
        # The tie-breaks are not decoration. On a dataset the model separates
        # cleanly, `routed_fraction` hits 0.0 in the first epoch and stays
        # there, so ranking on it alone is a three-way tie that the first epoch
        # wins by arriving first — and the first epoch is the worst model in the
        # run. PR-AUC and F1 break that tie towards the checkpoint that is
        # actually better at the underlying classification.
        if config.model_selection_metric == "routing_cost" and calibration["feasible"]:
            score = (calibration["routed_fraction"], -pr_auc, -plain["f1"])
        else:
            score = (10.0, -pr_auc, -plain["f1"])

        if best_score is None or score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    training_seconds = time.perf_counter() - training_started

    logger.info(
        "Training finished in %.1fs; best epoch: %s", training_seconds, best_epoch
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.to(device)

    # ── Final calibration on validation ─────────────────────
    validation_probs, validation_entity_probs = predict_probabilities(
        model, eval_loaders[VALIDATION_SPLIT], device
    )

    calibration = calibrate_thresholds(
        probs=validation_probs,
        y_true=validation_binary.astype(bool),
        recall_target=config.recall_target,
        precision_target=config.precision_target,
        grid_steps=config.threshold_grid_steps,
    )

    # One decision threshold per entity label, fitted on the same validation
    # split. `predict` reads these back out of calibration.json.
    entity_thresholds = calibrate_entity_thresholds(
        entity_probs=validation_entity_probs,
        entity_true=validation_entity,
        labels=ENTITY_LABELS,
        default_threshold=config.entity_threshold,
        grid_steps=config.threshold_grid_steps,
    )
    calibration["entity_thresholds"] = entity_thresholds

    calibration["selected_epoch"] = best_epoch
    calibration["calibrated_on"] = VALIDATION_SPLIT
    calibration["split_mode"] = resolved_mode
    calibration["baseline_recall"] = BASELINE_RECALL

    if not calibration["feasible"]:
        logger.warning(
            "Routing constraints could not be met on validation. The saved "
            "operating point routes everything to the LLM (t_low=0, t_high=1), "
            "which is safe but saves nothing."
        )

    frontier = routing_frontier(
        probs=validation_probs,
        y_true=validation_binary.astype(bool),
        precision_target=config.precision_target,
        grid_steps=config.threshold_grid_steps,
    )

    # ── Persist ─────────────────────────────────────────────
    model.save(output_dir, config)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    save_calibration(calibration, output_dir / "calibration.json")
    frontier.to_csv(output_dir / "routing_frontier.csv", index=False)

    single_threshold_sweep(
        validation_probs, validation_binary.astype(bool), config.threshold_grid_steps
    ).to_csv(output_dir / "single_threshold_sweep.csv", index=False)

    pd.DataFrame(
        {
            "document_id": frames[VALIDATION_SPLIT]["document_id"],
            "pii_probability": validation_probs,
            "contains_personal_data": frames[VALIDATION_SPLIT][
                "contains_personal_data"
            ],
        }
    ).to_csv(output_dir / "validation_scores.csv", index=False)

    published_dir = REPORTS_DIR / config.run_name
    published_dir.mkdir(parents=True, exist_ok=True)

    for target_dir in (output_dir, published_dir):
        plot_routing_frontier(
            frontier=frontier,
            output_file=target_dir / "routing_frontier.png",
            operating_point=calibration,
            baseline_recall=BASELINE_RECALL,
        )
        plot_probability_distribution(
            probs=validation_probs,
            y_true=validation_binary.astype(bool),
            t_low=calibration["t_low"],
            t_high=calibration["t_high"],
            output_file=target_dir / "score_distribution.png",
        )

    entity_metrics = entity_metrics_at_thresholds(
        entity_probs=validation_entity_probs,
        entity_true=validation_entity,
        labels=ENTITY_LABELS,
        thresholds=entity_thresholds,
    )
    entity_metrics.to_csv(output_dir / "validation_entity_metrics.csv", index=False)

    logger.info(
        "Entity head (validation, calibrated thresholds):\n%s",
        entity_metrics[
            ["entity", "threshold", "support", "pr_auc", "precision", "recall", "f1"]
        ].to_string(index=False),
    )

    summary = {
        "run_name": config.run_name,
        "seed": config.seed,
        "device": str(device),
        "split_mode": resolved_mode,
        "selected_epoch": best_epoch,
        "training_seconds": round(training_seconds, 2),
        "inference_seconds_per_document": None,  # filled in by predict
        **count_parameters(model),
        "split_summary": describe_split(df, "split").to_dict(orient="records"),
        "entity_support_train": entity_support(frames["train"]).to_dict(
            orient="records"
        ),
        "validation_binary_at_0_5": binary_metrics(
            validation_binary.astype(bool),
            validation_probs >= config.standalone_threshold,
        ),
        "validation_pr_auc": round(
            average_precision(validation_binary.astype(bool), validation_probs), 6
        ),
        "validation_entity_metrics": entity_metrics.to_dict(orient="records"),
        "calibration": calibration,
        "baseline_recall": BASELINE_RECALL,
        "config": config.to_dict(),
    }

    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    _publish_reports(output_dir, config.run_name)

    logger.info("Run artifacts written to %s", output_dir)
    _log_headline(calibration, summary)

    return summary


#: Small artifacts copied from the git-ignored run directory into the committed
#: `reports/` folder, so the team can read a run's outcome from the repository
#: without re-running anything. The checkpoint is deliberately not among them.
PUBLISHED_REPORTS = [
    "calibration.json",
    "routing_frontier.csv",
    "training_history.csv",
    "validation_entity_metrics.csv",
    "validation_scores.csv",
    "metrics_summary.json",
]


def _publish_reports(output_dir: Path, run_name: str) -> None:
    """
    Copy the readable artifacts of a run into ``reports/<run_name>/``.

    Per run, not a flat folder. These filenames are fixed, so publishing every
    run into one directory means the second run silently overwrites the first
    one's results — which is exactly what makes comparing two runs impossible,
    and it is the comparison that the whole work package is judged on.
    """

    published_dir = REPORTS_DIR / run_name
    published_dir.mkdir(parents=True, exist_ok=True)

    for file_name in PUBLISHED_REPORTS:
        source = output_dir / file_name

        if source.exists():
            shutil.copyfile(source, published_dir / file_name)


def _log_headline(calibration: dict, summary: dict) -> None:
    """
    Print the numbers the brief asks to report, so a run is readable from the
    terminal without opening any file.
    """

    plain = summary["validation_binary_at_0_5"]

    print("\n" + "=" * 64)
    print("PRE-FILTER RESULT (validation split)")
    print("=" * 64)
    print(f"  split mode          : {summary['split_mode']}")
    print(f"  selected epoch      : {summary['selected_epoch']}")
    print(f"  training time       : {summary['training_seconds']:.1f}s")
    print(f"  model size          : {summary['model_size_mb']:.1f} MB "
          f"({summary['parameters_total']:,} params)")
    print("\n  Binary @ 0.5")
    print(f"    accuracy  : {plain['accuracy']:.4f}")
    print(f"    precision : {plain['precision']:.4f}")
    print(f"    recall    : {plain['recall']:.4f}")
    print(f"    f1        : {plain['f1']:.4f}")
    print(f"    TP={plain['TP']} TN={plain['TN']} FP={plain['FP']} FN={plain['FN']}")
    print(f"    PR-AUC    : {summary['validation_pr_auc']:.4f}")
    print("\n  Three-zone router")
    print(f"    feasible            : {calibration['feasible']}")
    print(f"    t_low / t_high      : {calibration['t_low']:.4f} / "
          f"{calibration['t_high']:.4f}")
    print(f"    routed to LLM       : {100 * calibration['routed_fraction']:.1f}%"
          f"  ({calibration['routed_n']}/{calibration['n']} documents)")
    print(f"    LLM calls avoided   : {calibration['llm_calls_avoided']}"
          f"  ({100 * calibration['llm_call_reduction']:.1f}%)")
    print(f"    pre-filter recall   : {calibration['prefilter_recall']}"
          f"   (baseline {summary['baseline_recall']})")
    print(f"    missed positives    : {calibration['missed_positives']}")
    print(f"    auto-PII precision  : {calibration['auto_yes_precision']}")
    print("=" * 64 + "\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = PreFilterConfig()

    parser = argparse.ArgumentParser(
        description="Train the transformer pre-filter for GDPR PII detection."
    )

    parser.add_argument("--data-file", default=defaults.data_file)
    parser.add_argument("--pretrained-dir", default=defaults.pretrained_dir)
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument(
        "--split-mode",
        default=defaults.split_mode,
        choices=["auto", "recommended", "stratified"],
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=defaults.validation_fraction,
        help="Only used when split-mode is 'stratified'.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=defaults.test_fraction,
        help="Only used when split-mode is 'stratified'.",
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--recall-target", type=float, default=defaults.recall_target
    )
    parser.add_argument(
        "--precision-target", type=float, default=defaults.precision_target
    )
    parser.add_argument(
        "--multilabel-loss-weight",
        type=float,
        default=defaults.multilabel_loss_weight,
    )
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log this training run to MLflow.",
    )

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PreFilterConfig:
    """
    Build a :class:`PreFilterConfig` from parsed CLI arguments.
    """

    return PreFilterConfig(
        data_file=args.data_file,
        pretrained_dir=args.pretrained_dir,
        model_name=args.model_name,
        run_name=args.run_name,
        split_mode=args.split_mode,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        recall_target=args.recall_target,
        precision_target=args.precision_target,
        multilabel_loss_weight=args.multilabel_loss_weight,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)

    summary = train(config)

    if args.log_mlflow:
        from classification.prefilter.mlflow_utils import log_training_run

        run_id = log_training_run(
            summary=summary,
            artifacts_dir=run_artifacts_dir(config.run_name),
        )
        logger.info("Logged training run to MLflow: %s", run_id)


if __name__ == "__main__":
    main()
