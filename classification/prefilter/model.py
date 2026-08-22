"""
Dual-head transformer classifier for the PII pre-filter.

One shared encoder, two heads:

    binary head       1 logit  -> "does this document contain personal data?"
    multi-label head  12 logits -> which of the 12 GDPR entity types are present

The binary head is what the router acts on; the multi-label head is trained
jointly because the entity signal is a useful auxiliary task and because the
evaluation reports per-entity metrics from it.

The heads read the encoder's ``[CLS]`` representation. DistilBERT has no
pretrained pooler, so a small pre-classifier layer plays that role — the same
arrangement ``DistilBertForSequenceClassification`` uses.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from classification.prefilter.config import ENTITY_LABELS, PreFilterConfig

logger = logging.getLogger(__name__)

CHECKPOINT_WEIGHTS = "prefilter_model.pt"
CHECKPOINT_CONFIG = "config.json"


class PiiPreFilterModel(nn.Module):
    """
    Encoder + binary head + multi-label head.
    """

    def __init__(
        self,
        pretrained_dir: str,
        num_entity_labels: int = len(ENTITY_LABELS),
        classifier_dropout: float = 0.2,
        encoder_config=None,
    ) -> None:
        super().__init__()

        if encoder_config is None:
            self.encoder = AutoModel.from_pretrained(pretrained_dir)
        else:
            # Used when loading a fine-tuned checkpoint: build the architecture
            # empty and let the state dict supply the weights, so the pretrained
            # directory is not required at inference time.
            self.encoder = AutoModel.from_config(encoder_config)

        hidden_size = self.encoder.config.hidden_size

        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(classifier_dropout)

        self.binary_head = nn.Linear(hidden_size, 1)
        self.entity_head = nn.Linear(hidden_size, num_entity_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns ``(binary_logits, entity_logits)`` with shapes ``(b,)`` and
        ``(b, 12)``.
        """

        hidden_state = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        pooled = hidden_state[:, 0]
        pooled = self.activation(self.pre_classifier(pooled))
        pooled = self.dropout(pooled)

        binary_logits = self.binary_head(pooled).squeeze(-1)
        entity_logits = self.entity_head(pooled)

        return binary_logits, entity_logits

    # ── Persistence ─────────────────────────────────────────

    def save(self, output_dir: Path, config: PreFilterConfig) -> Path:
        """
        Save weights, run config and the encoder config to ``output_dir``.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": self.state_dict(),
                "encoder_config": self.encoder.config.to_dict(),
                "entity_labels": ENTITY_LABELS,
            },
            output_dir / CHECKPOINT_WEIGHTS,
        )

        config.save(output_dir / CHECKPOINT_CONFIG)

        return output_dir


def load_checkpoint(
    checkpoint_dir: Path,
    device: torch.device | str = "cpu",
) -> tuple[PiiPreFilterModel, PreFilterConfig]:
    """
    Load a fine-tuned pre-filter from a checkpoint directory.

    The encoder architecture is rebuilt from the stored encoder config, so this
    works without reaching for the original pretrained directory.
    """

    checkpoint_dir = Path(checkpoint_dir)
    weights_file = checkpoint_dir / CHECKPOINT_WEIGHTS
    config_file = checkpoint_dir / CHECKPOINT_CONFIG

    if not weights_file.exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights_file}")

    if not config_file.exists():
        raise FileNotFoundError(f"Checkpoint config not found: {config_file}")

    config = PreFilterConfig.load(config_file)

    payload = torch.load(weights_file, map_location="cpu", weights_only=False)

    stored_labels = payload.get("entity_labels", ENTITY_LABELS)
    if stored_labels != ENTITY_LABELS:
        raise ValueError(
            "Checkpoint entity labels do not match the current label "
            f"vocabulary.\n  checkpoint: {stored_labels}\n  current:    "
            f"{ENTITY_LABELS}"
        )

    encoder_config = AutoConfig.for_model(**payload["encoder_config"])

    model = PiiPreFilterModel(
        pretrained_dir=config.pretrained_dir,
        num_entity_labels=len(ENTITY_LABELS),
        classifier_dropout=config.classifier_dropout,
        encoder_config=encoder_config,
    )

    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    return model, config


def load_tokenizer(pretrained_dir: str | Path):
    """
    Load the tokenizer for the pre-filter encoder.
    """

    pretrained_dir = Path(pretrained_dir)

    if not pretrained_dir.exists():
        raise FileNotFoundError(
            f"Pretrained model directory not found: {pretrained_dir}\n"
            "Run: python -m classification.prefilter.fetch_model"
        )

    return AutoTokenizer.from_pretrained(str(pretrained_dir))


def count_parameters(model: nn.Module) -> dict:
    """
    Parameter counts and the on-disk size the checkpoint will take (fp32).
    """

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "parameters_total": int(total),
        "parameters_trainable": int(trainable),
        "model_size_mb": round(total * 4 / (1024 ** 2), 2),
    }


def build_optimizer_groups(
    model: PiiPreFilterModel,
    learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> list[dict]:
    """
    Parameter groups: a lower LR for the pretrained encoder, a higher one for
    the randomly initialised heads, and no weight decay on biases/LayerNorm.
    """

    no_decay = ("bias", "LayerNorm.weight", "layer_norm")

    encoder_params = list(model.encoder.named_parameters())
    head_params = [
        (name, param)
        for module_name in ("pre_classifier", "binary_head", "entity_head")
        for name, param in getattr(model, module_name).named_parameters()
    ]

    def _group(params, lr, decay):
        return {
            "params": [
                param
                for name, param in params
                if (not any(token in name for token in no_decay)) == (decay > 0)
            ],
            "lr": lr,
            "weight_decay": decay,
        }

    groups = [
        _group(encoder_params, learning_rate, weight_decay),
        _group(encoder_params, learning_rate, 0.0),
        _group(head_params, head_learning_rate, weight_decay),
        _group(head_params, head_learning_rate, 0.0),
    ]

    return [group for group in groups if group["params"]]


def describe_model(model: PiiPreFilterModel) -> str:
    """
    One-line human-readable model description for logs and MLflow.
    """
    stats = count_parameters(model)
    return json.dumps(
        {
            "encoder": model.encoder.config.model_type,
            "layers": getattr(
                model.encoder.config,
                "num_hidden_layers",
                getattr(model.encoder.config, "n_layers", None),
            ),
            "hidden_size": model.encoder.config.hidden_size,
            **stats,
        }
    )
