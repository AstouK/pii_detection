"""
Configuration for the transformer pre-filter.

Everything that changes a run lives in :class:`PreFilterConfig`, which is
serialised into every checkpoint and every run directory. A run is reproducible
from its ``config.json`` plus the pinned seed.

Non-run-specific constants (label vocabulary, output-interface column names)
are module-level, because the evaluation module depends on their exact
spelling.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from classification.config import (
    DEFAULT_INPUT_FILE,
    DEFAULT_PIPELINE_NAME,
    RESULTS_DIR,
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

PREFILTER_DIR = Path(__file__).resolve().parent

#: Checkpoints, calibration output and plots. Git-ignored.
ARTIFACTS_DIR = PREFILTER_DIR / "artifacts"

#: Small, committed artefacts the team is meant to read (curves, summaries).
REPORTS_DIR = PREFILTER_DIR / "reports"

#: Classification run outputs, shared with the rest of the pipeline:
#: ``classification/results/runs/<run_id>/<strategy>.csv``
CLASSIFICATION_RUNS_DIR = RESULTS_DIR / "runs"

#: Local copy of the pretrained encoder.
#:
#: huggingface.co is not reachable from the training environment, so the
#: weights are fetched once into this directory. See ``scripts/fetch_model.py``.
DEFAULT_PRETRAINED_DIR = (
    PREFILTER_DIR.parent.parent / "models" / "distilbert-base-uncased"
)


# ─────────────────────────────────────────────────────────────
# Label vocabulary
# ─────────────────────────────────────────────────────────────

#: The 12 entity types the multi-label head predicts.
#:
#: The evaluation module derives per-entity metrics from ground-truth columns
#: matching ``<ENTITY_TYPE>_yes_no`` (see
#: ``classification.evaluation.metrics.get_entity_types_from_columns``), so the
#: spelling here must match the dataset columns exactly.
ENTITY_LABELS: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "IBAN_CODE",
    "CREDIT_CARD",
    "PASSPORT",
    "NRP",
    "DATE_TIME",
    "IP_ADDRESS",
    "URL",
    "MEDICAL_LICENSE",
]

#: Ground-truth column suffix for entity labels.
ENTITY_LABEL_SUFFIX = "_yes_no"

#: Dataset columns consumed by the pre-filter.
TEXT_COL = "full_text"
DOCUMENT_ID_COL = "document_id"
BINARY_LABEL_COL = "contains_personal_data"
SPLIT_COL = "recommended_split"

#: Dataset columns carried into the prediction CSV for error slicing.
#: Missing columns are skipped rather than raising.
CONTEXT_COLS: list[str] = [
    "file_name",
    "document_type",
    "scenario_type",
    "language",
    "difficulty",
    "edge_case",
    "challenge_category",
    "personal_data_categories",
    "primary_pii_type",
    "category_count",
    "pii_count",
    "dataset_version",
    SPLIT_COL,
]

#: Split names as they appear in ``recommended_split``.
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

SPLIT_NAMES = [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT]


# ─────────────────────────────────────────────────────────────
# Output interface (see classification/prefilter/README.md)
# ─────────────────────────────────────────────────────────────

#: Standardised prediction column the evaluation reads.
PREDICTION_COL = "predicted_pii"

#: Strategy written as the *routed* pipeline output. Registered in
#: ``classification/evaluation/config.py::STRATEGIES``.
ROUTED_STRATEGY = "rule_plus_bert"

#: Strategy written as the standalone-model output (no routing applied).
STANDALONE_STRATEGY = "bert_prefilter"

PROVIDER = "local"
MODEL_FAMILY = "bert"
PREDICTION_SOURCE = "local_model"
PREDICTION_STAGE = "bert_prefilter"
PIPELINE_NAME = DEFAULT_PIPELINE_NAME

#: MLflow experiment. Kept identical to the evaluation module's experiment so
#: training runs and evaluation runs land side by side.
MLFLOW_EXPERIMENT_NAME = "gdpr-pii-detection-evaluation"


# ─────────────────────────────────────────────────────────────
# Run configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class PreFilterConfig:
    """
    Full specification of a training + calibration run.

    Serialised to ``<artifacts>/<run_name>/config.json`` and stored inside the
    checkpoint, so that ``predict`` reconstructs the exact preprocessing the
    model was trained with.
    """

    # ── Data ────────────────────────────────────────────────
    data_file: str = str(DEFAULT_INPUT_FILE)

    #: How train/validation/test are obtained.
    #:
    #: ``recommended``  use ``recommended_split`` verbatim (Sonja's split).
    #: ``stratified``   build a seeded, label-stratified, duplicate-grouped
    #:                  split with the proportions below.
    #: ``auto``         use ``recommended`` when it is usable, otherwise fall
    #:                  back to ``stratified`` with a loud warning.
    #:
    #: ``auto`` exists because the 500-row pilot currently in the repo puts all
    #: 60 positives in ``train``: validation and test contain zero positive
    #: documents, so recall on them is undefined and the recall-constrained
    #: threshold calibration cannot run. See README -> "Findings for the team".
    split_mode: str = "auto"

    #: Proportions used by the ``stratified`` fallback. Chosen to match the
    #: 350/75/75 shape of ``recommended_split``.
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    #: Exact-duplicate documents are assigned to the same split. The pilot
    #: dataset contains 171/500 duplicated ``full_text`` values; splitting them
    #: independently leaks training documents into validation and test.
    group_duplicate_texts: bool = True

    # ── Model ───────────────────────────────────────────────
    pretrained_dir: str = str(DEFAULT_PRETRAINED_DIR)

    #: Reported in the output CSV's ``model_name`` column.
    model_name: str = "distilbert-base-uncased-v1"

    #: Documents are at most ~821 characters (~200 word-piece tokens), so 256
    #: covers the full corpus and no chunking is required.
    max_length: int = 256

    classifier_dropout: float = 0.2

    # ── Optimisation ────────────────────────────────────────
    epochs: int = 8
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 3e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    #: Weight of the 12-label entity loss relative to the binary loss.
    #:
    #: 1.0 rather than 0.5: the binary task converges within two epochs and its
    #: gradients dominate the shared encoder, so halving the entity gradient on
    #: top of that left the entity head sitting at the base rate.
    multilabel_loss_weight: float = 1.0

    #: Cap on ``pos_weight`` in the binary BCE loss. The pilot is ~12%
    #: positive, so the uncapped weight is ~7.3; the cap keeps a much rarer
    #: positive class in the scaled dataset from destabilising training.
    max_pos_weight: float = 12.0

    #: Separate, much higher cap for the entity labels. They are an order of
    #: magnitude rarer than the binary label — 3 of 500 for IBAN_CODE in the
    #: pilot, an uncapped weight of ~122 — so the binary cap of 12 throttled
    #: exactly the labels that needed the most help.
    max_entity_pos_weight: float = 50.0

    #: Model selection criterion on the validation split. ``routing_cost`` picks
    #: the checkpoint that routes the fewest documents to the LLM while holding
    #: the recall target, which is what the work package is actually optimising.
    #: Falls back to ``pr_auc`` when no threshold pair satisfies the target.
    model_selection_metric: str = "routing_cost"

    # ── Routing ─────────────────────────────────────────────
    #: Minimum document-level recall the router must preserve. 0.98 is the
    #: rule-based baseline (0.9833) rounded down.
    recall_target: float = 0.98

    #: Minimum precision required of the "confident PII" zone. Without an
    #: upper-zone constraint the search is degenerate: routing nothing above
    #: ``t_low`` always minimises LLM calls, which collapses the router back to
    #: a single threshold and pushes every auto-approved false positive through
    #: unreviewed.
    precision_target: float = 0.90

    #: Resolution of the threshold grid searched during calibration.
    threshold_grid_steps: int = 201

    #: Probability threshold for the standalone (unrouted) model output.
    standalone_threshold: float = 0.5

    #: Fallback probability threshold for the 12 entity labels.
    #:
    #: Only used for labels that have no positive validation document to
    #: calibrate against. Per-label thresholds are fitted on validation during
    #: training and stored in ``calibration.json``; a fixed 0.5 cut suppressed
    #: the head entirely on the pilot, where its highest score on a true
    #: PERSON document was 0.49.
    entity_threshold: float = 0.5

    # ── Reproducibility ─────────────────────────────────────
    seed: int = 42
    run_name: str = "distilbert_prefilter"

    #: Populated at runtime; recorded so a run directory is self-describing.
    resolved_split_mode: str = ""
    device: str = ""

    extra: dict = field(default_factory=dict)

    # ── Helpers ─────────────────────────────────────────────
    @property
    def num_entity_labels(self) -> int:
        return len(ENTITY_LABELS)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_dict(cls, values: dict) -> "PreFilterConfig":
        """
        Build a config from a dict, ignoring keys this version does not know.

        Tolerating unknown keys keeps older checkpoints loadable after the
        config grows a field.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    @classmethod
    def load(cls, path: Path) -> "PreFilterConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def entity_label_columns() -> list[str]:
    """
    Ground-truth column names for the 12 entity labels.
    """
    return [f"{label}{ENTITY_LABEL_SUFFIX}" for label in ENTITY_LABELS]


def run_artifacts_dir(run_name: str) -> Path:
    """
    Directory holding the checkpoint and calibration output for a run.
    """
    return ARTIFACTS_DIR / run_name
