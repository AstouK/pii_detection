"""
Dataset loading, splitting and tokenisation for the transformer pre-filter.

Two things here are less obvious than they look:

1.  ``recommended_split`` is authoritative *when it is usable*. In the 500-row
    pilot it is not: all 60 positive documents sit in ``train``, leaving
    validation and test without a single positive. Recall is undefined on an
    all-negative split, so the recall-constrained threshold calibration that is
    the point of this work package cannot run against it. :func:`resolve_splits`
    therefore validates the given split and falls back to a seeded stratified
    split, rather than silently producing meaningless numbers.

2.  The pilot contains 171 duplicated ``full_text`` values out of 500. Any
    split that assigns duplicates independently leaks training documents into
    validation and test. The fallback splitter groups on a hash of the
    normalised text so identical documents stay together.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from classification.prefilter.config import (
    BINARY_LABEL_COL,
    DOCUMENT_ID_COL,
    ENTITY_LABELS,
    PreFilterConfig,
    SPLIT_COL,
    SPLIT_NAMES,
    TEST_SPLIT,
    TEXT_COL,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    entity_label_columns,
)

logger = logging.getLogger(__name__)

_TRUTHY = {"yes", "true", "1", "y", "ja", "pii"}


# ─────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────

def to_bool_series(series: pd.Series) -> pd.Series:
    """
    Normalise a yes/no-style column to booleans.

    Mirrors ``classification.evaluation.metrics.to_bool_series`` so that a
    label read here and a label read by the evaluation always agree. Unknown
    and missing values become ``False``.
    """

    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        return str(value).strip().lower() in _TRUTHY

    return series.apply(_to_bool)


def load_dataset(data_file: str | Path) -> pd.DataFrame:
    """
    Load the classification dataset and validate the columns we depend on.

    Entity columns that are absent are created as all-``no`` rather than
    raising, so a dataset revision that drops a rare entity type still trains.
    """

    data_file = Path(data_file)

    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    df = pd.read_csv(data_file)

    required = [DOCUMENT_ID_COL, TEXT_COL, BINARY_LABEL_COL]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset {data_file} is missing required columns: {missing}"
        )

    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    for column in entity_label_columns():
        if column not in df.columns:
            logger.warning(
                "Entity column '%s' missing from dataset; treating as all-'no'",
                column,
            )
            df[column] = "no"

    return df


def extract_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(binary_labels, entity_labels)`` as float arrays.

    Shapes: ``(n,)`` and ``(n, 12)``.
    """

    binary = to_bool_series(df[BINARY_LABEL_COL]).to_numpy(dtype=np.float32)

    entity = np.stack(
        [
            to_bool_series(df[column]).to_numpy(dtype=np.float32)
            for column in entity_label_columns()
        ],
        axis=1,
    )

    return binary, entity


# ─────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────

def _text_group_key(text: str) -> str:
    """
    Stable grouping key for near-identical documents.

    Whitespace is collapsed and case folded so that formatting-only differences
    do not split an otherwise duplicated document across two splits.
    """
    normalised = " ".join(str(text).split()).lower()
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


def describe_split(df: pd.DataFrame, split_col: str = SPLIT_COL) -> pd.DataFrame:
    """
    Per-split row counts and positive counts.
    """

    positive = to_bool_series(df[BINARY_LABEL_COL])

    summary = (
        pd.DataFrame(
            {
                "split": df[split_col],
                "positive": positive,
            }
        )
        .groupby("split")
        .agg(n=("positive", "size"), positives=("positive", "sum"))
        .reset_index()
    )

    summary["negatives"] = summary["n"] - summary["positives"]
    summary["positive_rate"] = (summary["positives"] / summary["n"]).round(4)

    return summary


def validate_split(df: pd.DataFrame, split_col: str = SPLIT_COL) -> list[str]:
    """
    Return the reasons a split cannot be used, or an empty list if it is fine.

    A split is usable when every one of train/validation/test exists and holds
    at least one positive and one negative document. Validation needs positives
    to calibrate the router; test needs them to report recall; train needs both
    to learn anything.
    """

    problems: list[str] = []

    if split_col not in df.columns:
        return [f"split column '{split_col}' is missing"]

    summary = describe_split(df, split_col).set_index("split")

    for split_name in SPLIT_NAMES:
        if split_name not in summary.index:
            problems.append(f"split '{split_name}' is absent")
            continue

        row = summary.loc[split_name]

        if row["positives"] == 0:
            problems.append(
                f"split '{split_name}' contains 0 positive documents "
                f"(n={int(row['n'])})"
            )

        if row["negatives"] == 0:
            problems.append(
                f"split '{split_name}' contains 0 negative documents "
                f"(n={int(row['n'])})"
            )

    unexpected = set(df[split_col].dropna().unique()) - set(SPLIT_NAMES)
    if unexpected:
        problems.append(f"unexpected split values: {sorted(unexpected)}")

    return problems


def make_stratified_split(
    df: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    group_duplicate_texts: bool = True,
) -> pd.Series:
    """
    Build a label-stratified, duplicate-grouped train/validation/test split.

    Groups (identical documents) are the unit of assignment, and each label
    stratum is dealt round-robin into the three splits in shuffled order, so
    the positive rate is preserved as closely as group sizes allow.

    Returns a Series of split names aligned to ``df.index``.
    """

    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be in (0, 1)")

    if validation_fraction + test_fraction >= 1:
        raise ValueError(
            "validation_fraction + test_fraction must leave room for training"
        )

    positive = to_bool_series(df[BINARY_LABEL_COL])

    if group_duplicate_texts:
        groups = df[TEXT_COL].map(_text_group_key)
    else:
        # Every row its own group.
        groups = pd.Series(df.index.astype(str), index=df.index)

    # A duplicated document could in principle carry conflicting labels; treat
    # a group as positive if any member is, so positives are never diluted into
    # a split that thinks it is all-negative.
    group_frame = pd.DataFrame({"group": groups, "positive": positive})
    group_positive = group_frame.groupby("group")["positive"].max()

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}

    train_fraction = 1.0 - validation_fraction - test_fraction
    targets = [
        (TRAIN_SPLIT, train_fraction),
        (VALIDATION_SPLIT, validation_fraction),
        (TEST_SPLIT, test_fraction),
    ]

    for is_positive in (True, False):
        stratum = group_positive[group_positive == is_positive].index.to_numpy()
        rng.shuffle(stratum)

        # Deal by cumulative quota: walk the shuffled groups and hand each one
        # to whichever split is furthest below its target share so far.
        counts = {name: 0.0 for name, _ in targets}

        for group_id in stratum:
            deficits = [
                (counts[name] / max(fraction, 1e-9), name)
                for name, fraction in targets
            ]
            deficits.sort()
            chosen = deficits[0][1]
            assignment[group_id] = chosen
            counts[chosen] += 1.0

    return groups.map(assignment)


def resolve_splits(
    df: pd.DataFrame,
    config: PreFilterConfig,
) -> tuple[pd.DataFrame, str]:
    """
    Attach a usable ``split`` column to ``df``.

    Returns ``(df_with_split, resolved_mode)`` where ``resolved_mode`` is one of
    ``recommended`` or ``stratified`` and is recorded in the run config.

    ``auto`` prefers the dataset's own ``recommended_split`` and only falls back
    when :func:`validate_split` rejects it. The fallback is logged at WARNING
    with the specific reasons, because using a split other than Sonja's makes
    this model's numbers non-comparable to the rest of the pipeline and the team
    has to know that happened.
    """

    mode = config.split_mode.lower().strip()

    if mode not in {"auto", "recommended", "stratified"}:
        raise ValueError(
            f"Unknown split_mode '{config.split_mode}'. "
            "Expected one of: auto, recommended, stratified."
        )

    df = df.copy()
    problems = validate_split(df)

    if mode == "recommended":
        if problems:
            logger.warning(
                "split_mode='recommended' but the split is not usable: %s",
                "; ".join(problems),
            )
        df["split"] = df[SPLIT_COL]
        return df, "recommended"

    if mode == "auto" and not problems:
        logger.info("Using dataset '%s' as-is (validated).", SPLIT_COL)
        df["split"] = df[SPLIT_COL]
        return df, "recommended"

    if mode == "auto":
        logger.warning(
            "Dataset '%s' is NOT usable for recall-constrained calibration: %s",
            SPLIT_COL,
            "; ".join(problems),
        )
        logger.warning(
            "Falling back to a seeded stratified split (seed=%s). "
            "Metrics from this run are NOT comparable to runs that use the "
            "dataset's own split. Report this to the team.",
            config.seed,
        )

    df["split"] = make_stratified_split(
        df=df,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
        group_duplicate_texts=config.group_duplicate_texts,
    )

    return df, "stratified"


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split a resolved dataframe into ``{split_name: frame}``.
    """
    return {
        name: df[df["split"] == name].reset_index(drop=True)
        for name in SPLIT_NAMES
    }


# ─────────────────────────────────────────────────────────────
# Torch dataset
# ─────────────────────────────────────────────────────────────

class PiiDocumentDataset(Dataset):
    """
    Tokenised documents with a binary label and 12 entity labels.

    Tokenisation happens once in ``__init__``: the corpus is at most 1.4k short
    documents, so pre-tokenising is cheaper than re-encoding every epoch.
    """

    def __init__(
        self,
        texts: list[str],
        binary_labels: np.ndarray | None,
        entity_labels: np.ndarray | None,
        tokenizer,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        self.binary_labels = (
            torch.tensor(binary_labels, dtype=torch.float32)
            if binary_labels is not None
            else None
        )

        self.entity_labels = (
            torch.tensor(entity_labels, dtype=torch.float32)
            if entity_labels is not None
            else None
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, index: int) -> dict:
        item = {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
        }

        if self.binary_labels is not None:
            item["binary_label"] = self.binary_labels[index]

        if self.entity_labels is not None:
            item["entity_labels"] = self.entity_labels[index]

        return item


def build_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    with_labels: bool = True,
) -> PiiDocumentDataset:
    """
    Build a :class:`PiiDocumentDataset` from a dataframe slice.
    """

    binary_labels, entity_labels = (
        extract_labels(df) if with_labels else (None, None)
    )

    return PiiDocumentDataset(
        texts=df[TEXT_COL].tolist(),
        binary_labels=binary_labels,
        entity_labels=entity_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )


def build_dataloader(
    dataset: PiiDocumentDataset,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    """
    Wrap a dataset in a DataLoader with a seeded shuffle generator.
    """

    generator = None

    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def compute_pos_weight(
    binary_labels: np.ndarray,
    max_pos_weight: float,
) -> float:
    """
    ``pos_weight`` for the binary BCE loss: negatives per positive, capped.

    The pilot is ~12% positive, giving ~7.3. The cap keeps a much rarer positive
    class in a scaled dataset from producing a weight large enough to make
    training diverge.
    """

    positives = float(binary_labels.sum())
    negatives = float(len(binary_labels) - positives)

    if positives <= 0:
        logger.warning("No positive documents in training data; pos_weight=1.0")
        return 1.0

    return float(min(negatives / positives, max_pos_weight))


def compute_entity_pos_weights(
    entity_labels: np.ndarray,
    max_pos_weight: float,
) -> np.ndarray:
    """
    Per-label ``pos_weight`` for the 12-label BCE loss.

    Entity labels are far rarer than the binary label (3/500 for IBAN_CODE in
    the pilot), so the cap does most of the work here.
    """

    positives = entity_labels.sum(axis=0)
    negatives = len(entity_labels) - positives

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(positives > 0, negatives / np.maximum(positives, 1), 1.0)

    return np.clip(weights, 1.0, max_pos_weight).astype(np.float32)


def entity_support(df: pd.DataFrame) -> pd.DataFrame:
    """
    Positive count per entity label — used by the EDA and error reports.
    """

    _, entity = extract_labels(df)

    return pd.DataFrame(
        {
            "entity": ENTITY_LABELS,
            "positives": entity.sum(axis=0).astype(int),
            "n": len(df),
        }
    ).assign(positive_rate=lambda frame: (frame.positives / frame.n).round(4))
