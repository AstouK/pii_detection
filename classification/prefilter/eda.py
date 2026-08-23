"""
Dataset report for the pre-filter work package.

Answers the questions Phase 1 of the brief asks — class balance, the
distribution of the 12 entity labels, coverage across ``document_type`` and
``difficulty`` — and additionally checks the two dataset properties that decide
whether the training setup is sound at all:

    * does every split contain both classes?  (recall is undefined otherwise)
    * how many documents are exact duplicates? (they leak across splits)

Run:

    python -m classification.prefilter.eda
    python -m classification.prefilter.eda --output-dir classification/prefilter/reports
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from config.logging_config import setup_logging

from classification.prefilter.config import (
    BINARY_LABEL_COL,
    REPORTS_DIR,
    SPLIT_COL,
    TEXT_COL,
    PreFilterConfig,
)
from classification.prefilter.data import (
    _text_group_key,
    describe_split,
    entity_support,
    load_dataset,
    to_bool_series,
    validate_split,
)

setup_logging()
logger = logging.getLogger(__name__)


def crosstab_by_label(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Positive/negative counts and positive rate for one grouping column.
    """

    if column not in df.columns:
        return pd.DataFrame()

    positive = to_bool_series(df[BINARY_LABEL_COL])

    summary = (
        pd.DataFrame({column: df[column], "positive": positive})
        .groupby(column)
        .agg(n=("positive", "size"), positives=("positive", "sum"))
        .reset_index()
    )

    summary["negatives"] = summary["n"] - summary["positives"]
    summary["positive_rate"] = (summary["positives"] / summary["n"]).round(4)

    return summary.sort_values("n", ascending=False).reset_index(drop=True)


#: Sequence lengths worth considering, cheapest first. 512 is DistilBERT's
#: hard limit (learned position embeddings), so beyond it the only options are
#: truncation or chunking.
CANDIDATE_MAX_LENGTHS = [128, 256, 384, 512]

#: A document may lose this share of its tail before the setting is rejected.
#: Not zero: chasing a handful of outliers to 512 doubles training time for
#: every document in the corpus.
TRUNCATION_TOLERANCE = 0.02


def token_lengths(
    df: pd.DataFrame,
    pretrained_dir: str | None,
) -> "pd.Series | None":
    """
    Real word-piece lengths, when a tokenizer is available.

    Character-count heuristics are unreliable across languages — German
    compounds fragment into more sub-words per character than English prose —
    so the recommendation below prefers measured lengths and only falls back to
    an estimate when no tokenizer is given.
    """

    if not pretrained_dir or not Path(pretrained_dir).exists():
        return None

    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(pretrained_dir))

    return pd.Series(
        [
            len(tokenizer(text, truncation=False)["input_ids"])
            for text in df[TEXT_COL].fillna("")
        ]
    )


def text_length_stats(
    df: pd.DataFrame,
    pretrained_dir: str | None = None,
) -> dict:
    """
    Length distribution, and the ``max_length`` it actually implies.

    The recommendation is computed, not asserted. The 500-row pilot fit
    comfortably in 256 tokens; the 1,400-row set does not, and a hard-coded
    claim that it does would have quietly truncated a quarter of the corpus.
    """

    lengths = df[TEXT_COL].str.len()

    stats = {
        "min": int(lengths.min()),
        "median": int(lengths.median()),
        "p90": int(lengths.quantile(0.90)),
        "p99": int(lengths.quantile(0.99)),
        "max": int(lengths.max()),
    }

    measured = token_lengths(df, pretrained_dir)

    if measured is not None:
        tokens = measured
        stats["token_source"] = "measured"
    else:
        # ~3.5 characters per word-piece as a rough upper bound. Pessimistic on
        # this corpus (it predicted 748 tokens where the tokenizer found 552),
        # which is the safe direction for a truncation decision.
        tokens = lengths / 3.5
        stats["token_source"] = "estimated"

    stats["max_tokens"] = int(tokens.max())
    stats["p95_tokens"] = int(tokens.quantile(0.95))

    stats["truncated_at"] = {
        str(candidate): round(float((tokens > candidate).mean()), 4)
        for candidate in CANDIDATE_MAX_LENGTHS
    }

    stats["recommended_max_length"] = next(
        (
            candidate
            for candidate in CANDIDATE_MAX_LENGTHS
            if (tokens > candidate).mean() <= TRUNCATION_TOLERANCE
        ),
        CANDIDATE_MAX_LENGTHS[-1],
    )

    return stats


def duplicate_stats(df: pd.DataFrame) -> dict:
    """
    Exact-duplicate document counts, and whether duplicates cross splits.

    Duplicates that straddle a split boundary mean validation and test scores
    are measured partly on memorised training documents.
    """

    groups = df[TEXT_COL].map(_text_group_key)

    group_sizes = groups.value_counts()
    duplicated_groups = group_sizes[group_sizes > 1]

    stats = {
        "n_documents": int(len(df)),
        "n_unique_texts": int(groups.nunique()),
        "n_duplicated_rows": int(len(df) - groups.nunique()),
        "n_duplicate_groups": int(len(duplicated_groups)),
        "largest_duplicate_group": (
            int(duplicated_groups.max()) if not duplicated_groups.empty else 0
        ),
    }

    if SPLIT_COL in df.columns:
        splits_per_group = (
            pd.DataFrame({"group": groups, "split": df[SPLIT_COL]})
            .groupby("group")["split"]
            .nunique()
        )
        stats["duplicate_groups_spanning_splits"] = int(
            (splits_per_group > 1).sum()
        )

    return stats


def build_report(
    df: pd.DataFrame,
    pretrained_dir: str | None = None,
) -> dict:
    """
    Assemble the full EDA report as a plain dict.
    """

    positive = to_bool_series(df[BINARY_LABEL_COL])

    report: dict = {
        "n_documents": int(len(df)),
        "n_positive": int(positive.sum()),
        "n_negative": int((~positive).sum()),
        "positive_rate": round(float(positive.mean()), 4),
        "text_length_chars": text_length_stats(df, pretrained_dir),
        "duplicates": duplicate_stats(df),
        "entity_support": entity_support(df).to_dict(orient="records"),
        "split_problems": validate_split(df),
    }

    if SPLIT_COL in df.columns:
        report["split_summary"] = describe_split(df).to_dict(orient="records")

    for column in ("document_type", "difficulty", "challenge_category",
                   "scenario_type", "language", "primary_pii_type"):
        table = crosstab_by_label(df, column)
        if not table.empty:
            report[f"by_{column}"] = table.to_dict(orient="records")

    return report


def print_report(report: dict) -> None:
    """
    Print the report in the shape a human wants to read it.
    """

    print("\n" + "=" * 66)
    print("PII DATASET — EDA")
    print("=" * 66)

    print(f"\ndocuments      : {report['n_documents']}")
    print(f"positive       : {report['n_positive']} "
          f"({100 * report['positive_rate']:.1f}%)")
    print(f"negative       : {report['n_negative']}")

    lengths = report["text_length_chars"]
    print(f"\ntext length    : min={lengths['min']} median={lengths['median']} "
          f"p90={lengths['p90']} max={lengths['max']} chars")

    recommended = lengths["recommended_max_length"]
    truncated = lengths["truncated_at"][str(recommended)]

    print(f"                 {lengths['max_tokens']} tokens max, "
          f"{lengths['p95_tokens']} at p95 ({lengths['token_source']})")
    print(f"                 -> max_length={recommended}, which truncates "
          f"{100 * truncated:.1f}% of documents")

    for candidate, share in lengths["truncated_at"].items():
        marker = " <-" if int(candidate) == recommended else ""
        print(f"                    {candidate:>4}: {100 * share:5.1f}% truncated{marker}")

    duplicates = report["duplicates"]
    print(f"\nduplicates     : {duplicates['n_duplicated_rows']} of "
          f"{duplicates['n_documents']} rows are repeats of another document "
          f"({duplicates['n_unique_texts']} unique texts)")
    if "duplicate_groups_spanning_splits" in duplicates:
        print(f"                 {duplicates['duplicate_groups_spanning_splits']} "
              "duplicate groups span more than one split")

    if "split_summary" in report:
        print("\nsplit summary  :")
        print(pd.DataFrame(report["split_summary"]).to_string(index=False))

    problems = report["split_problems"]
    if problems:
        print("\n  ⚠ SPLIT IS NOT USABLE AS-IS:")
        for problem in problems:
            print(f"    - {problem}")
        print("    A split with no positive documents makes recall undefined, so")
        print("    the recall-constrained threshold calibration cannot run on it.")
    else:
        print("\n  ✓ split is usable (every split has both classes)")

    print("\nentity labels  :")
    print(pd.DataFrame(report["entity_support"]).to_string(index=False))

    for column in ("document_type", "difficulty", "challenge_category"):
        key = f"by_{column}"
        if key in report:
            print(f"\nby {column}:")
            print(pd.DataFrame(report[key]).to_string(index=False))

    print("\n" + "=" * 66 + "\n")


def main(argv: list[str] | None = None) -> None:
    defaults = PreFilterConfig()

    parser = argparse.ArgumentParser(
        description="EDA report for the PII classification dataset."
    )
    parser.add_argument("--data-file", default=defaults.data_file)
    parser.add_argument("--output-dir", default=str(REPORTS_DIR))
    parser.add_argument(
        "--pretrained-dir",
        default=defaults.pretrained_dir,
        help=(
            "Tokenizer used to measure real sequence lengths. Without it the "
            "max_length recommendation falls back to a character estimate."
        ),
    )

    args = parser.parse_args(argv)

    df = load_dataset(args.data_file)
    report = build_report(df, pretrained_dir=args.pretrained_dir)

    print_report(report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Named after the dataset, not fixed. A single `eda_report.json` meant that
    # running the EDA on a second dataset silently overwrote the first one's
    # report — and the pilot's report is what documents the broken-split finding.
    output_file = output_dir / f"eda_report_{Path(args.data_file).stem}.json"
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("EDA report written to %s", output_file)


if __name__ == "__main__":
    main()
