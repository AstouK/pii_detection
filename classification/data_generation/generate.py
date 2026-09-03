"""Command-line entry point for production synthetic dataset generation.

Production generation always enables LLM enrichment.

The lower-level generator API still supports ``use_llm=False`` for
deterministic regression tests and development workflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from classification.data_generation.archetypes import ARCHETYPES
from classification.data_generation.generator import (
    generate_dataset_dataframe,
    generate_dataset_manifest,
    save_dataset,
    save_manifest,
)


DEFAULT_SEED = 42
MIN_DOCUMENTS_PER_SCENARIO = 38


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic PII classification dataset "
            "with LLM enrichment enabled."
        )
    )

    parser.add_argument(
        "--documents-per-scenario",
        type=int,
        default=DEFAULT_DOCUMENTS_PER_SCENARIO,
        help=(
            "Number of documents generated for each scenario "
            f"(default: {DEFAULT_DOCUMENTS_PER_SCENARIO})."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Base random seed used for deterministic planning "
            f"(default: {DEFAULT_SEED})."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output CSV filename. "
            "If omitted, the filename is derived automatically "
            "from the total dataset size, e.g. "
            "synthetic_dataset_1400.csv."
        ),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """Validate generation arguments."""

    if args.documents_per_scenario < MIN_DOCUMENTS_PER_SCENARIO:
        raise ValueError(
            "--documents-per-scenario must be at least "
            f"{MIN_DOCUMENTS_PER_SCENARIO} so every split contains "
            "both positive and negative examples."
        )

    if args.output is None:
        return

    output_path = Path(args.output)

    if output_path.name != args.output:
        raise ValueError(
            "--output must be a filename, not a path. "
            "Files are written to the configured data-generation "
            "output directory."
        )

    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            "--output must use the .csv extension."
        )


def dataset_file_name(
    documents_per_scenario: int,
    output_override: str | None = None,
) -> str:
    """Return the dataset filename for one generation run."""

    if output_override is not None:
        return output_override

    total_documents = (
        documents_per_scenario
        * len(ARCHETYPES)
    )

    return (
        f"synthetic_dataset_"
        f"{total_documents}.csv"
    )


def manifest_file_name(
    dataset_name: str,
) -> str:
    """Derive the manifest filename from the dataset filename."""

    path = Path(dataset_name)

    return (
        f"{path.stem}_manifest.csv"
    )


def main() -> int:
    """Generate and save a production synthetic dataset."""

    args = parse_args()

    try:
        validate_args(args)
    except ValueError as error:
        print(
            f"Invalid arguments: {error}"
        )
        return 2

    dataset_name = dataset_file_name(
        documents_per_scenario=(
            args.documents_per_scenario
        ),
        output_override=args.output,
    )

    manifest_name = manifest_file_name(
        dataset_name
    )

    total_documents = (
        args.documents_per_scenario
        * len(ARCHETYPES)
    )

    print("=" * 72)
    print("SYNTHETIC DATASET GENERATION")
    print("=" * 72)

    print(
        "Scenarios:",
        len(ARCHETYPES),
    )

    print(
        "Documents per scenario:",
        args.documents_per_scenario,
    )

    print(
        "Total documents:",
        total_documents,
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Dataset file:",
        dataset_name,
    )

    print(
        "Manifest file:",
        manifest_name,
    )

    print(
        "LLM enrichment: enabled"
    )

    print()

    # Production generation always uses LLM enrichment.
    dataset = generate_dataset_dataframe(
        documents_per_scenario=(
            args.documents_per_scenario
        ),
        seed=args.seed,
        use_llm=True,
    )

    # Generate matching internal metadata.
    manifest = generate_dataset_manifest(
        documents_per_scenario=(
            args.documents_per_scenario
        ),
        seed=args.seed,
    )

    # Safety checks before saving.
    if len(dataset) != len(manifest):
        raise RuntimeError(
            "Dataset and manifest row counts do not match: "
            f"{len(dataset)} != {len(manifest)}"
        )

    dataset_ids = (
        dataset["document_id"]
        .astype(str)
        .tolist()
    )

    manifest_ids = (
        manifest["document_id"]
        .astype(str)
        .tolist()
    )

    if dataset_ids != manifest_ids:
        raise RuntimeError(
            "Dataset and manifest document IDs "
            "do not align one-to-one."
        )

    dataset_path = save_dataset(
        df=dataset,
        file_name=dataset_name,
    )

    manifest_path = save_manifest(
        manifest=manifest,
        file_name=manifest_name,
    )

    print()
    print("=" * 72)
    print("GENERATION COMPLETE")
    print("=" * 72)

    print(
        "Dataset:",
        dataset_path,
    )

    print(
        "Dataset shape:",
        dataset.shape,
    )

    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "Manifest shape:",
        manifest.shape,
    )

    print()
    print("Labels:")
    print(
        dataset[
            "contains_personal_data"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Splits:")
    print(
        dataset[
            "recommended_split"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Generation methods:")
    print(
        dataset[
            "synthetic_generator"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Prompt versions:")
    print(
        dataset[
            "generation_prompt_version"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Audit command:")
    print(
        "python -m classification.data_generation.audit "
        f"{dataset_path} "
        f"--manifest {manifest_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )