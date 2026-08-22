"""Plan balanced synthetic-document requests across scenarios and splits."""


import random

from classification.data_generation.archetypes import get_archetype
from classification.data_generation.models import GenerationRequest
from classification.data_generation.config import (
    SCENARIO_ENTITY_COMBINATIONS,
    SPLIT_RATIOS,
    TARGET_BLANK_RATE,
    TARGET_LANGUAGE_RATIOS,
    TARGET_POSITIVE_RATE,
)


def allocate_counts(
    total: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    """Allocate an integer total across ratios without losing or adding rows."""

    if total < 0:
        raise ValueError("total must be non-negative.")

    if not ratios:
        raise ValueError("ratios cannot be empty.")

    ratio_sum = sum(ratios.values())

    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            f"Ratios must sum to 1.0, got {ratio_sum:.6f}."
        )

    raw_counts = {
        name: total * ratio
        for name, ratio in ratios.items()
    }

    counts = {
        name: int(value)
        for name, value in raw_counts.items()
    }

    remaining = total - sum(counts.values())

    remainders = sorted(
        ratios,
        key=lambda name: raw_counts[name] - counts[name],
        reverse=True,
    )

    for name in remainders[:remaining]:
        counts[name] += 1

    return counts


def allocate_positive_counts(
    positive_count: int,
    split_counts: dict[str, int],
) -> dict[str, int]:
    """Allocate positive examples across splits while preserving totals."""

    if positive_count < 0:
        raise ValueError(
            "positive_count must be non-negative."
        )

    if positive_count > sum(split_counts.values()):
        raise ValueError(
            "positive_count cannot exceed total split capacity."
        )

    split_total = sum(split_counts.values())

    split_ratios = {
        split: count / split_total
        for split, count in split_counts.items()
    }

    positive_by_split = allocate_counts(
        positive_count,
        split_ratios,
    )

    return positive_by_split


def validate_split_class_coverage(
    split_details: dict[str, dict[str, int]],
) -> None:
    """Ensure each non-empty split contains both target classes."""

    for split, counts in split_details.items():
        total = counts["total"]
        positive = counts["positive"]
        negative = counts["negative"]

        if total == 0:
            continue

        if positive == 0 or negative == 0:
            raise ValueError(
                f"Split '{split}' does not contain both target classes: "
                f"{positive} positive, {negative} negative."
            )


def calculate_scenario_counts(
    n_documents: int,
) -> dict:
    """Calculate target and split counts for one scenario."""

    if n_documents <= 0:
        raise ValueError(
            "n_documents must be greater than zero."
        )

    positive_count = round(
        n_documents * TARGET_POSITIVE_RATE
    )

    negative_count = (
        n_documents - positive_count
    )

    split_counts = allocate_counts(
        n_documents,
        SPLIT_RATIOS,
    )

    positive_by_split = allocate_positive_counts(
        positive_count,
        split_counts,
    )


    split_details = {}

    for split, total_count in split_counts.items():
        positive = positive_by_split[split]
        negative = total_count - positive

        split_details[split] = {
            "total": total_count,
            "positive": positive,
            "negative": negative,
        }

    validate_split_class_coverage(split_details)

    return {
        "total": n_documents,
        "positive": positive_count,
        "negative": negative_count,
        "splits": split_details,
    }


def allocate_variants(
    archetype,
    count: int,
    contains_personal_data: bool,
    rng: random.Random,
) -> list[str]:
    """Allocate variants reproducibly with controlled blank frequency."""

    if count <= 0:
        return []

    variants = list(archetype.variants)

    blank_available = "blank" in variants

    filled_variants = [
        variant
        for variant in variants
        if variant != "blank"
    ]

    if not filled_variants:
        if contains_personal_data:
            raise ValueError(
                f"Scenario '{archetype.name}' has no non-blank "
                "variant available for PII-positive documents."
            )

        return ["blank"] * count

    # Positive rows always use populated variants.
    if contains_personal_data:
        blank_count = 0

    # Negative form-based rows may include a small share of blank templates.
    elif blank_available:
        blank_count = round(
            count * TARGET_BLANK_RATE
        )

    else:
        blank_count = 0

    filled_count = count - blank_count

    filled_ratios = {
        variant: 1 / len(filled_variants)
        for variant in filled_variants
    }

    filled_counts = allocate_counts(
        filled_count,
        filled_ratios,
    )

    allocated = []

    for variant, variant_count in filled_counts.items():
        allocated.extend(
            [variant] * variant_count
        )

    allocated.extend(
        ["blank"] * blank_count
    )

    rng.shuffle(allocated)

    return allocated


def allocate_languages(
    count: int,
    rng: random.Random,
) -> list[str]:
    """Allocate languages according to configured target ratios."""

    if count <= 0:
        return []

    language_counts = allocate_counts(
        count,
        TARGET_LANGUAGE_RATIOS,
    )

    languages = []

    for language, language_count in language_counts.items():
        languages.extend(
            [language] * language_count
        )

    rng.shuffle(languages)

    return languages


def allocate_entity_combinations(
    scenario: str,
    count: int,
) -> list[tuple[str, ...]]:
    """Allocate entity combinations evenly across positive documents."""

    if count <= 0:
        return []

    combinations = list(
        SCENARIO_ENTITY_COMBINATIONS[scenario]
    )

    if not combinations:
        raise ValueError(
            f"No entity combinations configured for "
            f"scenario '{scenario}'."
        )

    return [
        combinations[index % len(combinations)]
        for index in range(count)
    ]


def build_scenario_requests(
    scenario: str,
    n_documents: int = 100,
    seed: int = 42,
) -> list[GenerationRequest]:
    """Build a reproducible quota-aware request plan for one scenario."""

    counts = calculate_scenario_counts(
        n_documents
    )

    archetype = get_archetype(
        scenario
    )

    rng = random.Random(seed)


    requests = []

    all_entity_combinations = allocate_entity_combinations(
        scenario=scenario,
        count=counts["positive"],
    )

    entity_offset = 0

    for split, split_counts in counts["splits"].items():
        positive_count = split_counts["positive"]
        negative_count = split_counts["negative"]

        positive_languages = allocate_languages(
            positive_count,
            rng,
        )

        negative_languages = allocate_languages(
            negative_count,
            rng,
        )

        positive_variants = allocate_variants(
            archetype=archetype,
            count=positive_count,
            contains_personal_data=True,
            rng=rng,
        )

        negative_variants = allocate_variants(
            archetype=archetype,
            count=negative_count,
            contains_personal_data=False,
            rng=rng,
        )

        entity_combinations = all_entity_combinations[
            entity_offset:
            entity_offset + positive_count
        ]

        entity_offset += positive_count

        for index in range(
            positive_count
        ):
            request = GenerationRequest(
                scenario=scenario,
                language=positive_languages[index],
                contains_personal_data=True,
                entity_types=list(
                    entity_combinations[index]
                ),
                recommended_split=split,
                variant=positive_variants[index],
            )

            requests.append(
                request
            )

        for index in range(
            negative_count
        ):
            request = GenerationRequest(
                scenario=scenario,
                language=negative_languages[index],
                contains_personal_data=False,
                entity_types=[],
                recommended_split=split,
                variant=negative_variants[index],
            )

            requests.append(
                request
            )

    rng.shuffle(
        requests
    )

    return requests