"""
Evidence fusion layer.

Responsibilities:
- Combine regex and Presidio evidence
- Normalize entity categories
- Aggregate confidence scores
- Produce a unified Sweep 1 result

This module does not make routing decisions.
"""

from collections import defaultdict

STRONG_PII = {
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "EMPLOYEE_ID",
    "USER_ID",
    "MEDICAL_LICENSE",
}

POTENTIAL_PII = {
    "PERSON",
    "ADDRESS",
    "DATE_OF_BIRTH",
}

REGEX_BOOST = 0.3


def fuse_detection_results(
    regex_result: dict,
    presidio_result: dict,
) -> dict:

    entities = []

    entities.extend(
        presidio_result["entities"]
    )

    entities.extend(
        regex_result["entities"]
    )

    per_type_scores = defaultdict(list)

    for entity in entities:

        per_type_scores[
            entity["type"]
        ].append(
            entity["confidence"]
        )

    regex_entities = {
        entity["type"]
        for entity in regex_result["entities"]
    }

    presidio_entities = {
        entity["type"]
        for entity in presidio_result["entities"]
    }

    agreements = regex_entities & presidio_entities

    for entity_type in agreements:

        boosted = [
            min(1.0, score + REGEX_BOOST)
            for score in per_type_scores[entity_type]
        ]

        per_type_scores[entity_type] = boosted

    per_type_conf = {
        entity_type: max(scores)
        for entity_type, scores
        in per_type_scores.items()
    }

    strong_categories = []
    potential_categories = []

    for entity_type in per_type_conf:

        if entity_type in STRONG_PII:
            strong_categories.append(entity_type)

        if entity_type in POTENTIAL_PII:
            potential_categories.append(entity_type)

    return {
        "entities": entities,
        "per_type_conf": per_type_conf,
        "detected_categories": sorted(
            per_type_conf.keys()
        ),
        "strong_pii_categories": strong_categories,
        "potential_pii_categories": potential_categories,
        "detected_pii": len(strong_categories) > 0,
        "detected_any_pii": (
            len(strong_categories) > 0
            or len(potential_categories) > 0
        ),
        "has_person_hint": regex_result[
            "has_person_hint"
        ],
    }
