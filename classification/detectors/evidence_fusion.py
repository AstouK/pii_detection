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
    "PASSPORT",
}

POTENTIAL_PII = {
    "PERSON",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "LOCATION",
    "DATE_TIME",
    "NRP",
    "URL",
}

REGEX_BOOST = 0.3

DATE_TIME_CONTEXT_WINDOW = 200


def _near_any(entity: dict, spans: list) -> bool:
    for s_start, s_end in spans:
        if abs(s_start - entity["end"]) <= DATE_TIME_CONTEXT_WINDOW:
            return True
        if abs(entity["start"] - s_end) <= DATE_TIME_CONTEXT_WINDOW:
            return True
    return False


def _filter_weak_date_time_entities(entities: list) -> list:

    person_spans = [
        (e["start"], e["end"])
        for e in entities
        if e.get("type") == "PERSON"
    ]

    location_spans = [
        (e["start"], e["end"])
        for e in entities
        if e.get("type") == "LOCATION"
    ]

    filtered = []

    for entity in entities:

        if entity.get("type") != "DATE_TIME":
            filtered.append(entity)
            continue

        if _near_any(entity, person_spans) or _near_any(entity, location_spans):
            filtered.append(entity)

    return filtered


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

    # GDPR Art. 4(1): a bare date is not independently identifying. Only
    # keep a DATE_TIME detection when it sits near a PERSON entity (a
    # dated event tied to an identified individual) or near a LOCATION
    # entity (a specific event at a specific place). Boilerplate dates
    # (e.g. a template's "Effective Date" field) with no such context
    # are dropped here, before they ever enter per_type_conf.
    entities = _filter_weak_date_time_entities(entities)

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

    # GDPR Art. 4(1): a place or a date is only personal data if it is
    # tied to an identified individual. Previously, LOCATION and DATE_TIME
    # could protect each other from removal even with no PERSON anywhere
    # in the document - e.g. a stray place-noun misfire co-occurring with
    # a boilerplate date, neither tied to any person, would both survive.
    # That doesn't satisfy the "identified individual" requirement either
    # way. PERSON is now the sole anchor: without it, both are dropped.
    if "PERSON" not in per_type_conf:
        if "LOCATION" in potential_categories:
            potential_categories.remove("LOCATION")
        if "DATE_TIME" in potential_categories:
            potential_categories.remove("DATE_TIME")

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
