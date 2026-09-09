"""
Checks for the relaxed fusion rule.

Previously LOCATION/DATE_TIME evidence without a co-occurring PERSON was stripped
from the potential categories, sending address-only documents to local_non_pii
and costing recall. It is now retained so those documents enter the ambiguous
route and are decided by the pre-filter. Strong-PII behaviour is unchanged.
"""

from __future__ import annotations

from classification.detectors.evidence_fusion import fuse_detection_results


def _entity(entity_type: str, value: str, start: int, confidence: float = 0.85):
    return {
        "type": entity_type,
        "value": value,
        "confidence": confidence,
        "source": "presidio",
        "start": start,
        "end": start + len(value),
    }


def _regex_result(entities=None, has_person_hint=False):
    return {"entities": entities or [], "has_person_hint": has_person_hint}


def _presidio_result(entities=None):
    return {"entities": entities or []}


def test_location_and_date_without_person_are_kept_as_potential():
    # An address plus a nearby date, no PERSON anywhere.
    presidio = _presidio_result(
        [
            _entity("LOCATION", "Loechelweg 308, 31722 Arnstadt", 10),
            _entity("DATE_TIME", "2025-07-19", 60),
        ]
    )

    fused = fuse_detection_results(
        regex_result=_regex_result(),
        presidio_result=presidio,
    )

    assert fused["detected_pii"] is False
    assert set(fused["potential_pii_categories"]) == {"LOCATION", "DATE_TIME"}
    # detected_any_pii drives the ambiguous route via needs_llm_review.
    assert fused["detected_any_pii"] is True


def test_strong_pii_detection_unchanged():
    # A strong identifier (EMAIL_ADDRESS) plus a location, still no PERSON.
    presidio = _presidio_result(
        [
            _entity("EMAIL_ADDRESS", "a@b.com", 5, confidence=0.95),
            _entity("LOCATION", "Arnstadt", 40),
        ]
    )

    fused = fuse_detection_results(
        regex_result=_regex_result(),
        presidio_result=presidio,
    )

    assert fused["detected_pii"] is True
    assert "EMAIL_ADDRESS" in fused["strong_pii_categories"]
    # LOCATION is still carried as potential evidence.
    assert "LOCATION" in fused["potential_pii_categories"]
