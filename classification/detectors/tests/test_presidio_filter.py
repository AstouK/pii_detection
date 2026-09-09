"""
Checks for the reference-id guard in the Presidio detector.

Presidio scores the digit tail of codes like ``REF-4324364`` as a phone number,
which produced false positives (SYN-EMPLOYEE_RECORD-0020,
SYN-SUPPLIER_ONBOARDING-0120/0221). The guard drops any PHONE_NUMBER whose span
directly follows a ``<LETTERS>-`` prefix.

Run:

    python -m pytest classification/detectors/tests/ -q
"""

from __future__ import annotations

from classification.detectors.presidio_detector import (
    _looks_like_reference_id,
    detect_presidio,
)


# ─────────────────────────────────────────────────────────────
# Unit: the pure prefix check
# ─────────────────────────────────────────────────────────────

def test_reference_prefix_positive_cases():
    # The digit run starts right after "REF-".
    text = "Employee Id: REF-4324364"
    start = text.index("4324364")
    assert _looks_like_reference_id(text, start) is True

    text = "Code EMP-42 assigned"
    start = text.index("42")
    assert _looks_like_reference_id(text, start) is True


def test_reference_prefix_negative_cases():
    # A real phone number with no letter-hyphen prefix must not be filtered.
    text = "Tel: 1234567"
    start = text.index("1234567")
    assert _looks_like_reference_id(text, start) is False

    text = "Contact +49 30 12345678 for details"
    start = text.index("49")
    assert _looks_like_reference_id(text, start) is False


# ─────────────────────────────────────────────────────────────
# Integration: the guard removes the FP end to end
# ─────────────────────────────────────────────────────────────

def test_detect_presidio_drops_reference_phone():
    text = (
        "Employee Record\n"
        "Employee Id: REF-4324364\n"
        "License Number: REF-1649446\n"
        "Identification Number: REF-4568871"
    )

    result = detect_presidio(text=text, language="en")

    phone_entities = [
        entity for entity in result["entities"]
        if entity["type"] == "PHONE_NUMBER"
    ]

    assert phone_entities == []
