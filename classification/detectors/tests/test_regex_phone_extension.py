"""
Checks for the phone-number digit cap in the regex detector.

The cap was raised from 15 to 20 so a number with a trailing extension
(``+1-805-872-3511x96849``, 16 digits, SYN-INTERNAL_EMAIL-0126) is captured,
while absurdly long digit runs are still rejected.
"""

from __future__ import annotations

from classification.detectors.regex_detector import detect_regex


def test_phone_with_extension_is_detected():
    text = "For any urgent concerns, you can reach me at +1-805-872-3511x96849."

    result = detect_regex(text)

    phone_values = [
        entity["value"]
        for entity in result["entities"]
        if entity["type"] == "PHONE_NUMBER"
    ]

    assert len(phone_values) == 1
    assert "96849" in phone_values[0]


def test_overlong_digit_run_is_rejected():
    # 25 digits: above the 20-digit cap, must not be treated as a phone number.
    text = "Ref 1234567890123456789012345 end"

    result = detect_regex(text)

    phone_entities = [
        entity for entity in result["entities"]
        if entity["type"] == "PHONE_NUMBER"
    ]

    assert phone_entities == []
