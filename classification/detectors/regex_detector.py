"""
Regex-based PII detection.

Responsibilities:
- Detect structured identifiers using regular expressions
- Detect contextual person-related hints
- Return normalized entity evidence

This module does not make routing decisions.
"""

import datetime
import re
from typing import Dict, List


REGEX_BASE_CONFIDENCE = 0.9

_RE_EMAIL = re.compile(
    r"\b[a-zA-Z0-9._%+\-äöüÄÖÜß]+@[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}\b",
    re.IGNORECASE,
)

_RE_IBAN = re.compile(
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"
)

_RE_CREDIT_CARD = re.compile(
    r"\b(?:\d[ -]*?){12,19}\b"
)

_RE_MEDICAL = re.compile(
    r"\b(?:Approbation(?:snummer)?|Arztnummer|Heilpraktiker|Medical\s+ID|Medizinische\s+Lizenz)\b",
    re.IGNORECASE,
)

_RE_PASSPORT = re.compile(
    r"\bP\d{8}\b"
)

_RE_URL = re.compile(
    r"\bhttps?://[^\s]+\b",
    re.IGNORECASE,
)

_RE_PHONE = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"(?:\+\d{1,3}\s?)?(?:\(0\)\s?|\(\d{2,5}\)\s?|0)\d[\d\s\-]{3,12}\d"
    r"|"
    r"(?:\+1[\s.\-]?)?\d{3}[.\-]\d{3}[.\-]\d{4}"
    r")"
    r"(?:\s?(?:x|ext\.?)\s?\d{2,6})?"
    r"(?!\d)"
)

_MIN_PHONE_DIGITS = 7
_MAX_PHONE_DIGITS = 15

_DATE_FORMATS = (
    "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d",
    "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y",
    "%m-%d-%Y", "%m/%d/%Y",
)


def _looks_like_date(value: str) -> bool:
    candidate = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            datetime.datetime.strptime(candidate, fmt)
            return True
        except ValueError:
            continue
    return False


def _extract_phone_entities(text: str) -> List:
    entities = []

    for match in _RE_PHONE.finditer(text):
        value = match.group()

        digit_count = sum(c.isdigit() for c in value)
        if not (_MIN_PHONE_DIGITS <= digit_count <= _MAX_PHONE_DIGITS):
            continue

        if _looks_like_date(value):
            continue

        entities.append(
            {
                "type": "PHONE_NUMBER",
                "value": value,
                "confidence": REGEX_BASE_CONFIDENCE,
                "source": "custom_regex",
                "start": match.start(),
                "end": match.end(),
            }
        )

    return entities


def _suppress_phone_matches_inside_iban(entities: List) -> List:
    """
    A phone-shaped digit run that falls entirely inside an already-detected
    IBAN span (e.g. the numeric tail of an IBAN) is not a real phone number.
    Drop any PHONE_NUMBER entity whose span overlaps an IBAN_CODE entity.
    """

    iban_spans = [
        (e["start"], e["end"])
        for e in entities
        if e["type"] == "IBAN_CODE"
    ]

    if not iban_spans:
        return entities

    filtered = []

    for entity in entities:

        if entity["type"] != "PHONE_NUMBER":
            filtered.append(entity)
            continue

        overlaps_iban = any(
            entity["start"] < iban_end and entity["end"] > iban_start
            for iban_start, iban_end in iban_spans
        )

        if not overlaps_iban:
            filtered.append(entity)

    return filtered


_NRP_VALUES = [
    "German", "French", "Polish", "Austrian", "Swiss", "Dutch",
    "Italian", "Portuguese", "Danish", "Swedish",
    "deutsch", "französisch", "polnisch", "österreichisch",
    "schweizerisch", "niederländisch", "italienisch",
    "portugiesisch", "dänisch", "schwedisch",
]

_RE_NRP = re.compile(
    r"\b(?:" + "|".join(re.escape(value) for value in _NRP_VALUES) + r")\b",
    re.IGNORECASE,
)

_RE_PERSON_HINT = re.compile(
    r"\b(?:"
    r"Name|Vorname|Nachname|Mitarbeiter|Employee|"
    r"Herr|Frau|Mr|Ms|Dr|"
    r"Adresse|Address|"
    r"Geburtsdatum|Geburtstag|geb\.|geboren|"
    r"Unterschrift|Signature|"
    r"Ausweis|Reisepass|Passport|"
    r"Führerschein|Fahrerlaubnis|"
    r"Personalnummer|Personal.?ID|Employee.?ID"
    r")\b",
    re.IGNORECASE,
)


def has_person_hint(text: str) -> bool:
    if not isinstance(text, str):
        return False

    return bool(_RE_PERSON_HINT.search(text))


def _extract_pattern_entities(
    text: str,
    pattern: re.Pattern,
    entity_type: str,
) -> List:
    entities = []

    for match in pattern.finditer(text):
        entities.append(
            {
                "type": entity_type,
                "value": match.group(),
                "confidence": REGEX_BASE_CONFIDENCE,
                "source": "custom_regex",
                "start": match.start(),
                "end": match.end(),
            }
        )

    return entities


def detect_regex(text: str) -> Dict:

    if not isinstance(text, str) or not text.strip():
        return {
            "entities": [],
            "has_person_hint": False,
        }

    entities = []

    entities.extend(
        _extract_pattern_entities(
            text,
            _RE_EMAIL,
            "EMAIL_ADDRESS",
        )
    )

    entities.extend(
        _extract_pattern_entities(
            text,
            _RE_IBAN,
            "IBAN_CODE",
        )
    )

    entities.extend(
        _extract_pattern_entities(
            text,
            _RE_CREDIT_CARD,
            "CREDIT_CARD",
        )
    )

    entities.extend(
        _extract_pattern_entities(
            text,
            _RE_MEDICAL,
            "MEDICAL_LICENSE",
        )
    )

    entities.extend(
        _extract_pattern_entities(
            text,
            _RE_PASSPORT,
            "PASSPORT",
        )
    )

    entities.extend(
        _extract_pattern_entities(
            text,
            _RE_URL,
            "URL",
        )
    )

    entities.extend(
        _extract_phone_entities(
            text,
        )
    )

    entities.extend(
        _extract_pattern_entities(
            text,
            _RE_NRP,
            "NRP",
        )
    )

    entities = _suppress_phone_matches_inside_iban(entities)

    return {
        "entities": entities,
        "has_person_hint": has_person_hint(text),
    }