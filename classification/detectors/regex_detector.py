"""
Regex-based PII detection.

Responsibilities:
- Detect structured identifiers using regular expressions
- Detect contextual person-related hints
- Return normalized entity evidence

This module does not make routing decisions.
"""

import re
from typing import Dict, List


REGEX_BASE_CONFIDENCE = 0.9

_RE_EMAIL = re.compile(
    r"\b[a-zA-Z0-9._%+\-äöüÄÖÜß]+@[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}\b",
    re.IGNORECASE,
)

_RE_IBAN = re.compile(
    r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}\b"
)

_RE_CREDIT_CARD = re.compile(
    r"\b(?:\d[ -]*?){13,16}\b"
)

_RE_MEDICAL = re.compile(
    r"\b(?:Approbation(?:snummer)?|Arztnummer|Heilpraktiker|Medical\s+ID|Medizinische\s+Lizenz)\b",
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

    return {
        "entities": entities,
        "has_person_hint": has_person_hint(text),
    }