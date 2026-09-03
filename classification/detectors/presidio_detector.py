"""
Microsoft Presidio detector.

Responsibilities:
- Initialize Presidio
- Execute named entity recognition
- Normalize Presidio results

This module produces entity evidence only.
"""

import logging
import re

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

logger = logging.getLogger(__name__)

MEANINGFUL_PII = {
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "EMPLOYEE_ID",
    "USER_ID",
    "MEDICAL_LICENSE",
    "PERSON",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "LOCATION",
    "DATE_TIME",
    "URL",
    "NRP",
}

_PERSON_FALSE_POSITIVE_DENYLIST = {
    "betreff",
    "unterschrift",
    "standardmäßiger",
}

_PERSON_MIN_TOKEN_COUNT = 2

_LOCATION_FALSE_POSITIVE_DENYLIST = {
    "kundenorganisation",
    "arbeitsplatz",
    "patientenangaben",
    "anbieterorganisation",
    "zentrale kontaktstelle",
    "support-fachbereich",
    "antragstellerangaben",
    "standardmäßigen",
    "compliance-abteilung",
    "lagerbereich",
    "teilnehmerangaben",
    "projektstatus-update",
    "supplier location",
    "department",
    "service desk",
    "regional office",
    "client site",
    "training center",
    "head office",
}

_LOCATION_GENERIC_SUFFIXES = (
    "organisation",
    "abteilung",
    "bereich",
    "fachbereich",
    "stelle",
    "angaben",
    "office",
    "area",
    "center",
    "department",
)

_RE_LOCATION_DEPT_CODE = re.compile(r"^[A-Z]{2,6}-\d+$")
_RE_LOCATION_QUARTER = re.compile(r"^Q[1-4]$")

_LOCATION_TITLE_TOKENS = {"univ.", "prof.", "univ.prof."}


def _is_generic_location_fp(value: str) -> bool:

    normalized = value.strip().lower()
    first_line = normalized.splitlines()[0].strip() if normalized else ""

    if "@" in normalized:
        return True

    if _RE_LOCATION_DEPT_CODE.match(value.strip()):
        return True

    if _RE_LOCATION_QUARTER.match(value.strip()):
        return True

    if len(first_line.replace(".", "")) <= 2:
        return True

    if any(first_line.startswith(t) for t in _LOCATION_TITLE_TOKENS):
        return True

    if first_line in _LOCATION_FALSE_POSITIVE_DENYLIST:
        return True

    for suffix in _LOCATION_GENERIC_SUFFIXES:
        if first_line.endswith(suffix):
            return True

    return False


def _create_presidio_engine() -> AnalyzerEngine:

    logger.info("Initializing Presidio NLP engine")

    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "de", "model_name": "de_core_news_md"},
        ],
    }

    provider = NlpEngineProvider(
        nlp_configuration=nlp_config
    )

    nlp_engine = provider.create_engine()

    return AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en", "de"],
    )


ENGINE = _create_presidio_engine()


def detect_presidio(
    text: str,
    language: str = "en",
) -> dict:

    if not isinstance(text, str) or not text.strip():
        return {"entities": []}

    raw_entities = ENGINE.analyze(
        text=text,
        language=language,
        score_threshold=0.5,
    )

    entities = []

    for item in raw_entities:

        entity_type = item.entity_type.upper()

        if entity_type not in MEANINGFUL_PII:
            continue

        value = text[item.start:item.end]
        stripped_value = value.strip()

        if entity_type == "PERSON":
            if stripped_value.lower() in _PERSON_FALSE_POSITIVE_DENYLIST:
                continue
            if len(stripped_value.split()) < _PERSON_MIN_TOKEN_COUNT:
                continue

        if entity_type == "LOCATION":
            if _is_generic_location_fp(stripped_value):
                continue

        entities.append(
            {
                "type": entity_type,
                "value": value,
                "confidence": round(float(item.score), 3),
                "source": "presidio",
                "start": item.start,
                "end": item.end,
            }
        )

    return {
        "entities": entities,
    }