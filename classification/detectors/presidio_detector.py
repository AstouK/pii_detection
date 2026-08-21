"""
Microsoft Presidio detector.

Responsibilities:
- Initialize Presidio
- Execute named entity recognition
- Normalize Presidio results

This module produces entity evidence only.
"""

import logging

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
}


def _create_presidio_engine() -> AnalyzerEngine:

    logger.info("Initializing Presidio NLP engine")

    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "de", "model_name": "de_core_news_sm"},
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

        entities.append(
            {
                "type": entity_type,
                "value": text[item.start:item.end],
                "confidence": round(float(item.score), 3),
                "source": "presidio",
                "start": item.start,
                "end": item.end,
            }
        )

    return {
        "entities": entities,
    }