"""
Sweep 1 PII detection.

Combines Microsoft Presidio and custom regular expressions to identify
strong and potential PII categories. Documents containing strong PII
are classified immediately. Ambiguous documents are routed to LLM review.
"""

import re
import warnings
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

warnings.filterwarnings("ignore")


# ── Set up Logging ────────────────────────────────────

import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Presidio Engine Initialization (done once)
# ─────────────────────────────────────────────────────────────

logger.info("Initializing Presidio NLP engine")
nlp_config = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_lg"},
        {"lang_code": "de", "model_name": "de_core_news_sm"},
    ],
}

provider = NlpEngineProvider(nlp_configuration=nlp_config)
nlp_engine = provider.create_engine()

engine = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["en", "de"],
)
logger.info("Presidio analyzer initialized for languages: en, de")

# ─────────────────────────────────────────────────────────────
# Category Definitions
# ─────────────────────────────────────────────────────────────

STRONG_PII = {
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "MEDICAL_LICENSE",
}

POTENTIAL_PII = {
    "PERSON",
    "EMAIL_ADDRESS",
}

MEANINGFUL_PII = STRONG_PII | POTENTIAL_PII

REGEX_BOOST = 0.3
REGEX_BASE_CONFIDENCE = 0.9

# ─────────────────────────────────────────────────────────────
# Regex Patterns
# ─────────────────────────────────────────────────────────────

_RE_EMAIL = re.compile(
    r"\b[a-zA-Z0-9._%+\-äöüÄÖÜß]+@[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}\b",
    re.IGNORECASE,
)

_RE_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}\b")

_RE_CREDIT_CARD = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

_RE_MEDICAL = re.compile(
    r"\b(?:Approbation(?:snummer)?|Arztnummer|Heilpraktiker|Medical\s+ID|Medizinische\s+Lizenz)\b",
    re.IGNORECASE,
)


def custom_detect(text):
    if not isinstance(text, str) or not text.strip():
        return {
            "custom_EMAIL_ADDRESS": False,
            "custom_IBAN_CODE": False,
            "custom_CREDIT_CARD": False,
            "custom_MEDICAL_LICENSE": False,
        }
    return {
        "custom_EMAIL_ADDRESS": bool(_RE_EMAIL.search(text)),
        "custom_IBAN_CODE": bool(_RE_IBAN.search(text)),
        "custom_CREDIT_CARD": bool(_RE_CREDIT_CARD.search(text)),
        "custom_MEDICAL_LICENSE": bool(_RE_MEDICAL.search(text)),
    }


# ─────────────────────────────────────────────────────────────
# Keyword Hint Detector
# Catches person-context language without a hard PII identifier.
# Documents matching these keywords are ambiguous and go to Sweep 2.
# Documents with no Presidio signal AND no keyword hint are
# definitively clean and skip the LLM entirely.
# ─────────────────────────────────────────────────────────────

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


def _has_person_hint(text: str) -> bool:
    """
    Returns True if the text contains keywords suggesting a person
    is described, even if no hard PII identifier was detected.
    Used to route borderline documents to LLM review.
    """
    if not isinstance(text, str):
        return False
    return bool(_RE_PERSON_HINT.search(text))


# ─────────────────────────────────────────────────────────────
# Fusion Layer
# ─────────────────────────────────────────────────────────────


def fuse_entities(text, language="en"):
    if not isinstance(text, str) or not text.strip():
        return {
            "entities": [],
            "per_type_conf": {},
        }

    raw = engine.analyze(text=text, language=language, score_threshold=0.5)

    entities = []
    per_type_scores = {}

    # Presidio entities
    for r in raw:
        etype = r.entity_type.upper()
        if etype not in MEANINGFUL_PII:
            continue

        value = text[r.start : r.end]
        conf = float(r.score)

        entities.append(
            {
                "type": etype,
                "value": value,
                "confidence": round(conf, 3),
                "source": "presidio",
                "start": r.start,
                "end": r.end,
            }
        )

        per_type_scores.setdefault(etype, []).append(conf)

    # Regex entities
    custom = custom_detect(text)
    regex_map = {
        "EMAIL_ADDRESS": custom["custom_EMAIL_ADDRESS"],
        "IBAN_CODE": custom["custom_IBAN_CODE"],
        "CREDIT_CARD": custom["custom_CREDIT_CARD"],
        "MEDICAL_LICENSE": custom["custom_MEDICAL_LICENSE"],
    }

    for etype, detected in regex_map.items():
        if not detected:
            continue

        if etype in per_type_scores:
            boosted = [min(1.0, s + REGEX_BOOST) for s in per_type_scores[etype]]
            per_type_scores[etype] = boosted
        else:
            per_type_scores.setdefault(etype, []).append(REGEX_BASE_CONFIDENCE)
            entities.append(
                {
                    "type": etype,
                    "value": "",
                    "confidence": REGEX_BASE_CONFIDENCE,
                    "source": "custom_regex",
                    "start": None,
                    "end": None,
                }
            )

    per_type_conf = {etype: max(scores) for etype, scores in per_type_scores.items()}

    return {
        "entities": entities,
        "per_type_conf": per_type_conf,
    }


# ─────────────────────────────────────────────────────────────
# Scan Text (Strong vs Potential PII)
# ─────────────────────────────────────────────────────────────


def scan_text(text, language="en"):
    base = {
        "detected_categories": [],
        "strong_pii_categories": [],
        "potential_pii_categories": [],
        "detected_any_pii": False,
        "detected_pii": False,
        "entities": [],
        "per_type_conf": {},
    }

    if not isinstance(text, str) or not text.strip():
        return base

    fusion = fuse_entities(text, language)

    per_type_conf = fusion["per_type_conf"]
    entities = fusion["entities"]

    strong_cats = []
    potential_cats = []

    for etype, conf in per_type_conf.items():
        if etype in STRONG_PII:
            strong_cats.append(etype)
        if etype in POTENTIAL_PII:
            potential_cats.append(etype)

    base.update(
        {
            "entities": entities,
            "per_type_conf": per_type_conf,
            "strong_pii_categories": strong_cats,
            "potential_pii_categories": potential_cats,
            "detected_pii": len(strong_cats) > 0,
            "detected_any_pii": len(strong_cats) > 0 or len(potential_cats) > 0,
            "detected_categories": sorted(per_type_conf.keys()),
        }
    )

    return base


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT FOR SWEEP 1
# ─────────────────────────────────────────────────────────────


def run_presidio_regex(df):
    """
    Sweep 1:
    - Runs Presidio + regex fusion
    - Classifies strong vs potential PII
    - Flags documents needing LLM review
    - Returns updated dataframe
    """

    logger.info("Starting Sweep 1 on %s documents", len(df))

    has_language_col = "language" in df.columns

    if has_language_col:
        logger.info("Language column found. Using row-level language values.")
    else:
        logger.warning("No language column found. Defaulting all rows to English.")

    scan_results = df.apply(
        lambda row: pd.Series(
            scan_text(
                text=row.get("full_text", "") or "",
                language=row["language"] if has_language_col else "en",
            )
        ),
        axis=1,
    )

    df = pd.concat([df, scan_results], axis=1)

    # Routes ambiguous documents to LLM
    df["needs_llm_review"] = (
        ~df["detected_pii"]  # not already confirmed
    ) & (
        (
            df["potential_pii_categories"].apply(  # Presidio found a name
                lambda cats: len(cats) > 0
            )
        )
        | (df["full_text"].apply(_has_person_hint))  # or keyword hints present
    )

    logger.info(
        "Sweep 1 completed. Strong PII: %s/%s. LLM review needed: %s/%s.",
        int(df["detected_pii"].sum()),
        len(df),
        int(df["needs_llm_review"].sum()),
        len(df),
    )

    return df
