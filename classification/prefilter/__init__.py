"""
Transformer pre-filter for the GDPR PII detection pipeline.

The pre-filter sits between Sweep 1 (Presidio + spaCy + regex) and the LLM
review stage. Sweep 1 resolves the clear cases locally; the ambiguous ones
would otherwise all become LLM calls. This module trains a small transformer
encoder to resolve as many of those as possible locally, and routes only the
documents it is genuinely unsure about to the LLM.

The success criterion is not maximum accuracy. It is:

    how far can the LLM call volume drop while document-level recall
    stays at or above the rule-based baseline (~0.98)?

Public entry points (all runnable as ``python -m classification.prefilter.<x>``):

    eda           dataset report: class balance, entity labels, split sanity
    train         fine-tune the dual-head encoder
    thresholds    calibrate the three-zone router on the validation split
    predict       write an evaluation-compatible prediction CSV
    error_report  slice errors by document_type / difficulty / challenge
"""

from classification.prefilter.config import (
    ENTITY_LABELS,
    PreFilterConfig,
)

__all__ = [
    "ENTITY_LABELS",
    "PreFilterConfig",
]
