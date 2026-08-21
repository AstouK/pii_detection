"""
Sweep 1 execution.

Responsibilities:
- Run regex detection
- Run Presidio detection
- Fuse detector evidence
- Apply initial routing
- Produce deterministic classification output

This module implements the local rule-based stage of the pipeline.
"""

import logging

import pandas as pd

from classification.detectors.regex_detector import detect_regex
from classification.detectors.presidio_detector import detect_presidio
from classification.detectors.evidence_fusion import (
    fuse_detection_results,
)
from classification.review.routing import (
    determine_initial_route,
)

logger = logging.getLogger(__name__)


def classify_document_sweep1(
    text: str,
    language: str,
) -> dict:

    regex_result = detect_regex(text)

    presidio_result = detect_presidio(
        text=text,
        language=language,
    )

    fused_result = fuse_detection_results(
        regex_result=regex_result,
        presidio_result=presidio_result,
    )

    route_result = determine_initial_route(
        fused_result
    )

    return {
        **fused_result,
        **route_result,
    }


def run_sweep1(
    df: pd.DataFrame,
) -> pd.DataFrame:

    has_language_col = "language" in df.columns

    results = df.apply(
        lambda row: pd.Series(
            classify_document_sweep1(
                text=row.get("full_text", "") or "",
                language=row.get("language", "en")
                if has_language_col
                else "en",
            )
        ),
        axis=1,
    )

    return pd.concat(
        [df, results],
        axis=1,
    )