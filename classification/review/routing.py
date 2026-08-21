"""
Routing policy for hybrid classification.

Responsibilities:
- Decide whether documents are:
    - resolved locally
    - sent to LLM review
    - escalated in future versions

Current implementation reproduces the existing
needs_llm_review behavior.

Future versions may support:
- low-cost LLM routing
- advanced LLM escalation
- human review
"""

from typing import Dict

LOCAL_PII = "local_pii"
LOCAL_NON_PII = "local_non_pii"
LOW_COST_LLM = "low_cost_llm"


def determine_initial_route(
    fused_result: dict,
) -> dict:

    if fused_result["detected_pii"]:

        return {
            "route": LOCAL_PII,
            "routing_reason": "strong_pii_detected",
            "needs_llm_review": False,
        }

    has_potential_pii = (
        len(
            fused_result[
                "potential_pii_categories"
            ]
        ) > 0
    )

    has_hint = fused_result[
        "has_person_hint"
    ]

    if has_potential_pii or has_hint:

        return {
            "route": LOW_COST_LLM,
            "routing_reason": (
                "potential_pii_or_person_hint"
            ),
            "needs_llm_review": True,
        }

    return {
        "route": LOCAL_NON_PII,
        "routing_reason": (
            "no_meaningful_signal"
        ),
        "needs_llm_review": False,
    }