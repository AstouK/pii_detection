"""
Sweep 2 LLM review.

Reviews ambiguous documents identified by Sweep 1 using an LLM.
The goal is to reduce false positives while preserving recall.
"""

import re
import time
import requests
import json
import textwrap
import pandas as pd

# ── Set up Logging ────────────────────────────────────

import logging

logger = logging.getLogger(__name__)

# ── Set up Providers ────────────────────────────────────

# Compliance note:
# OpenRouter may route requests through infrastructure that does not guarantee
# EU data residency. For production use with real personal data, replace this
# with a GDPR-compliant provider such as Azure OpenAI in an EU region, Mistral
# with appropriate data processing terms, or another approved enterprise provider.

SUPPORTED_PROVIDERS = {
    "openrouter",
    "qwen",
}

# Qwen
from openai import OpenAI

from config.settings import (
    QWEN_API_KEY,
    QWEN_MODEL,
)

client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# OpenRouter (gpt-4o-mini)
from config.settings import (
    OPENROUTER_API_URL,
    OPENROUTER_MODEL,
    require_openrouter_api_key,
)

_RETRY_DELAYS = [2, 5, 20]  # Waiting time in seconds in case of lagging request

def create_llm_result(
    contains_pii: bool = False,
    reason: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    request_cost: float | None = None,
    success: bool = True,
) -> dict:
    """
    Create a standardized provider result.

    All LLM providers must return this structure.
    """

    return {
        "contains_pii": contains_pii,
        "reason": reason,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "reasoning_tokens": int(reasoning_tokens or 0),
        "cached_tokens": int(cached_tokens or 0),
        "request_cost": request_cost,
        "success": success,
    }
# ─────────────────────────────────────────────────────────────
# 1. Context-aware text extraction for LLM
# ─────────────────────────────────────────────────────────────


def extract_llm_text(text: str, entities: list, window: int = 200) -> str:
    """
    Build a minimal text snippet for the LLM by extracting context windows
    around each detected entity.

    Handles two cases:
      - Presidio entities: have start/end offsets → extract surrounding window
      - Regex-only entities: start=None/end=None → fall back to full text
        (these were detected without offsets so we must pass the whole document)

    Returns the original text unchanged if no offset-bearing entities exist,
    ensuring the LLM always receives something to classify.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if not entities:
        return text  # no entity hints at all → send full text

    # Separate entities with and without offsets
    offset_entities = [e for e in entities if e.get("start") is not None and e.get("end") is not None]

    # If all detections came from regex (no offsets), send full text
    if not offset_entities:
        return text

    # Build merged context windows from offset entities
    spans = [(max(0, e["start"] - window), min(len(text), e["end"] + window)) for e in offset_entities]
    spans.sort()

    merged = [spans[0]]
    for curr_start, curr_end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if curr_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append((curr_start, curr_end))

    return "\n\n---\n\n".join(text[s:e].strip() for s, e in merged)


# ─────────────────────────────────────────────────────────────
# 2. Prompt
# ─────────────────────────────────────────────────────────────


def build_llm_prompt(doc_text: str) -> str:
    return textwrap.dedent(f"""
        You are a data protection assistant specialised in GDPR compliance.

        Determine whether the following text contains personal data as defined under
        GDPR Article 4. Personal data includes any information relating to an
        identified or identifiable natural person, such as:

        - Real names of individuals
        - Email addresses
        - Phone numbers
        - Passport numbers, ID card numbers, or other government-issued identifiers
        - Financial identifiers (IBAN, credit card numbers)
        - Medical or professional licence numbers
        - Any information that directly identifies a person
        - Any information that indirectly identifies a person when combined with context
          (e.g., a unique employee ID used only for one person)

        Important: The following are NOT personal data and must NOT be flagged:
        - Company VAT numbers or Tax IDs (e.g., DE + 9 digits)
        - Generic invoice numbers, order numbers, tracking numbers, ticket numbers
        - Random alphanumeric strings without personal context
        - Product IDs, system IDs, database keys, UUIDs, hashes
        - Dates, times, or locations without a link to a specific person
        - Job titles or roles without a named or identifiable person

        Passport detection rule:
        - Only classify a passport or ID number as personal data if the text clearly
          indicates it belongs to a person (e.g., "passport", "Reisepass", "ID number",
          "Ausweis", "Passnummer"). Do NOT treat standalone 8–9 digit numbers as PII.

        Your task:
        - Be conservative with false positives.
        - Only return true if the text clearly contains personal data.
        - If uncertain, return false.

        Respond ONLY with valid JSON, no preamble, no markdown:

        {{
          "contains_pii": true or false,
          "reason": "one sentence explanation"
        }}

        Text:
        \"\"\"{doc_text}\"\"\"
    """).strip()


# ─────────────────────────────────────────────────────────────
# 3. API Provider calls (with retry on rate limit)
# ─────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def call_qwen(prompt: str) -> dict:
    """
    Send prompt to Qwen via DashScope.

    Returns:
        (contains_pii, reason)
    """

    logger.debug("Sending request to Qwen")
    logger.debug("Model: %s", QWEN_MODEL)

    delays = [0] + _RETRY_DELAYS

    for attempt, delay in enumerate(delays):
        if delay:
            logger.warning(
                "Rate limit received. Waiting %s seconds before retry %s/%s.",
                delay,
                attempt,
                len(_RETRY_DELAYS),
            )
            time.sleep(delay)

        try:
            completion = client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )

            content = completion.choices[0].message.content

            logger.debug(
                "Qwen response preview: %s",
                content[:300],
            )

            usage = completion.usage

            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0

            completion_details = getattr(
                usage,
                "completion_tokens_details",
                None,
            )

            reasoning_tokens = (
                getattr(completion_details, "reasoning_tokens", 0) or 0
                if completion_details
                else 0
            )

            prompt_details = getattr(
                usage,
                "prompt_tokens_details",
                None,
            )

            cached_tokens = (
                getattr(prompt_details, "cached_tokens", 0) or 0
                if prompt_details
                else 0
            )

            break

        except Exception as e:
            error_msg = str(e).lower()

            if "429" in error_msg:
                if attempt < len(_RETRY_DELAYS):
                    continue

                return create_llm_result(
                    reason=f"Rate limit exceeded after {len(_RETRY_DELAYS)} retries",
                    success=False,
                )

            logger.error("Qwen request failed: %s", e)

            return create_llm_result(
                reason=f"Request failed: {e}",
                success=False,
            )

    else:
        return create_llm_result(
            reason="All retries exhausted",
            success=False,
        )
    

    content_fixed = re.sub(
        r'"contains_pii"\s*:\s*,',
        '"contains_pii": false,',
        content,
    )

    try:
        parsed = json.loads(content_fixed)

    except json.JSONDecodeError:
        match = _JSON_RE.search(content_fixed)

        if match:
            try:
                parsed = json.loads(match.group())

            except json.JSONDecodeError:
                return create_llm_result(
                    reason=f"JSON parse error. Raw response: {content[:300]}",
                    success=False,
                )

        else:
            return create_llm_result(
                reason=f"No JSON found in response: {content[:300]}",
                success=False,
            )

    raw_val = parsed.get("contains_pii", False)

    if isinstance(raw_val, str):
        contains_pii = raw_val.strip().lower() == "true"
    else:
        contains_pii = bool(raw_val)

    reason = parsed.get("reason", "")

    return create_llm_result(
        contains_pii=contains_pii,
        reason=reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        request_cost=None,
    )


def call_openrouter(prompt: str) -> dict:
    """
    Call the configured model via OpenRouter.

    Returns:
        tuple[bool, str]: A tuple containing:
            - contains_pii: whether the model detected personal data
            - reason: one-sentence explanation or error message

    Retries up to 3 times on rate limits.
    On unrecoverable failure, returns (False, <error description>).
    """

    api_key = require_openrouter_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/gdpr-scanner",
        "X-Title": "GDPR PII Scanner",
    }

    logger.debug("Sending request to OpenRouter")
    logger.debug("Model: %s", OPENROUTER_MODEL)
    logger.debug("Prompt preview: %s ...", prompt[:300].replace("\n", " "))

    delays = [0] + _RETRY_DELAYS

    for attempt, delay in enumerate(delays):
        if delay:
            logger.warning(
                "Rate limit received. Waiting %s seconds before retry %s/%s.",
                delay,
                attempt,
                len(_RETRY_DELAYS),
            )
            time.sleep(delay)

        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
                timeout=120,
            )

            if response.status_code == 429:
                if attempt < len(_RETRY_DELAYS):
                    continue

                return create_llm_result(
                    reason=f"Rate limit exceeded after {len(_RETRY_DELAYS)} retries",
                    success=False,
                )

            response.raise_for_status()

            logger.debug("OpenRouter HTTP status: %s", response.status_code)

            api_response = response.json()

            # Cost Analysis
            usage = api_response.get("usage", {})

            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0
            total_tokens = usage.get("total_tokens", 0) or 0
            request_cost = usage.get("cost")

            completion_details = usage.get(
                "completion_tokens_details",
                {},
            ) or {}

            reasoning_tokens = completion_details.get(
                "reasoning_tokens",
                0,
            ) or 0

            prompt_details = usage.get(
                "prompt_tokens_details",
                {},
            ) or {}

            cached_tokens = prompt_details.get(
                "cached_tokens",
                0,
            ) or 0

            msg = api_response["choices"][0]["message"]["content"]

            if isinstance(msg, list):
                msg = msg[0].get("text", "")

            content = msg
            logger.debug("OpenRouter response preview: %s", content[:300])

            break

        except requests.HTTPError as e:
            return create_llm_result(
                reason=f"HTTP error {e.response.status_code}: {e.response.text[:200]}",
                success=False,
            )

        except requests.RequestException as e:
            return create_llm_result(
                reason=f"Request failed: {e}",
                success=False,
            )

        except (KeyError, IndexError) as e:
            return create_llm_result(
                reason=f"Unexpected API response structure: {e}",
                success=False,
            )

    else:
        return create_llm_result(
            reason="All retries exhausted",
            success=False,
        )

    content_fixed = re.sub(
        r'"contains_pii"\s*:\s*,',
        '"contains_pii": false,',
        content,
    )

    try:
        parsed = json.loads(content_fixed)

    except json.JSONDecodeError:
        match = _JSON_RE.search(content_fixed)

        if match:
            try:
                parsed = json.loads(match.group())

            except json.JSONDecodeError:
                return create_llm_result(
                    reason=f"JSON parse error. Raw response: {content[:300]}",
                    success=False,
                )

        else:
            return create_llm_result(
                reason=f"No JSON found in response: {content[:300]}",
                success=False,
            )

    raw_val = parsed.get("contains_pii", False)

    if isinstance(raw_val, str):
        contains_pii = raw_val.strip().lower() == "true"
    else:
        contains_pii = bool(raw_val)

    reason = parsed.get("reason", "")

    return create_llm_result(
        contains_pii=contains_pii,
        reason=reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        request_cost=request_cost,
    )


def call_llm(
    prompt: str,
    provider: str,
) -> dict:
    """
    Route the prompt to the selected LLM provider.

    Args:
        prompt: Prompt sent to the LLM.
        provider: Provider name. Supported values:
            - "openrouter"
            - "qwen"

    Returns:
        tuple[bool, str]:
            - contains_pii
            - reason
    """

    provider = provider.lower().strip()

    if provider == "openrouter":
        return call_openrouter(prompt)

    if provider == "qwen":
        return call_qwen(prompt)

    raise ValueError(f"Unsupported LLM provider: {provider}. Expected 'openrouter' or 'qwen'.")


# ─────────────────────────────────────────────────────────────
# 4. MAIN ENTRY POINT FOR SWEEP 2
# ─────────────────────────────────────────────────────────────


def run_llm(
    df: pd.DataFrame,
    provider: str,
) -> pd.DataFrame:
    """
    Sweep 2: LLM review for documents flagged by Sweep 1.

    Args:
        df: DataFrame containing Sweep 1 results.
        provider: LLM provider to use. Supported values include:
            - "openrouter"
            - "qwen"

    Adds columns:
        llm_pii: bool
            LLM verdict.
        llm_reason: str
            LLM explanation or error message.
        llm_provider: str
            Provider used for the LLM review.

    Does NOT compute final_pii.
    That remains the pipeline's responsibility.
    """

    provider = provider.lower().strip()

    df["llm_pii"] = False
    df["llm_reason"] = ""
    df["llm_provider"] = provider
    df["llm_prompt_tokens"] = 0
    df["llm_completion_tokens"] = 0
    df["llm_total_tokens"] = 0
    df["llm_reasoning_tokens"] = 0
    df["llm_cached_tokens"] = 0
    df["llm_request_cost"] = 0.0
    df["llm_request_success"] = False

    flagged = df[df["needs_llm_review"]]

    if flagged.empty:
        logger.info(
            "No documents require LLM review for provider '%s'",
            provider,
        )
        return df

    logger.info(
        "Running LLM review with provider '%s' on %s flagged documents",
        provider,
        len(flagged),
    )

    total = len(flagged)

    for i, (idx, row) in enumerate(flagged.iterrows(), start=1):
        if i % 10 == 0 or i == total:
            logger.info(
                "LLM review progress for provider '%s': %s/%s documents",
                provider,
                i,
                total,
            )

        text = extract_llm_text(
            text=str(row.get("full_text", "") or ""),
            entities=row.get("entities", []) or [],
        )

        if not text.strip():
            logger.warning(
                "Skipping row %s for provider '%s' because extracted text is empty",
                idx,
                provider,
            )
            df.at[idx, "llm_reason"] = "empty text after extraction"
            continue

        prompt = build_llm_prompt(text)

        result = call_llm(
            prompt=prompt,
            provider=provider,
        )

        df.at[idx, "llm_pii"] = result["contains_pii"]
        df.at[idx, "llm_reason"] = result["reason"]

        df.at[idx, "llm_prompt_tokens"] = result["prompt_tokens"]
        df.at[idx, "llm_completion_tokens"] = result["completion_tokens"]
        df.at[idx, "llm_total_tokens"] = result["total_tokens"]
        df.at[idx, "llm_reasoning_tokens"] = result["reasoning_tokens"]
        df.at[idx, "llm_cached_tokens"] = result["cached_tokens"]
        df.at[idx, "llm_request_success"] = result["success"]

        if result["request_cost"] is not None:
            df.at[idx, "llm_request_cost"] = result["request_cost"]


    n_positive = df.loc[flagged.index, "llm_pii"].sum()

    logger.info(
        "Sweep 2 completed for provider '%s'. LLM flagged %s/%s documents as containing PII.",
        provider,
        int(n_positive),
        len(flagged),
    )

    logger.info(
        "Provider '%s' token usage: prompt=%s completion=%s total=%s",
        provider,
        int(df["llm_prompt_tokens"].sum()),
        int(df["llm_completion_tokens"].sum()),
        int(df["llm_total_tokens"].sum()),
    )
    
    return df
