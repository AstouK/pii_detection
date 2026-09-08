"""Constrained LLM enrichment for synthetic document generation."""

import json
import os
import re
import time

import requests


DEFAULT_MODEL = "openai/gpt-4o-mini"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


LLM_FREE_TEXT_FIELDS = {
    "contract": {
        "subject",
        "terms",
        "termination",
    },
    "customer_support": {
        "subject",
        "description",
        "resolution",
    },
    "employee_record": {
        "notes",
    },
    "expense_report": {
        "description",
        "summary",
    },
    "incident_report": {
        "description",
        "root_cause",
        "corrective_action",
    },
    "invoice": {
        "line_items",
    },
    "it_access_request": {
        "justification",
        "comments",
    },
    "medical_record": {
        "reason",
        "assessment",
        "treatment",
        "follow_up",
        "notes",
    },
    "meeting_notes": {
        "title",
        "agenda",
        "discussion",
        "decisions",
        "action_items",
    },
    "supplier_onboarding": {
        "comments",
        "notes",
    },
    "training_evaluation": {
        "content",
        "comments",
        "recommendation",
    },
}

VARIANT_STYLE_INSTRUCTIONS = {
    "contract": {
        "contract_excerpt": (
            "Use concise contract language focused on the core agreement terms."
        ),
        "signature_section": (
            "Emphasize parties, approval, signature context, and execution details."
        ),
        "amendment": (
            "Write as a contract amendment describing changes to an existing agreement."
        ),
    },

    "customer_support": {
        "portal_ticket": (
            "Write like a structured customer support portal ticket."
        ),
        "email_ticket": (
            "Use wording that reflects a support request submitted by email."
        ),
        "escalation_note": (
            "Emphasize escalation context, impact, investigation, and next steps."
        ),
    },

    "employee_record": {
        "structured_record": (
            "Keep the tone concise and record-like."
        ),
        "hr_profile": (
            "Use slightly more descriptive HR profile wording."
        ),
        "employee_summary": (
            "Use concise summary-style wording focused on relevant employment details."
        ),
    },

    "expense_report": {
        "filled_a": (
            "Use concise expense-report wording with practical reimbursement context."
        ),
        "filled_b": (
            "Use slightly more detailed wording and supporting business context."
        ),
    },

    "general_document": {
        "internal_memo": (
            "Write as an internal corporate memo."
        ),
        "project_update": (
            "Write as a concise project status update."
        ),
        "policy_note": (
            "Write as an internal policy or procedural note."
        ),
    },

    "incident_report": {
        "filled_a": (
            "Use concise incident-report wording focused on what happened and corrective action."
        ),
        "filled_b": (
            "Use more detailed investigative wording including cause and remediation context."
        ),
    },

    "internal_email": {
        "short_email": (
            "Write a short, natural internal business email."
        ),
        "thread_reply": (
            "Write as a reply in an existing internal email thread."
        ),
        "internal_announcement": (
            "Write as an internal announcement to a broader employee audience."
        ),
    },

    "invoice": {
        "standard_invoice": (
            "Use concise standard invoice wording."
        ),
        "service_invoice": (
            "Reflect services delivered rather than physical goods."
        ),
        "invoice_summary": (
            "Use a compact summary style while preserving invoice context."
        ),
    },

    "it_access_request": {
        "filled_a": (
            "Use concise access-request justification and approval wording."
        ),
        "filled_b": (
            "Use slightly more detailed business justification and review comments."
        ),
    },

    "medical_record": {
        "clinical_note": (
            "Use concise clinical documentation language."
        ),
        "appointment_summary": (
            "Use a shorter summary-oriented style covering the visit and follow-up."
        ),
        "occupational_health_note": (
            "Use workplace-related medical context appropriate for occupational health."
        ),
    },

    "meeting_notes": {
        "formal_minutes": (
            "Use formal meeting-minutes language with clear decisions and actions."
        ),
        "informal_notes": (
            "Use concise, slightly informal working-note style, including bullets where natural."
        ),
        "decision_log": (
            "Prioritize decisions, owners, and action items over detailed discussion."
        ),
    },

    "passport_record": {
        "identity_check": (
            "Use concise identity-verification wording."
        ),
        "travel_document_record": (
            "Use administrative wording focused on travel-document details."
        ),
        "verification_note": (
            "Use short verification-oriented wording."
        ),
    },

    "supplier_onboarding": {
        "filled_a": (
            "Use concise supplier-onboarding and compliance wording."
        ),
        "filled_b": (
            "Use slightly more detailed onboarding and review context."
        ),
    },

    "training_evaluation": {
        "filled_a": (
            "Use concise participant feedback and recommendations."
        ),
        "filled_b": (
            "Use more detailed evaluation wording with constructive recommendations."
        ),
    },
}

def build_rewrite_prompt(
    base_text: str,
    scenario: str,
    variant: str | None,
    language: str,
    required_values: list[str],
) -> str:
    """Build a constrained prompt for realistic synthetic text variation."""

    required_values_text = "\n".join(
        f"- {value}"
        for value in required_values
    )

    if not required_values_text:
        required_values_text = "- None"

    if required_values:
        label_specific_rules = """
This is a PII-positive document.

The required fictional values listed below are the ONLY personal data
or identifiers that may appear in the output.
""".strip()

    else:
        label_specific_rules = """
This is a PII-negative document.

Do NOT introduce any personal data or realistic identifier-like values.

In particular, do not invent:
- natural-person names
- email addresses
- phone numbers
- street addresses or postal addresses
- bank or card details
- passport or identity numbers
- employee IDs
- medical licence numbers
- IP addresses
- URLs

Use organizations, departments, generic roles, business processes,
and non-identifying operational information instead.

If the document normally contains a person-specific field such as
author, sender, recipient, patient, doctor, employee, manager,
participant, contact person, or responsible person, replace the
individual with a generic organizational role or department.

Examples:
- "Medical Team" instead of a doctor's name
- "Project Management" instead of a project manager's name
- "Customer Support Team" instead of an agent's name

Never use placeholder person names such as "John Doe", "Jane Doe",
"Max Mustermann", "Erika Mustermann", or similar fictional examples.
""".strip()

    variant_instruction = (
        VARIANT_STYLE_INSTRUCTIONS
        .get(scenario, {})
        .get(
            variant,
            "Use realistic wording appropriate for the document variant."
        )
    )

    return f"""
You are creating a realistic synthetic enterprise document
for a PII-detection dataset.

Rewrite the source document into a realistic document that could plausibly
occur in an organization.

Scenario: {scenario}
Variant: {variant or "default"}
Language: {language}

Variant style instruction:
{variant_instruction}

{label_specific_rules}

Rules:

1. Write only in the requested language.

2. Preserve every required fictional value EXACTLY as written.

3. The ONLY personal data or identifiers allowed in the output are the
   exact values listed under "Required fictional values".

   Do NOT derive, infer, or create related identifiers from those values.

   For example:
   - if a person's name is provided, do NOT invent their email address
   - do NOT invent their phone number
   - do NOT invent their home address
   - do NOT invent an employee ID or other identifier
   - if an email address is provided, do NOT invent a related phone number
   - if an address is provided, do NOT invent a person associated with it

   Do not introduce any additional:
   - natural-person names
   - personal email addresses
   - phone numbers
   - physical home addresses
   - bank or card details
   - passport or identity numbers
   - medical licence numbers
   - IP addresses
   - personal URLs
   - other identifying values

4. You MAY invent realistic NON-PERSONAL business context where useful,
   including:
   - company names
   - project names
   - project or process descriptions
   - service types
   - departments
   - business purposes
   - generic organizational roles
   - amounts
   - statuses
   - deadlines
   - issue descriptions
   - operational details

   If the source contains a contact or similar field, improve the surrounding
   wording but do NOT add contact information that is not explicitly listed
   under "Required fictional values".

5. Make the document meaningfully more realistic than the source.
   Avoid generic filler such as:
   "standard business information",
   "internal documentation",
   or equivalent vague wording.

6. Vary document structure, wording, headings, sentence length,
   and level of formality according to the scenario and variant.

7. Do not mention PII, GDPR, classification, labels, synthetic data,
   training data, prompts, or generation instructions.

8. Do not reproduce the BEGIN/END markers.

9. Before returning the document, proofread it for spacing,
   punctuation, grammar, and formatting errors.

10. Return ONLY the final document text.

Required fictional values:
{required_values_text}

---BEGIN SOURCE DOCUMENT---
{base_text}
---END SOURCE DOCUMENT---
""".strip()


def validate_required_values(
    text: str,
    required_values: list[str],
) -> None:
    """Check that all controlled values survived the rewrite."""

    missing = [
        value
        for value in required_values
        if value not in text
    ]

    if missing:
        raise ValueError(
            "LLM output dropped required values: "
            f"{missing}"
        )


EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

IP_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

URL_PATTERN = re.compile(
    r"https?://[^\s]+"
)


def validate_no_unexpected_identifiers(
    text: str,
    required_values: list[str],
) -> None:
    """Reject common identifiers introduced by the LLM."""

    allowed_text = "\n".join(required_values)

    checks = {
        "email": EMAIL_PATTERN.findall(text),
        "ip_address": IP_PATTERN.findall(text),
        "url": URL_PATTERN.findall(text),
    }

    unexpected = {}

    for entity_type, matches in checks.items():
        invalid_matches = [
            match
            for match in matches
            if match not in allowed_text
        ]

        if invalid_matches:
            unexpected[entity_type] = invalid_matches

    if unexpected:
        raise ValueError(
            "LLM introduced unexpected identifiers: "
            f"{unexpected}"
        )
        

def enrich_free_text_fields(
    scenario: str,
    variant: str | None,
    language: str,
    field_values: dict[str, str],
    required_values: list[str],
    model: str = DEFAULT_MODEL,
    max_attempts: int = 3,
) -> tuple[dict[str, str], bool]:
    """Generate realistic text only for designated free-text fields."""

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not available in the environment."
        )

    free_text_fields = LLM_FREE_TEXT_FIELDS.get(
        scenario,
        set(),
    )

    if not free_text_fields:
        return field_values, False

    fields_to_generate = [
        field
        for field in field_values
        if field in free_text_fields
    ]

    if not fields_to_generate:
        return field_values, False

    # Determine which controlled fictional values are already
    # assigned to each free-text field. These values must survive
    # the LLM enrichment in exactly the same field.
    protected_values_by_field = {
        field: [
            value
            for value in required_values
            if value in field_values.get(field, "")
        ]
        for field in fields_to_generate
    }

    required_values_text = "\n".join(
        f"- {value}"
        for value in required_values
    )

    if not required_values_text:
        required_values_text = "- None"

    structured_context = "\n".join(
        f"{field}: {value}"
        for field, value in field_values.items()
        if field not in free_text_fields
    )

    target_fields = "\n".join(
        f"- {field}"
        for field in fields_to_generate
    )

    existing_target_values = "\n".join(
        f"{field}: "
        + (
            field_values[field]
            if protected_values_by_field[field]
            else "[generate new content]"
            )
    for field in fields_to_generate
    )

    protected_field_values = "\n".join(
        f"{field}: "
        + (
            "; ".join(values)
            if values
            else "None"
        )
        for field, values in protected_values_by_field.items()
    )

    variant_instruction = (
        VARIANT_STYLE_INSTRUCTIONS
        .get(scenario, {})
        .get(
            variant,
            "Use realistic wording appropriate for the document variant."
        )
    )

    prompt = f"""
You are generating realistic free-text content for a synthetic
enterprise document used in a controlled PII-detection dataset.

Scenario: {scenario}
Variant: {variant or "default"}
Language: {language}
Variant style instruction:
{variant_instruction}

The document structure and all structured values have already been fixed.

Your task is ONLY to generate content for these fields:

{target_fields}

Existing values in those fields:

{existing_target_values}

Structured document context:

{structured_context}

Required fictional values:

{required_values_text}

Protected values that MUST remain in each field:

{protected_field_values}

Rules:

1. Write only in the requested language.

2. Do not alter, replace, or reinterpret any structured field values.

3. The ONLY personal data or identifiers allowed are the exact values
   listed under "Required fictional values".

4. If a target field has protected values, every protected value listed
   for that field MUST appear character-for-character in the generated
   value for that same field.

   Write natural surrounding text around the protected value where
   appropriate.

   Do not modify the protected value.

   Do not move the protected value to another field.

5. Do not invent additional:
   - natural-person names
   - email addresses
   - phone numbers
   - postal addresses
   - bank or card details
   - passport or identity numbers
   - employee IDs
   - medical licence numbers
   - IP addresses
   - URLs
   - other identifying values

6. Generate realistic, scenario-appropriate business content.

7. The generated fields must be mutually consistent with the
   structured context.

   For example, amounts, dates, statuses, departments, categories,
   systems, and decisions mentioned in the generated text must not
   contradict the structured fields.

8. Do not invent unnecessary specific facts when they are not supported
   by the structured context. Add enough operational detail to make the
   document realistic, but avoid introducing unrelated factual details.

9. Respect the document variant when deciding tone, detail,
   and writing style.

10. Avoid generic filler such as:
    "standard business information"
    or
    "internal documentation for the standard business process".

11. Before returning the result, proofread spacing, punctuation,
    grammar, and formatting.

12. Return ONLY a valid JSON object.

    The JSON object must contain exactly these keys:

{target_fields}

    Every value must be a string.

    Do not include Markdown fences, explanations, comments,
    or additional keys.
""".strip()


    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key.strip()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "temperature": 0.7,
                },
                timeout=(10, 90),
            )

            response.raise_for_status()

            data = response.json()

            raw_text = (
                data["choices"][0]["message"]["content"]
                .strip()
            )

            generated_fields = json.loads(
                raw_text
            )

            if not isinstance(generated_fields, dict):
                raise ValueError(
                    "LLM output is not a JSON object."
                )

            if set(generated_fields) != set(fields_to_generate):
                raise ValueError(
                    "LLM returned unexpected field keys. "
                    f"Expected {sorted(fields_to_generate)}, "
                    f"got {sorted(generated_fields)}."
                )

            for field, value in generated_fields.items():
                if not isinstance(value, str):
                    raise ValueError(
                        f"Generated field '{field}' is not a string."
                    )

                if not value.strip():
                    raise ValueError(
                        f"Generated field '{field}' is empty."
                    )

                # Any controlled value originally assigned to this
                # particular field must remain exactly in that field.
                validate_required_values(
                    text=value,
                    required_values=(
                        protected_values_by_field[field]
                    ),
                )

                # Reject common new identifiers that were not part
                # of our controlled fictional values.
                validate_no_unexpected_identifiers(
                    text=value,
                    required_values=required_values,
                )

            enriched = dict(field_values)

            enriched.update(
                {
                    field: value.strip()
                    for field, value in generated_fields.items()
                }
            )

            return enriched, True

        except (
            requests.RequestException,
            json.JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
        ):

            if attempt < max_attempts:
                time.sleep(2 * attempt)
                continue

    return field_values, False


def rewrite_with_openai(
    base_text: str,
    scenario: str,
    variant: str | None,
    language: str,
    required_values: list[str],
    model: str = DEFAULT_MODEL,
    max_attempts: int = 3,
) -> tuple[str, bool]:
    """Rewrite one synthetic document through OpenRouter."""

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not available in the environment."
        )

    prompt = build_rewrite_prompt(
        base_text=base_text,
        scenario=scenario,
        variant=variant,
        language=language,
        required_values=required_values,
    )

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key.strip()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "temperature": 0.7,
                },
                timeout=(10, 90),
            )

            response.raise_for_status()

            data = response.json()

        except requests.RequestException as error:

            if attempt < max_attempts:
                time.sleep(2 * attempt)
                continue

            raise RuntimeError(
                f"OpenRouter request failed after "
                f"{max_attempts} attempts: {error}"
            ) from error

        rewritten_text = (
            data["choices"][0]["message"]["content"]
            .strip()
        )

        for marker in (
            "---BEGIN DOCUMENT---",
            "---END DOCUMENT---",
            "---BEGIN SOURCE DOCUMENT---",
            "---END SOURCE DOCUMENT---",
        ):
            rewritten_text = rewritten_text.replace(
                marker,
                "",
            )

        rewritten_text = rewritten_text.strip()

        if not rewritten_text:
            continue

        try:
            validate_required_values(
                text=rewritten_text,
                required_values=required_values,
            )

            validate_no_unexpected_identifiers(
                text=rewritten_text,
                required_values=required_values,
            )

            return rewritten_text, True

        except ValueError as error:
            protected_values = "\n".join(
                f"- {value}"
                for value in required_values
            )


        prompt += f"""

IMPORTANT CORRECTION:

The previous attempt failed validation.

The following values are protected literals and MUST appear character-for-character
exactly as shown below. Do not change titles, punctuation, spacing, capitalization,
word order, abbreviations, or formatting inside these values:

{protected_values}

Generate the document again while following all original rules.
""".strip()

    # Safe fallback after all retries fail.
    return base_text, False