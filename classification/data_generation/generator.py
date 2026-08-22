"""Generate synthetic documents, datasets, and generation manifests."""

import random
from pathlib import Path

import pandas as pd
from faker import Faker

from classification.data_generation.config import (
    DEFAULT_DOCUMENTS_PER_SCENARIO,
)

from classification.data_generation.archetypes import (
    ARCHETYPES,
    get_archetype,
)
from classification.data_generation.planner import (
    build_scenario_requests,
)
from classification.data_generation.records import (
    SyntheticRecord,
)

from classification.data_generation.llm_render import (
    LLM_FREE_TEXT_FIELDS,
    enrich_free_text_fields,
    rewrite_with_openai,
)

from classification.data_generation.generation_values import (
    ENTITY_FIELD_CANDIDATES,
    FAKER_LOCALES,
    GENERIC_STRUCTURED_VALUES,
    GERMAN_FIELD_LABELS,
    GERMAN_TITLES,
    SCENARIO_ENTITY_FIELD_PREFERENCES,
    SCENARIO_VALUES,
    VARIANT_VALUES,
    DATE_FIELDS,
    FREE_TEXT_PLACEHOLDER_FIELDS,
    GENERIC_FIELD_VALUES,
    REFERENCE_FIELDS,
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output"
)

FULL_DOCUMENT_LLM_SCENARIOS = {
    "general_document",
    "internal_email",
}

def make_faker(
    language: str,
    seed: int,
) -> Faker:
    """Create a reproducible Faker instance."""

    fake = Faker(
        FAKER_LOCALES[language]
    )
    fake.seed_instance(seed)

    return fake


def generate_entity_values(
    entity_types: list[str],
    language: str,
    seed: int,
) -> dict[str, str]:
    """Generate fictional values for requested PII entity types."""

    fake = make_faker(
        language=language,
        seed=seed,
    )

    values = {}

    if "PERSON" in entity_types:
        values["PERSON"] = fake.name()

    if "EMAIL_ADDRESS" in entity_types:
        values["EMAIL_ADDRESS"] = (
            fake.safe_email()
        )

    if "PHONE_NUMBER" in entity_types:
        values["PHONE_NUMBER"] = (
            fake.phone_number()
        )

    if "LOCATION" in entity_types:
        values["LOCATION"] = (
            fake.address()
            .replace("\n", ", ")
        )

    if "DATE_TIME" in entity_types:
        values["DATE_TIME"] = str(
            fake.date_between(
                start_date="-3y",
                end_date="today",
            )
        )

    if "IBAN_CODE" in entity_types:
        values["IBAN_CODE"] = fake.iban()

    if "CREDIT_CARD" in entity_types:
        values["CREDIT_CARD"] = (
            fake.credit_card_number()
        )

    if "IP_ADDRESS" in entity_types:
        values["IP_ADDRESS"] = (
            fake.ipv4_public()
        )

    if "URL" in entity_types:
        values["URL"] = (
            fake.url()
        )

    if "PASSPORT" in entity_types:
        values["PASSPORT"] = (
            f"P{fake.random_number(digits=8, fix_len=True)}"
        )

    if "NRP" in entity_types:
        values["NRP"] = (
            "German"
            if language == "en"
            else "deutsch"
        )

    if "MEDICAL_LICENSE" in entity_types:
        values["MEDICAL_LICENSE"] = (
            f"MED-{fake.random_number(digits=7, fix_len=True)}"
        )

    return values

def assign_entities_to_fields(
    fields: list[str],
    entity_values: dict[str, str],
    scenario: str | None = None,
    seed: int | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Assign requested entities to realistic document fields."""

    rng = random.Random(seed)

    assignments = {}

    scenario_preferences = (
        SCENARIO_ENTITY_FIELD_PREFERENCES.get(
            scenario,
            {},
        )
    )

    for entity, value in entity_values.items():
        candidates = scenario_preferences.get(
            entity,
            ENTITY_FIELD_CANDIDATES.get(
                entity,
                [],
            ),
        )

        available_fields = [
            field
            for field in candidates
            if field in fields
        ]

        if not available_fields:
            raise ValueError(
                f"Could not place entity '{entity}' "
                f"in scenario '{scenario}' "
                f"with available fields {fields}."
            )

        # The first candidate is the preferred placement. Most rows use it; the remainder vary placement across other valid fields.
        if (
            len(available_fields) == 1
            or rng.random() < 0.70
        ):
            matching_field = available_fields[0]
        else:
            matching_field = rng.choice(
                available_fields[1:]
            )

        assignments.setdefault(
            matching_field,
            [],
        ).append(
            (entity, value)
        )

    return assignments


def format_entity_assignment(
    assignments: list[tuple[str, str]],
    language: str,
) -> str:
    """Format controlled entity values naturally inside a field."""

    values = dict(assignments)

    if "PERSON" in values and "EMAIL_ADDRESS" in values:
        return (
            f"{values['PERSON']} "
            f"({values['EMAIL_ADDRESS']})"
        )

    if "PERSON" in values and "PHONE_NUMBER" in values:
        return (
            f"{values['PERSON']}, "
            f"{values['PHONE_NUMBER']}"
        )

    if "PERSON" in values and "IBAN_CODE" in values:
        return (
            f"{values['PERSON']}; "
            f"IBAN: {values['IBAN_CODE']}"
        )

    if len(values) == 1:
        return next(iter(values.values()))

    return "; ".join(
        values.values()
    )


def field_label(
    field: str,
    language: str,
) -> str:
    """Return a human-readable field label."""

    if language == "de":
        return GERMAN_FIELD_LABELS.get(
            field,
            field.replace("_", " ").title(),
        )

    return (
        field.replace("_", " ").title()
    )


def document_title(
    scenario: str,
    language: str,
) -> str:
    """Return a document title for the scenario."""

    if language == "de":
        return GERMAN_TITLES.get(
            scenario,
            scenario.replace("_", " ").title(),
        )

    return (
        scenario
        .replace("_", " ")
        .title()
    )


def generate_generic_value(
    field: str,
    language: str,
    seed: int,
    scenario: str | None = None,
    variant: str | None = None,
) -> str:
    """Generate a fictional non-PII/default value for a document field."""

    fake = make_faker(
        language=language,
        seed=seed,
    )

    if (
        scenario in VARIANT_VALUES
        and variant in VARIANT_VALUES[scenario]
    ):
        field_values = VARIANT_VALUES[scenario][variant]

        if field in field_values:
            language_values = field_values[field]

            return fake.random_element(
                elements=language_values[language]
            )

    if scenario in SCENARIO_VALUES:
        field_values = SCENARIO_VALUES[scenario]

        if field in field_values:
            return fake.random_element(
                elements=field_values[field]
            )

    if field in DATE_FIELDS:
        return str(
            fake.date_between(
                start_date="-2y",
                end_date="+1y",
            )
        )

    if field == "amount":
        return (
            f"{fake.random_int(20, 900)}.00 EUR"
        )

    if field in GENERIC_FIELD_VALUES:
        return fake.random_element(
            elements=GENERIC_FIELD_VALUES[field]
        )

    if field == "receipt_reference":
        return (
            f"RCPT-"
            f"{fake.random_number(digits=6, fix_len=True)}"
        )

    if field in REFERENCE_FIELDS:
        return (
            f"REF-"
            f"{fake.random_number(digits=7, fix_len=True)}"
        )

    # Free-text placeholders are replaced by LLM enrichment
    # in the production generation path.
    if field in FREE_TEXT_PLACEHOLDER_FIELDS:
        if language == "de":
            return (
                "Interne Dokumentation für den "
                "regulären Geschäftsvorgang."
            )

        return (
            "Internal documentation for the "
            "standard business process."
        )

    if field in GENERIC_STRUCTURED_VALUES:
        return fake.random_element(
            elements=GENERIC_STRUCTURED_VALUES[field][language]
        )

    return (
        "Information not available"
        if language == "en"
        else "Information nicht verfügbar"
    )


def build_document_fields(
    request,
    seed: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build controlled field values before rendering the document."""

    archetype = get_archetype(
        request.scenario
    )

    # Blank templates preserve the document structure without values or PII.
    if request.variant == "blank":
        field_values = {
            field: ""
            for field in archetype.fields
        }

        return field_values, {}

    entity_values = generate_entity_values(
        entity_types=request.entity_types,
        language=request.language,
        seed=seed,
    )

    entity_assignments = assign_entities_to_fields(
        fields=archetype.fields,
        entity_values=entity_values,
        scenario=request.scenario,
        seed=seed,
    )

    field_values = {}

    for index, field in enumerate(
        archetype.fields
    ):
        if field in entity_assignments:
            value = format_entity_assignment(
                assignments=entity_assignments[field],
                language=request.language,
            )
        else:
            value = generate_generic_value(
                field=field,
                language=request.language,
                seed=seed + index,
                scenario=request.scenario,
                variant=request.variant,
            )

        field_values[field] = value

    return field_values, entity_values


def render_document(
    scenario: str,
    language: str,
    field_values: dict[str, str],
) -> str:
    """Render structured field values into document text."""

    lines = [
        document_title(
            scenario,
            language,
        ),
        "",
    ]

    for field, value in field_values.items():
        lines.append(
            f"{field_label(field, language)}: {value}"
        )

    return "\n".join(lines)


def generation_metadata(
    scenario: str,
    variant: str | None,
    use_llm: bool,
    llm_success: bool,
) -> tuple[str, str]:
    """Return generator and prompt metadata for one generated document."""

    if (
        use_llm
        and scenario in LLM_FREE_TEXT_FIELDS
        and variant != "blank"
    ):
        if llm_success:
            return (
                "openrouter_gpt4o_mini_field",
                "field_llm_v1",
            )

        return (
            "faker_archetype_fallback",
            "field_llm_v1_fallback",
        )

    if (
        use_llm
        and scenario in FULL_DOCUMENT_LLM_SCENARIOS
    ):
        if llm_success:
            return (
                "openrouter_gpt4o_mini_document",
                "llm_rewrite_v1",
            )

        return (
            "faker_archetype_fallback",
            "llm_rewrite_v1_fallback",
        )

    return (
        "faker_archetype_v1",
        "template_v1",
    )


def generate_scenario_dataframe(
    scenario: str,
    n_documents: int = DEFAULT_DOCUMENTS_PER_SCENARIO,
    seed: int = 42,
    use_llm: bool = False,
    generation_limit: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic rows for one scenario."""

    requests = build_scenario_requests(
        scenario=scenario,
        n_documents=n_documents,
        seed=seed,
    )

    if generation_limit is not None:
        requests = requests[:generation_limit]

    archetype = get_archetype(
        scenario
    )

    rows = []

    for index, request in enumerate(
        requests,
        start=1,
    ):
        row_seed = derive_row_seed(
            scenario_seed=seed,
            row_index=index,
        )

        llm_success = False

        field_values, entity_values = build_document_fields(
            request=request,
            seed=row_seed,
        )

        if (
            use_llm
            and scenario in LLM_FREE_TEXT_FIELDS
            and request.variant != "blank"
        ):
            field_values, llm_success = enrich_free_text_fields(
                scenario=request.scenario,
                variant=request.variant,
                language=request.language,
                field_values=field_values,
                required_values=list(
                    entity_values.values()
                ),
            )

        base_text = render_document(
            scenario=request.scenario,
            language=request.language,
            field_values=field_values,
        )

        full_text = base_text

        if (
            use_llm
            and scenario in FULL_DOCUMENT_LLM_SCENARIOS
        ):
            full_text, llm_success = rewrite_with_openai(
                base_text=base_text,
                scenario=request.scenario,
                variant=request.variant,
                language=request.language,
                required_values=list(
                    entity_values.values()
                ),
            )

        synthetic_generator, generation_prompt_version = (
            generation_metadata(
                scenario=request.scenario,
                variant=request.variant,
                use_llm=use_llm,
                llm_success=llm_success,
            )
        )

        record = SyntheticRecord(
            document_id=(
                f"SYN-{scenario.upper()}-"
                f"{index:04d}"
            ),
            file_name=(
                f"synthetic_{scenario}_"
                f"{index:04d}.txt"
            ),
            document_type=archetype.document_type,
            scenario_type=request.scenario,
            language=request.language,
            full_text=full_text,
            contains_personal_data=(
                request.contains_personal_data
            ),
            entity_types=request.entity_types,
            difficulty=request.difficulty,
            edge_case=request.edge_case,
            challenge_category=(
                request.challenge_category
            ),
            recommended_split=(
                request.recommended_split
            ),
            synthetic_generator=synthetic_generator,
            generation_prompt_version=(
                generation_prompt_version
            ),
        )

        rows.append(
            record.to_dataset_row()
        )

    return pd.DataFrame(rows)


def derive_scenario_seed(
    base_seed: int,
    scenario_index: int,
) -> int:
    """Derive a reproducible seed for one scenario."""

    return (
        base_seed
        + scenario_index * 1000
    )


def derive_row_seed(
    scenario_seed: int,
    row_index: int,
) -> int:
    """Derive a reproducible seed for one generated document."""

    return (
        scenario_seed * 10_000
        + row_index
    )


def generate_scenario_manifest(
    scenario: str,
    n_documents: int = DEFAULT_DOCUMENTS_PER_SCENARIO,
    seed: int = 42,
) -> pd.DataFrame:
    """Create internal generation metadata for one scenario."""

    requests = build_scenario_requests(
        scenario=scenario,
        n_documents=n_documents,
        seed=seed,
    )

    rows = []

    for index, request in enumerate(
        requests,
        start=1,
    ):
        row_seed = derive_row_seed(
            scenario_seed=seed,
            row_index=index,
        )

        rows.append(
            {
                "document_id": (
                    f"SYN-{scenario.upper()}-"
                    f"{index:04d}"
                ),
                "scenario_type": request.scenario,
                "variant": request.variant,
                "language": request.language,
                "contains_personal_data": (
                    "yes"
                    if request.contains_personal_data
                    else "no"
                ),
                "entity_types": ";".join(
                    request.entity_types
                ),
                "recommended_split": (
                    request.recommended_split
                ),
                "difficulty": request.difficulty,
                "edge_case": request.edge_case,
                "challenge_category": (
                    request.challenge_category
                ),
                "scenario_seed": seed,
                "row_seed": row_seed,
            }
        )

    return pd.DataFrame(rows)


def generate_dataset_dataframe(
    documents_per_scenario: int = DEFAULT_DOCUMENTS_PER_SCENARIO,
    seed: int = 42,
    use_llm: bool = False,
) -> pd.DataFrame:
    """Generate the complete synthetic dataset across all scenarios."""

    frames = []

    for index, scenario in enumerate(
        sorted(ARCHETYPES)
    ):
        scenario_seed = derive_scenario_seed(
            base_seed=seed,
            scenario_index=index,
        )

        frame = generate_scenario_dataframe(
            scenario=scenario,
            n_documents=documents_per_scenario,
            seed=scenario_seed,
            use_llm=use_llm,
        )

        frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def generate_dataset_manifest(
    documents_per_scenario: int = DEFAULT_DOCUMENTS_PER_SCENARIO,
    seed: int = 42,
) -> pd.DataFrame:
    """Create the generation manifest across all scenarios."""

    frames = []

    for index, scenario in enumerate(
        sorted(ARCHETYPES)
    ):
        scenario_seed = derive_scenario_seed(
            base_seed=seed,
            scenario_index=index,
        )

        frame = generate_scenario_manifest(
            scenario=scenario,
            n_documents=documents_per_scenario,
            seed=scenario_seed,
        )

        frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def save_manifest(
    manifest: pd.DataFrame,
    file_name: str = "synthetic_dataset_manifest.csv",
) -> Path:
    """Save generation metadata separately from the training dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / file_name

    manifest.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_dataset(
    df: pd.DataFrame,
    file_name: str = "synthetic_dataset.csv",
) -> Path:
    """Save a generated dataset without modifying the existing seed dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / file_name

    df.to_csv(
        output_path,
        index=False,
    )

    return output_path