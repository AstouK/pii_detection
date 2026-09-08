"""Generation request model and validation."""

from dataclasses import dataclass, field

from classification.data_generation.archetypes import get_archetype
from classification.data_generation.config import (
    SUPPORTED_DIFFICULTIES,
    SUPPORTED_ENTITY_TYPES,
    SUPPORTED_LANGUAGES,
    SUPPORTED_SCENARIOS,
    SUPPORTED_SPLITS,
)


@dataclass
class GenerationRequest:
    """Describe one synthetic document that should be generated."""

    scenario: str
    language: str
    contains_personal_data: bool
    entity_types: list[str] = field(default_factory=list)
    difficulty: str = "easy"
    edge_case: bool = False
    challenge_category: str = "none"
    recommended_split: str = "train"
    variant: str | None = None

    def __post_init__(self) -> None:
        """Validate that the generation request is internally consistent."""

        # 1. PII-positive documents need at least one PII entity.
        if self.contains_personal_data and not self.entity_types:
            raise ValueError(
                "PII-positive requests must specify at least one entity type."
            )

        # 2. PII-negative documents cannot request PII entities.
        if not self.contains_personal_data and self.entity_types:
            raise ValueError(
                "PII-negative requests cannot specify PII entity types."
            )

        # 3. Language must be supported.
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {self.language}. "
                f"Expected one of {sorted(SUPPORTED_LANGUAGES)}."
            )

        # 4. Difficulty must be supported.
        if self.difficulty not in SUPPORTED_DIFFICULTIES:
            raise ValueError(
                f"Unsupported difficulty: {self.difficulty}. "
                f"Expected one of {sorted(SUPPORTED_DIFFICULTIES)}."
            )

        # 5. Every requested PII entity must exist in our taxonomy.
        unsupported_entities = (
            set(self.entity_types) - SUPPORTED_ENTITY_TYPES
        )

        if unsupported_entities:
            raise ValueError(
                f"Unsupported entity types: "
                f"{sorted(unsupported_entities)}."
            )

        # 6. Scenario must be supported.
        if self.scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(
                f"Unsupported scenario: {self.scenario}. "
                f"Expected one of {sorted(SUPPORTED_SCENARIOS)}."
            )

        # 7. Find the implemented archetype for this scenario.
        archetype = get_archetype(self.scenario)

        # 8. If a specific variant was requested, make sure it exists.
        if self.variant is not None and self.variant not in archetype.variants:
            raise ValueError(
                f"Unsupported variant '{self.variant}' "
                f"for scenario '{self.scenario}'. "
                f"Expected one of {archetype.variants}."
            )

        if self.recommended_split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported split: {self.recommended_split}. "
                f"Expected one of {sorted(SUPPORTED_SPLITS)}."
            )