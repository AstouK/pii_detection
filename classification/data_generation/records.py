"""Serialize generated documents into the canonical dataset schema."""

from dataclasses import dataclass, field
from datetime import date

from classification.data_generation.config import (
    DATASET_COLUMNS,
    SUPPORTED_ENTITY_TYPES,
)


@dataclass
class SyntheticRecord:
    """Represent one generated document before dataset export."""

    document_id: str
    file_name: str
    document_type: str
    scenario_type: str
    language: str
    full_text: str

    contains_personal_data: bool

    entity_types: list[str] = field(default_factory=list)

    difficulty: str = "easy"
    edge_case: bool = False
    challenge_category: str = "none"

    recommended_split: str = "train"

    synthetic_generator: str = "synthetic_generator_v1"
    generation_prompt_version: str = "v1"

    dataset_version: str = "v2"


    def to_dataset_row(self) -> dict:
        """Convert the generated record to the shared dataset schema."""

        entity_flags = {
            f"{entity}_yes_no": (
                "yes"
                if entity in self.entity_types
                else "no"
            )
            for entity in SUPPORTED_ENTITY_TYPES
        }

        if not self.entity_types:
            primary_pii_type = "NONE"
        elif len(self.entity_types) == 1:
            primary_pii_type = self.entity_types[0]
        else:
            primary_pii_type = "MULTIPLE"

        today = date.today().isoformat()

        row = {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "document_type": self.document_type,
            "scenario_type": self.scenario_type,
            "language": self.language,

            "source_system": "synthetic_generation",
            "responsible_owner": "",
            "owner_email": "",
            "data_source": "synthetic",

            "file_created_date": today,
            "last_modified_date": today,
            "dataset_created_at": today,
            "file_size_mb": 0,

            "full_text": self.full_text,

            "contains_personal_data": (
                "yes"
                if self.contains_personal_data
                else "no"
            ),

            **entity_flags,

            "personal_data_categories": "; ".join(
                self.entity_types
            ),
            "primary_pii_type": primary_pii_type,
            "category_count": len(self.entity_types),
            "pii_count": len(self.entity_types),

            "difficulty": self.difficulty,
            "edge_case": (
                "yes"
                if self.edge_case
                else "no"
            ),
            "challenge_category": self.challenge_category,
            "retention_period_exceeded_3y": "no",
            "recommended_split": self.recommended_split,

            "synthetic": "yes",
            "synthetic_generator": self.synthetic_generator,
            "generation_prompt_version": (
                self.generation_prompt_version
            ),

            "human_reviewed": "no",
            "review_status": "",
            "review_notes": "",

            "dataset_version": self.dataset_version,
            "labeling_notes": (
                "Synthetic document generated for controlled "
                "PII training data."
            ),
        }

        return {
            column: row[column]
            for column in DATASET_COLUMNS
        }