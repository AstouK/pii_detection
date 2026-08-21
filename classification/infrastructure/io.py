from pathlib import Path
import pandas as pd

from classification.config import (
    DEFAULT_INPUT_FILE,
    CLASSIFICATION_LIMIT,
)


def load_input_data(
    input_file: Path = DEFAULT_INPUT_FILE,
) -> pd.DataFrame:
    """
    Load the classification dataset.
    """

    df = pd.read_csv(input_file)

    date_cols = [
        "file_created_date",
        "last_modified_date",
        "dataset_created_at",
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
            )

    if CLASSIFICATION_LIMIT:
        df = df.head(CLASSIFICATION_LIMIT)

    return df