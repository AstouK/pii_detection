"""
Production classification pipeline.

Executes:
1. Presidio + regex detection
2. LLM review of ambiguous documents
3. Result export
"""

from classification.pii_detector import run_presidio_regex
from classification.llm_reviewer import run_llm
import pandas as pd


"""
Production classification pipeline.

Executes:
1. Presidio + regex detection
2. LLM review of ambiguous documents
3. Result export
"""

# ── Set up Logging ────────────────────────────────────

import logging
from config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ── Load labeled test data ────────────────────────────────────

from pathlib import Path

DATA_FILE = (
    Path(__file__).parent
    / "data"
    / "pii_dataset.xlsx"
)

df = pd.read_excel(
    DATA_FILE,
    parse_dates=["file_created_date", "last_modified_date"],
)
logger.info("Loaded %s rows", len(df))

logger.info("Running Sweep 1: Presidio + regex")
df = run_presidio_regex(df)

logger.info("Running Sweep 2: LLM review")
df = run_llm(df)

df.to_csv("final_output.csv", index=False)

logger.info("Classification pipeline completed. Results saved to %s", DATA_FILE)