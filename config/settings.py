"""
Application settings.

This module is the single place where environment variables are loaded.
Secrets are read from the local .env file or from system environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# Project root is one level above this config/ folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env from the project root
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def require_openrouter_api_key() -> str:
    """
    Return the OpenRouter API key or raise a clear error if it is missing.
    """
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Add OPENROUTER_API_KEY=sk-or-... to your local .env file."
        )

    return OPENROUTER_API_KEY