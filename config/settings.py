"""
Application settings.

This module is the single place where environment variables are loaded.
Secrets are read from the local .env file or from system environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def require_openrouter_api_key() -> str:
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Add OPENROUTER_API_KEY=... to your local .env file.")
    return OPENROUTER_API_KEY


QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.7-plus")


def require_qwen_api_key() -> str:
    if not QWEN_API_KEY:
        raise EnvironmentError("QWEN_API_KEY is not set. Add QWEN_API_KEY=... to your local .env file.")
    return QWEN_API_KEY
