"""
Application settings.

This module is the single place where environment variables are loaded.
Secrets are read from the local .env file or from system environment variables.
"""

import os
import urllib.error
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

# ── Compliance note ────────────────────────────────────
# OpenRouter may route requests through infrastructure that does not guarantee
# EU data residency. For production use with real personal data, replace this
# with a GDPR-compliant provider such as Azure OpenAI in an EU region, Mistral
# with appropriate data processing terms, or another approved enterprise provider.

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


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

_ollama_available = False


def require_ollama_available() -> str:
    """
    Fail fast if the local Ollama server is not reachable.

    The OpenAI SDK requires an API key even when Ollama does not; the
    default dummy value is "ollama". Set OLLAMA_API_KEY for authenticated
    remote instances.
    """
    global _ollama_available

    if _ollama_available:
        return OLLAMA_BASE_URL

    url = OLLAMA_BASE_URL.rstrip("/") + "/models"
    request = urllib.request.Request(url, method="GET")
    request.add_header("Authorization", f"Bearer {OLLAMA_API_KEY}")

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
    except urllib.error.URLError as exc:
        raise EnvironmentError(
            "Ollama is not reachable at "
            f"{OLLAMA_BASE_URL}. Start Ollama with `ollama serve` "
            f"and pull the model with `ollama pull {OLLAMA_MODEL}`."
        ) from exc

    _ollama_available = True
    return OLLAMA_BASE_URL
