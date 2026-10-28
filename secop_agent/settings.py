"""Paths, environment loading and shared defaults."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
COURSE_DATA_DIR = REPO_ROOT / "datos"  # Provided by the course; kept unchanged (Spanish keys).
DATA_DIR = REPO_ROOT / "data"  # Our own derived data, such as the eligibility key.
RESULTS_DIR = REPO_ROOT / "results"

SAMPLE_PROCESSES_FILE = COURSE_DATA_DIR / "secop_muestra.json"
REFERENCE_COMPANY_FILE = COURSE_DATA_DIR / "empresa.json"
REFERENCE_QUESTIONS_FILE = COURSE_DATA_DIR / "conjunto_referencia.json"
MULTIPURPOSE_COMPANY_FILE = REPO_ROOT / "empresa_multiproposito.json"
ELIGIBILITY_KEY_FILE = DATA_DIR / "eligibility_key.json"

DEFAULT_MODEL = "anthropic:claude-haiku-5-5"

# A hung request must fail loudly instead of stalling a batch of 20 tenders.
MODEL_TIMEOUT_S = 60
MODEL_MAX_RETRIES = 2
# `json_schema` requests hung for over 90 s on Claude Haiku 5.5 (measured 2026-10-28), while
# `function_calling` returned in about 7 s. langchain-anthropic only forces tool choice on models
# that support it, so this method is also safe for Claude Sonnet 5.5.
STRUCTURED_OUTPUT_METHOD = "function_calling"

# LangSmith stops recording traces when these are set together with LANGSMITH_PROJECT.
_CONFLICTING_LANGSMITH_VARS = ("LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT")


def load_environment() -> None:
    """Load `.env` and remove LangSmith variables known to silently disable tracing."""
    load_dotenv(REPO_ROOT / ".env")
    for name in _CONFLICTING_LANGSMITH_VARS:
        if os.environ.pop(name, None) is not None:
            logger.warning("Unset %s: it conflicts with LANGSMITH_PROJECT and disables tracing.", name)


def model_from_env() -> str:
    return os.getenv("MODEL", DEFAULT_MODEL)


def chat_model(model: str | None = None, *, max_tokens: int) -> BaseChatModel:
    """The single place where chat models are built.

    Never passes temperature/top_p/top_k: Claude Haiku 5.5 and API version 2026-10-21 reject them.
    """
    return init_chat_model(
        model or model_from_env(),
        max_tokens=max_tokens,
        timeout=MODEL_TIMEOUT_S,
        max_retries=MODEL_MAX_RETRIES,
    )


def team_name() -> str:
    return os.getenv("TEAM_NAME") or os.getenv("LANGSMITH_PROJECT") or "unnamed-team"


def repository_url() -> str | None:
    """The `origin` remote of this checkout, used when reporting results."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None
