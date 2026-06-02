"""Central configuration: model names, thresholds, the loop guard, and key loading.

Everything tunable lives here so the pipeline's behaviour is observable and cost can be
controlled from one place. Values may be overridden via environment variables (see
``.env.example``); otherwise the defaults below apply.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load .env from the project root (if present) the moment config is imported.
load_dotenv()


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------
#: When true, get_llm()/get_search_client() return deterministic fakes (no keys needed).
MOCK_MODE: bool = _truthy(os.environ.get("PIPELINE_MOCK"))

# ---------------------------------------------------------------------------
# Models  (cost-aware: a stronger model for prose, a cheaper one for graders)
# ---------------------------------------------------------------------------
MODEL_STRONG: str = os.environ.get("MODEL_STRONG", "gpt-4.1")
MODEL_MINI: str = os.environ.get("MODEL_MINI", "gpt-4.1-mini")
TEMPERATURE: float = _env_float("LLM_TEMPERATURE", 0.3)
LLM_MAX_RETRIES: int = _env_int("LLM_MAX_RETRIES", 3)

#: Which model each agent role uses.
MODEL_BY_ROLE: dict[str, str] = {
    "research": MODEL_MINI,
    "writer": MODEL_STRONG,
    "fact_check": MODEL_MINI,
    "seo": MODEL_MINI,
    "editor": MODEL_STRONG,
}


def model_for_role(role: str) -> str:
    """Return the model name configured for an agent ``role`` (mini if unknown)."""

    return MODEL_BY_ROLE.get(role, MODEL_MINI)


# ---------------------------------------------------------------------------
# Quality thresholds + loop guard
# ---------------------------------------------------------------------------
#: Below this fact-check score (0.0-1.0) the router sends the draft back to the editor.
FACT_CHECK_THRESHOLD: float = _env_float("FACT_CHECK_THRESHOLD", 0.85)
#: Below this SEO score (0-100) the router sends the draft back to the editor.
SEO_THRESHOLD: int = _env_int("SEO_THRESHOLD", 70)
#: Hard cap on editor revisions; the router checks this FIRST so the loop can never run away.
MAX_REVISIONS: int = _env_int("MAX_REVISIONS", 3)

# ---------------------------------------------------------------------------
# Research / search
# ---------------------------------------------------------------------------
TAVILY_MAX_RESULTS: int = _env_int("TAVILY_MAX_RESULTS", 7)
TAVILY_SEARCH_DEPTH: str = os.environ.get("TAVILY_SEARCH_DEPTH", "advanced")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "output")


class MissingKeyError(RuntimeError):
    """Raised with a friendly, actionable message when a required API key is absent."""


def require_key(name: str) -> str:
    """Return the value of env var ``name`` or raise :class:`MissingKeyError`.

    Only called on the live (non-mock) code path, so mock mode never needs keys.
    """

    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingKeyError(
            f"Missing required environment variable {name}.\n"
            f"  -> Copy .env.example to .env and set {name}=<your key>, or\n"
            f"  -> run in mock mode with PIPELINE_MOCK=1 (no API keys required)."
        )
    return value
