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

# V2: role-specific model overrides (default to the configured family above).
OUTLINE_MODEL: str = os.environ.get("OUTLINE_MODEL", MODEL_MINI)
RESEARCH_MODEL: str = os.environ.get("RESEARCH_MODEL", MODEL_MINI)
WRITER_MODEL: str = os.environ.get("WRITER_MODEL", MODEL_STRONG)
EDITOR_MODEL: str = os.environ.get("EDITOR_MODEL", MODEL_STRONG)
FACTCHECK_MODEL: str = os.environ.get("FACTCHECK_MODEL", MODEL_MINI)
SEO_MODEL: str = os.environ.get("SEO_MODEL", MODEL_MINI)
# V2: embeddings + audio (used via the raw OpenAI client, not langchain).
EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "whisper-1")
TTS_MODEL: str = os.environ.get("TTS_MODEL", "tts-1")
TTS_VOICE: str = os.environ.get("TTS_VOICE", "nova")

#: Which model each agent role uses (V2 roles included; values resolve to the family above).
MODEL_BY_ROLE: dict[str, str] = {
    "research": RESEARCH_MODEL,
    "outline": OUTLINE_MODEL,
    "writer": WRITER_MODEL,
    "fact_check": FACTCHECK_MODEL,
    "seo": SEO_MODEL,
    "editor": EDITOR_MODEL,
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

# ---------------------------------------------------------------------------
# V2: memory, loop visibility, batch, research RAG, server
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./memory/chroma_store")
MEMORY_COLLECTION: str = os.environ.get("MEMORY_COLLECTION", "pipeline_memory")
MEMORY_SIMILARITY_THRESHOLD: float = _env_float("MEMORY_SIMILARITY_THRESHOLD", 0.82)

#: Below this editor diff magnitude (1 - difflib ratio) the router escalates to HITL (stall gate).
EDIT_DISTANCE_STALL: float = _env_float("EDIT_DISTANCE_STALL", 0.03)

#: Cap on parallel subgraphs in a multi-topic batch (cost control).
MAX_PARALLEL_SUBGRAPHS: int = _env_int("MAX_PARALLEL_SUBGRAPHS", 3)

# Research RAG (full-page fetch + chunking + credibility).
PAGE_FETCH_TOP_N: int = _env_int("PAGE_FETCH_TOP_N", 3)
CHUNK_SIZE_TOKENS: int = _env_int("CHUNK_SIZE_TOKENS", 500)
CREDIBILITY_MIN: float = _env_float("CREDIBILITY_MIN", 0.4)

#: Writing styles the writer/editor understand.
STYLE_PERSONAS: tuple[str, ...] = ("technical deep-dive", "beginner-friendly", "op-ed")

# FastAPI server
V2_PORT: int = _env_int("V2_PORT", 8000)


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
