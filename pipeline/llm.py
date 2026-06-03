"""Factories for the LLM and web-search clients, with an offline mock mode.

``get_llm(role)`` returns a chat model configured for an agent role; ``get_search_client()``
returns a Tavily client. When :data:`pipeline.config.MOCK_MODE` is on, both return
deterministic fakes so ``pytest``/CI run with no API keys and no network calls.

The mock chat model mirrors the small slice of the LangChain runnable surface the agents
use: ``.with_structured_output(schema)`` and ``.with_retry(...)``, plus ``.invoke(...)``.
Agents register canned, schema-shaped responses in :data:`MOCK_RESPONSES`; anything not
registered falls back to a generic, type-aware filler so a call never crashes in mock mode.
"""

from __future__ import annotations

import hashlib
import math
import random
import types
import typing
from typing import Any, Callable, Type, TypeVar, get_args, get_origin

from pydantic import BaseModel

from pipeline import config

T = TypeVar("T", bound=BaseModel)

#: role/schema-name -> dict | callable(schema)->dict. Populated by agents (and tests) to
#: return deterministic structured outputs in mock mode. Keys are tried in this order:
#: ``f"{role}:{SchemaName}"``, then ``role``, then ``SchemaName``.
MOCK_RESPONSES: dict[str, Any] = {}

#: role -> plain-text response used for non-structured ``.invoke`` calls in mock mode.
MOCK_TEXT: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Generic, type-aware Pydantic filler (mock fallback)
# ---------------------------------------------------------------------------
def _is_optional(annotation: Any) -> bool:
    return get_origin(annotation) in (typing.Union, types.UnionType) and type(None) in get_args(
        annotation
    )


def _mock_value(annotation: Any, field_name: str) -> Any:
    """Synthesise a plausible value for a type annotation (used only in mock mode)."""

    if annotation in (str, "str"):
        return f"mock {field_name}"
    if annotation in (bool,):
        return False
    if annotation in (int,):
        return 0
    if annotation in (float,):
        return 0.0
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return build_mock_instance(annotation)

    origin = get_origin(annotation)
    if origin in (list, set, tuple):
        return []
    if origin is dict:
        return {}
    if _is_optional(annotation):
        return None
    # Union of non-None types: use the first arm.
    if origin in (typing.Union, types.UnionType):
        return _mock_value(get_args(annotation)[0], field_name)
    return None


def build_mock_instance(schema: Type[T], role: str | None = None) -> T:
    """Build a deterministic instance of ``schema`` for mock mode.

    Prefers a registered canned response; otherwise fills required fields generically.
    """

    for key in (f"{role}:{schema.__name__}", role or "", schema.__name__):
        override = MOCK_RESPONSES.get(key)
        if override is not None:
            data = override(schema) if callable(override) else override
            return schema.model_validate(data)

    data: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if not field.is_required():
            continue
        data[name] = _mock_value(field.annotation, name)
    return schema.model_validate(data)


# ---------------------------------------------------------------------------
# Mock chat model
# ---------------------------------------------------------------------------
class _MockStructuredRunnable:
    """Stand-in for ``ChatModel.with_structured_output(schema)`` in mock mode."""

    def __init__(self, schema: Type[BaseModel], role: str) -> None:
        self._schema = schema
        self._role = role

    def with_retry(self, *args: Any, **kwargs: Any) -> "_MockStructuredRunnable":
        return self

    def invoke(self, *args: Any, **kwargs: Any) -> BaseModel:
        return build_mock_instance(self._schema, self._role)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> BaseModel:
        return self.invoke(*args, **kwargs)


class MockChatModel:
    """Deterministic, offline stand-in for ``ChatOpenAI``."""

    def __init__(self, role: str, model: str) -> None:
        self.role = role
        self.model = model

    def with_structured_output(
        self, schema: Type[BaseModel], **_: Any
    ) -> _MockStructuredRunnable:
        return _MockStructuredRunnable(schema, self.role)

    def with_retry(self, *args: Any, **kwargs: Any) -> "MockChatModel":
        return self

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage

        return AIMessage(content=MOCK_TEXT.get(self.role, f"[mock {self.role} output]"))

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.invoke(*args, **kwargs)


# ---------------------------------------------------------------------------
# Mock search client
# ---------------------------------------------------------------------------
class MockSearchClient:
    """Offline stand-in mirroring ``TavilyClient.search``'s response shape."""

    def search(self, query: str, **_: Any) -> dict[str, Any]:
        return {
            "query": query,
            "results": [
                {
                    "title": f"Mock source {i} for {query}",
                    "url": f"https://example.com/{i}",
                    "content": f"Mock evidence snippet {i} about {query}.",
                    "score": 0.9 - i * 0.05,
                }
                for i in range(1, config.TAVILY_MAX_RESULTS + 1)
            ],
        }


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------
def get_llm(role: str) -> Any:
    """Return a chat model for an agent ``role``.

    Live mode -> a configured ``ChatOpenAI``; mock mode -> :class:`MockChatModel`.
    Callers apply ``.with_structured_output(...)`` and/or ``.with_retry(...)`` as needed.
    """

    model = config.model_for_role(role)
    if config.MOCK_MODE:
        return MockChatModel(role=role, model=model)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=config.require_key("OPENAI_API_KEY"),
        temperature=config.TEMPERATURE,
    )


def get_search_client() -> Any:
    """Return a Tavily client (live) or :class:`MockSearchClient` (mock mode)."""

    if config.MOCK_MODE:
        return MockSearchClient()

    from tavily import TavilyClient

    return TavilyClient(api_key=config.require_key("TAVILY_API_KEY"))


# ---------------------------------------------------------------------------
# V2: OpenAI client (embeddings + audio), deterministic in mock mode
# ---------------------------------------------------------------------------
def _deterministic_vector(text: str, dim: int = 64) -> list[float]:
    """A stable unit vector derived from ``text`` (mock embeddings).

    Identical text -> identical vector (cosine 1.0); different text -> ~orthogonal, so
    similarity thresholds behave predictably in tests without calling the embeddings API.
    """

    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class _MockEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _MockEmbeddingResponse:
    def __init__(self, data: list["_MockEmbeddingItem"]) -> None:
        self.data = data


class _MockEmbeddings:
    def create(self, model: str, input: Any, **_: Any) -> "_MockEmbeddingResponse":
        texts = input if isinstance(input, list) else [input]
        return _MockEmbeddingResponse([_MockEmbeddingItem(_deterministic_vector(t)) for t in texts])


class _MockTranscription:
    def __init__(self, text: str) -> None:
        self.text = text


class _MockTranscriptions:
    def create(self, model: str, file: Any, **_: Any) -> "_MockTranscription":
        name = getattr(file, "name", "audio")
        return _MockTranscription(f"Mock transcript for {name}")


class _MockSpeechResponse:
    def __init__(self, data: bytes) -> None:
        self.content = data

    def stream_to_file(self, path: str) -> None:
        with open(path, "wb") as handle:
            handle.write(self.content)

    write_to_file = stream_to_file


class _MockSpeech:
    def create(self, model: str, voice: str, input: str, **_: Any) -> "_MockSpeechResponse":
        return _MockSpeechResponse(b"ID3\x03mock-mp3-" + input[:48].encode("utf-8", "ignore"))


class _MockAudio:
    def __init__(self) -> None:
        self.transcriptions = _MockTranscriptions()
        self.speech = _MockSpeech()


class MockOpenAIClient:
    """Offline stand-in for the raw ``openai.OpenAI`` client (embeddings + audio)."""

    def __init__(self) -> None:
        self.embeddings = _MockEmbeddings()
        self.audio = _MockAudio()


def get_openai_client() -> Any:
    """Return a raw OpenAI client (live) or :class:`MockOpenAIClient` (mock mode).

    Used for embeddings and audio (Whisper/TTS); the chat path still uses :func:`get_llm`.
    """

    if config.MOCK_MODE:
        return MockOpenAIClient()

    from openai import OpenAI

    return OpenAI(api_key=config.require_key("OPENAI_API_KEY"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with the configured model (deterministic in mock mode)."""

    if not texts:
        return []
    response = get_openai_client().embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# V2: full-page fetch for deep-research RAG (async)
# ---------------------------------------------------------------------------
_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ContentPipelineBot/2.0; +https://example.com/bot)"
}


async def fetch_page_html(url: str, timeout: float = 10.0) -> str:
    """Fetch a page's raw HTML (async). Mock mode returns deterministic canned HTML."""

    if config.MOCK_MODE:
        filler = "This is mock page content describing the topic in detail. " * 60
        return (
            f"<html><body><article><h1>Mock Page for {url}</h1>"
            f"<p>{filler}</p></article></body></html>"
        )

    import httpx

    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers=_FETCH_HEADERS
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
