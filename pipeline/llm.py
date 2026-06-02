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
