"""Typed state threaded through the LangGraph pipeline.

Design note: the *graph state* is a ``TypedDict`` (``PipelineState``) because LangGraph
applies partial, per-node updates to the state dict and that maps cleanly onto a
``TypedDict``. The structured *payloads* inside the state (``Source``, ``FlaggedClaim``)
and every agent's structured output are Pydantic v2 models, so we still get validation
and typed access where it matters.
"""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single web source returned by the research agent."""

    url: str = Field(..., description="Canonical URL of the source.")
    title: str = Field(..., description="Page/article title.")
    retrieved_at: str = Field(..., description="ISO-8601 timestamp of retrieval.")
    snippet: str = Field(..., description="Short excerpt supporting the cited facts.")
    credibility_note: str = Field(
        ..., description="One-line assessment of how trustworthy this source is."
    )


class FlaggedClaim(BaseModel):
    """A factual claim the fact-check agent could not fully support."""

    claim: str = Field(..., description="The exact claim extracted from the draft.")
    reason: str = Field(..., description="Why it is flagged (unsupported / contradicted / vague).")
    suggested_fix: str = Field(..., description="A concrete rewrite or removal suggestion.")
    source_url: str | None = Field(
        default=None, description="Supporting source URL, or null if none exists."
    )


class PipelineState(TypedDict, total=False):
    """The single object threaded through every node of the graph.

    ``total=False`` so nodes may return partial updates; use :func:`initial_state`
    to construct a fully-populated starting state.
    """

    topic: str
    target_keyword: str
    sources: list[Source]
    fact_sheet: str
    outline: str
    draft: str
    revision_count: int
    fact_check_score: float | None
    flagged_claims: list[FlaggedClaim]
    seo_score: int | None
    seo_feedback: dict[str, Any] | None
    approved: bool
    warnings: list[str]
    # Populated by the publisher on approval (used by the meta.json export + the UI):
    title: str
    meta_description: str
    slug: str
    output_files: dict[str, str] | None


def initial_state(topic: str, target_keyword: str) -> PipelineState:
    """Return a fully-initialised :class:`PipelineState` for a fresh run."""

    return PipelineState(
        topic=topic.strip(),
        target_keyword=target_keyword.strip(),
        sources=[],
        fact_sheet="",
        outline="",
        draft="",
        revision_count=0,
        fact_check_score=None,
        flagged_claims=[],
        seo_score=None,
        seo_feedback=None,
        approved=False,
        warnings=[],
        title="",
        meta_description="",
        slug="",
        output_files=None,
    )
