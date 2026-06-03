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


# =====================================================================================
# V2 additions — new payload models + the extended V2 state
# =====================================================================================
class PageChunk(BaseModel):
    """A ~500-token chunk of full page text fetched for RAG (V2 research upgrade)."""

    url: str
    chunk_text: str
    chunk_index: int
    relevance_score: float = 0.0


class RevisionRound(BaseModel):
    """A scored record of one editor revision (V2 loop visibility)."""

    round_number: int
    fact_check_score: float
    seo_score: float
    diff_summary: str = ""
    sections_changed: list[str] = Field(default_factory=list)
    # 1 - difflib SequenceMatcher.ratio(): 0.0 = identical, higher = more changed.
    edit_distance: float = 0.0


class ClaimTrace(BaseModel):
    """Per-claim before/after trace across revision rounds (V2 fact-check upgrade)."""

    claim: str
    status: str  # "verified" | "fixed" | "removed" | "flagged"
    original_text: str = ""
    revised_text: str | None = None
    source_url: str | None = None
    round_number: int = 0


class BatchResult(BaseModel):
    """Status + result of one topic in a multi-topic batch run (V2)."""

    topic: str
    status: str = "running"  # "running" | "complete" | "failed"
    final_score: float | None = None
    output_path: str | None = None


class MemoryHit(BaseModel):
    """A past run retrieved from the ChromaDB memory layer (V2)."""

    slug: str
    topic: str
    similarity: float
    sources: list[Source] = Field(default_factory=list)
    fact_sheet: str = ""
    run_date: str = ""


class OutlineSection(BaseModel):
    """One H2 section in the structured outline (V2 outline_agent)."""

    title: str
    facts_to_cover: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    estimated_words: int = 0
    subsections: list[str] = Field(default_factory=list)


class V2PipelineState(PipelineState, total=False):
    """V1 state extended with V2 fields (voice, memory, RAG, loop visibility, batch).

    Inherits every V1 field so the upgraded agents stay backward-compatible; V1's graph
    keeps using :class:`PipelineState` unchanged.
    """

    # Voice I/O
    voice_input_path: str | None
    audio_summary_path: str | None
    # Memory
    memory_hit: bool
    reused_sources: list[Source]
    # Research upgrades
    full_page_chunks: list[PageChunk]
    credibility_scores: dict[str, float]
    # Writing upgrades
    style_persona: str
    self_critique: str
    # Loop visibility
    revision_history: list[RevisionRound]
    flagged_claims_trace: list[ClaimTrace]
    # Multi-topic batch
    batch_topics: list[str]
    batch_results: list[BatchResult]
    # HITL reject feedback
    human_feedback: str | None


def initial_v2_state(
    topic: str,
    target_keyword: str,
    *,
    style_persona: str = "technical deep-dive",
    voice_input_path: str | None = None,
    batch_topics: list[str] | None = None,
) -> V2PipelineState:
    """Return a fully-initialised :class:`V2PipelineState` (reuses V1's defaults)."""

    state: dict[str, Any] = dict(initial_state(topic, target_keyword))
    state.update(
        voice_input_path=voice_input_path,
        audio_summary_path=None,
        memory_hit=False,
        reused_sources=[],
        full_page_chunks=[],
        credibility_scores={},
        style_persona=style_persona,
        self_critique="",
        revision_history=[],
        flagged_claims_trace=[],
        batch_topics=batch_topics or [],
        batch_results=[],
        human_feedback=None,
    )
    return state  # type: ignore[return-value]
