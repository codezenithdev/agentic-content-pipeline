"""research_agent — turn a topic into a sourced fact sheet.

Input  (state): ``topic``
Output (state update): ``sources`` (list[Source]), ``fact_sheet`` (markdown str), ``warnings``

Flow:
1. Query Tavily for real web sources about the topic.
2. Ask the LLM (structured output) to assess each source's credibility and to extract a
   bullet list of facts, each tagged with the source URL(s) that support it.
3. Render a markdown fact sheet and return typed :class:`~pipeline.state.Source` objects.

Facts and source URLs are constrained to the URLs Tavily actually returned, so the agent
cannot invent references. Degrades gracefully when Tavily returns nothing or errors.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pipeline import config, llm
from pipeline.state import PipelineState, Source

# --- snippet length cap for what we feed the model / store on each Source -------------
_SNIPPET_CHARS = 500


# --- structured output schemas (LLM fills these) --------------------------------------
class _SourceAssessment(BaseModel):
    """The model's per-source assessment (timestamp is added in code, not by the LLM)."""

    url: str = Field(..., description="Must be one of the provided source URLs.")
    title: str
    snippet: str = Field(..., description="Short excerpt that supports the cited facts.")
    credibility_note: str = Field(
        ..., description="One line on how trustworthy/relevant this source is."
    )


class _FactItem(BaseModel):
    fact: str = Field(..., description="A single, verifiable factual statement.")
    source_urls: list[str] = Field(
        ..., description="URLs (from the provided list) that support this fact."
    )


class _ResearchOutput(BaseModel):
    sources: list[_SourceAssessment]
    facts: list[_FactItem]


# Deterministic mock-mode response (consistent with llm.MockSearchClient URLs).
llm.MOCK_RESPONSES["research"] = {
    "sources": [
        {
            "url": "https://example.com/1",
            "title": "Mock source 1",
            "snippet": "Mock evidence about the topic.",
            "credibility_note": "Illustrative mock source (offline mode).",
        },
        {
            "url": "https://example.com/2",
            "title": "Mock source 2",
            "snippet": "More mock evidence about the topic.",
            "credibility_note": "Illustrative mock source (offline mode).",
        },
    ],
    "facts": [
        {"fact": "This is a mock fact grounded in the sources.", "source_urls": ["https://example.com/1"]},
        {"fact": "A second mock fact.", "source_urls": ["https://example.com/1", "https://example.com/2"]},
    ],
}

_SYSTEM = (
    "You are a meticulous research analyst. You are given a topic and a numbered list of real "
    "web sources (title, URL, excerpt). Your job:\n"
    "1. For EACH source, write a one-line credibility note (who published it, how reliable/"
    "relevant it is).\n"
    "2. Extract 8-14 specific, verifiable facts about the topic. Tag each fact with the URL(s) "
    "from the provided list that support it.\n"
    "Rules: Use ONLY the provided sources. Never invent facts or URLs. Every fact's source_urls "
    "must be drawn from the provided list. Prefer concrete facts (numbers, dates, named entities) "
    "over vague statements."
)


def _tavily_search(topic: str) -> list[dict]:
    """Run a Tavily search and return the raw result dicts (empty list on any failure)."""

    client = llm.get_search_client()
    try:
        response = client.search(
            topic,
            max_results=config.TAVILY_MAX_RESULTS,
            search_depth=config.TAVILY_SEARCH_DEPTH,
        )
    except Exception as exc:  # network error, auth error, rate limit, etc.
        raise _SearchError(str(exc)) from exc
    return list(response.get("results") or [])


class _SearchError(RuntimeError):
    """Internal: Tavily call failed (surfaced to the user as a warning, not a crash)."""


def _render_fact_sheet(topic: str, sources: list[Source], facts: list[_FactItem]) -> str:
    """Render the markdown fact sheet: tagged facts + a numbered source list."""

    lines = [f"# Fact Sheet: {topic}", "", "## Facts"]
    for item in facts:
        tags = ", ".join(item.source_urls) if item.source_urls else "unverified"
        lines.append(f"- {item.fact}  _[source: {tags}]_")
    lines += ["", "## Sources"]
    for i, src in enumerate(sources, 1):
        lines.append(f"{i}. [{src.title}]({src.url}) — {src.credibility_note}")
    return "\n".join(lines)


def research_agent(state: PipelineState) -> dict:
    """Research the topic and produce a sourced fact sheet (LangGraph node).

    Returns a partial state update with ``sources``, ``fact_sheet`` and any ``warnings``.
    """

    topic = state["topic"]
    warnings = list(state.get("warnings", []))

    # 1) Real web search.
    try:
        raw = _tavily_search(topic)
    except _SearchError as exc:
        warnings.append(f"Web search failed for topic '{topic}': {exc}")
        return {"sources": [], "fact_sheet": "", "warnings": warnings}

    if not raw:
        warnings.append(f"Web search returned no sources for topic '{topic}'.")
        return {"sources": [], "fact_sheet": "", "warnings": warnings}

    # 2) Structured extraction: credibility notes + tagged facts.
    catalog = "\n".join(
        f"[{i}] {r.get('title', 'Untitled')}\n    URL: {r.get('url', '')}\n"
        f"    Excerpt: {(r.get('content') or '')[:_SNIPPET_CHARS]}"
        for i, r in enumerate(raw, 1)
    )
    model = llm.get_llm("research").with_structured_output(_ResearchOutput).with_retry(
        stop_after_attempt=config.LLM_MAX_RETRIES
    )
    result: _ResearchOutput = model.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"TOPIC: {topic}\n\nSOURCES:\n{catalog}"),
        ]
    )

    # 3) Build typed Source objects (timestamp added here, not trusted from the model).
    retrieved_at = datetime.now(timezone.utc).isoformat()
    valid_urls = {(r.get("url") or "").strip() for r in raw}
    sources = [
        Source(
            url=a.url,
            title=a.title,
            retrieved_at=retrieved_at,
            snippet=a.snippet[:_SNIPPET_CHARS],
            credibility_note=a.credibility_note,
        )
        for a in result.sources
        if a.url in valid_urls
    ]

    # Constrain fact citations to URLs we actually retrieved.
    untagged = 0
    for fact in result.facts:
        fact.source_urls = [u for u in fact.source_urls if u in valid_urls]
        if not fact.source_urls:
            untagged += 1
    if untagged:
        warnings.append(f"{untagged} fact(s) could not be tied to a retrieved source.")
    if not sources:
        warnings.append("No usable sources survived validation; fact sheet may be thin.")

    fact_sheet = _render_fact_sheet(topic, sources, result.facts)
    return {"sources": sources, "fact_sheet": fact_sheet, "warnings": warnings}
