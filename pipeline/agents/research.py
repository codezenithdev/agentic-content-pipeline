"""research_agent (V2, async) — deep research into a sourced, credibility-scored fact sheet.

Input  (state): ``topic``, ``target_keyword``, and optionally ``memory_hit`` / ``reused_sources``
Output (state update): ``sources``, ``fact_sheet`` (markdown with [HIGH/MEDIUM/LOW] tags),
                       ``full_page_chunks`` (RAG chunks), ``credibility_scores``, ``warnings``

Flow:
1. Source discovery — reuse memory sources on a hit, else Tavily search (5-8 URLs).
2. Deep read — fetch the full page HTML for the top ``PAGE_FETCH_TOP_N`` URLs (httpx, async),
   extract the main article text (BeautifulSoup), chunk it into ~500-token ``PageChunk`` pieces.
3. Structured extraction — the LLM assesses each source and extracts source-tagged facts, now
   grounded in the richer full-page excerpts (not just snippets).
4. Credibility scoring — per source: domain authority + recency + corroboration -> 0..1; sources
   below ``CREDIBILITY_MIN`` are flagged. Each fact gets a [HIGH/MEDIUM/LOW] confidence tag.

Degrades gracefully: search failure, page-fetch failure, or empty results add warnings instead
of crashing. Async so the V2 graph can stream; V1's graph drives it through a sync adapter.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pipeline import config, llm
from pipeline.state import PageChunk, Source, V2PipelineState

logger = logging.getLogger("pipeline.research")

_SNIPPET_CHARS = 500
_EXCERPT_CHARS = 1200          # full-page context fed to the model per source
_MAX_CHUNKS_PER_PAGE = 8

# Curated high-authority domains (extend freely; heuristic only).
_HIGH_AUTHORITY = {
    "nih.gov", "ncbi.nlm.nih.gov", "who.int", "cdc.gov", "nature.com", "science.org",
    "nejm.org", "thelancet.com", "harvard.edu", "health.harvard.edu", "mayoclinic.org",
    "stanford.edu", "mit.edu", "reuters.com", "apnews.com", "bbc.com", "nasa.gov",
    "ucdavis.edu", "hopkinsmedicine.org",
}
_TLD_SCORES = ((".gov", 1.0), (".edu", 0.95), (".org", 0.7))


# --- structured output schemas (LLM fills these) --------------------------------------
class _SourceAssessment(BaseModel):
    url: str = Field(..., description="Must be one of the provided source URLs.")
    title: str
    snippet: str = Field(..., description="Short excerpt that supports the cited facts.")
    credibility_note: str = Field(..., description="One line on trustworthiness/relevance.")


class _FactItem(BaseModel):
    fact: str = Field(..., description="A single, verifiable factual statement.")
    source_urls: list[str] = Field(..., description="URLs (from the provided list) supporting it.")


class _ResearchOutput(BaseModel):
    sources: list[_SourceAssessment]
    facts: list[_FactItem]


# Deterministic mock-mode response (consistent with llm.MockSearchClient URLs).
llm.MOCK_RESPONSES["research"] = {
    "sources": [
        {"url": "https://example.com/1", "title": "Mock source 1",
         "snippet": "Mock evidence about the topic.", "credibility_note": "Illustrative mock source."},
        {"url": "https://example.com/2", "title": "Mock source 2",
         "snippet": "More mock evidence about the topic.", "credibility_note": "Illustrative mock source."},
    ],
    "facts": [
        {"fact": "This is a mock fact grounded in the sources.", "source_urls": ["https://example.com/1"]},
        {"fact": "A second mock fact.", "source_urls": ["https://example.com/1", "https://example.com/2"]},
    ],
}

_SYSTEM = (
    "You are a meticulous research analyst. You are given a topic and a numbered list of real "
    "web sources (title, URL, and an excerpt drawn from the full page where available).\n"
    "1. For EACH source, write a one-line credibility note (publisher, reliability/relevance).\n"
    "2. Extract 8-14 specific, verifiable facts. Tag each fact with the URL(s) from the list that "
    "support it.\n"
    "Rules: Use ONLY the provided sources. Never invent facts or URLs. Prefer concrete facts "
    "(numbers, dates, named entities)."
)


class _SearchError(RuntimeError):
    """Internal: Tavily call failed (surfaced as a warning, not a crash)."""


def _tavily_search(topic: str) -> list[dict]:
    client = llm.get_search_client()
    try:
        response = client.search(
            topic, max_results=config.TAVILY_MAX_RESULTS, search_depth=config.TAVILY_SEARCH_DEPTH
        )
    except Exception as exc:
        raise _SearchError(str(exc)) from exc
    return list(response.get("results") or [])


# --- full-page fetch + chunking -------------------------------------------------------
def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    return re.sub(r"\s+", " ", main.get_text(separator=" ", strip=True))


def _parse_published_time(html: str) -> datetime | None:
    soup = BeautifulSoup(html, "html.parser")
    for key, attr in (("article:published_time", "property"), ("og:published_time", "property"),
                      ("date", "name"), ("publish-date", "name")):
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            try:
                dt = datetime.fromisoformat(tag["content"].replace("Z", "+00:00"))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _chunk_text(url: str, text: str) -> list[PageChunk]:
    words = text.split()
    if not words:
        return []
    size = max(50, int(config.CHUNK_SIZE_TOKENS * 0.75))  # ~words per ~500 tokens
    chunks: list[PageChunk] = []
    for start in range(0, len(words), size):
        if len(chunks) >= _MAX_CHUNKS_PER_PAGE:
            break
        chunk_words = words[start:start + size]
        if len(chunk_words) < 20 and chunks:  # fold a tiny tail into the previous chunk
            prev = chunks[-1]
            prev.chunk_text = f"{prev.chunk_text} {' '.join(chunk_words)}"
            break
        chunks.append(PageChunk(url=url, chunk_text=" ".join(chunk_words),
                                chunk_index=len(chunks), relevance_score=0.0))
    return chunks


async def _fetch_and_chunk(url: str) -> tuple[list[PageChunk], datetime | None]:
    html = await llm.fetch_page_html(url)
    return _chunk_text(url, _extract_main_text(html)), _parse_published_time(html)


# --- credibility heuristics -----------------------------------------------------------
def _domain_authority(url: str) -> float:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if any(host == d or host.endswith("." + d) for d in _HIGH_AUTHORITY):
        return 0.95
    for tld, score in _TLD_SCORES:
        if host.endswith(tld):
            return score
    return 0.5


def _recency_score(dt: datetime | None) -> float:
    if not dt:
        return 0.5
    days = max(0, (datetime.now(timezone.utc) - dt).days)
    if days <= 365:
        return 1.0
    if days <= 365 * 3:
        return 0.75
    if days <= 365 * 6:
        return 0.5
    return 0.3


def _corroboration(url: str, facts: list[_FactItem]) -> float:
    cited = [f for f in facts if url in f.source_urls]
    if not cited:
        return 0.3
    multi = sum(1 for f in cited if len(f.source_urls) >= 2)
    return round(min(1.0, 0.4 + 0.6 * multi / len(cited)), 3)


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}


def _relevance(text: str, terms: set[str]) -> float:
    if not terms:
        return 0.0
    low = text.lower()
    return round(sum(1 for t in terms if t in low) / len(terms), 3)


def _confidence_tag(fact: _FactItem, credibility: dict[str, float]) -> str:
    if not fact.source_urls:
        return "LOW"
    best = max((credibility.get(u, 0.0) for u in fact.source_urls), default=0.0)
    if best >= 0.7 or len(fact.source_urls) >= 2:
        return "HIGH"
    if best >= 0.5:
        return "MEDIUM"
    return "LOW"


def _render_fact_sheet(topic, sources, facts, credibility) -> str:
    lines = [f"# Fact Sheet: {topic}", "", "## Facts"]
    for fact in facts:
        tag = _confidence_tag(fact, credibility)
        tags = ", ".join(fact.source_urls) if fact.source_urls else "unverified"
        lines.append(f"- [{tag}] {fact.fact}  _[source: {tags}]_")
    lines += ["", "## Sources"]
    for i, src in enumerate(sources, 1):
        cred = credibility.get(src.url)
        suffix = f" (credibility {cred:.2f})" if cred is not None else ""
        lines.append(f"{i}. [{src.title}]({src.url}){suffix} — {src.credibility_note}")
    return "\n".join(lines)


async def research_agent(state: V2PipelineState) -> dict:
    """Deep-research the topic into a credibility-scored, RAG-backed fact sheet (async node)."""

    topic = state["topic"]
    keyword = state.get("target_keyword", "")
    warnings = list(state.get("warnings", []))

    # 1) Source discovery — reuse memory on a hit, else search.
    if state.get("memory_hit") and state.get("reused_sources"):
        reused = state["reused_sources"]
        raw = [{"title": s.title, "url": s.url, "content": s.snippet, "score": 1.0} for s in reused]
        logger.info("research: reusing %d sources from memory (skipping web search)", len(raw))
    else:
        try:
            raw = await asyncio.to_thread(_tavily_search, topic)
        except _SearchError as exc:
            warnings.append(f"Web search failed for topic '{topic}': {exc}")
            return {"sources": [], "fact_sheet": "", "full_page_chunks": [],
                    "credibility_scores": {}, "warnings": warnings}
        if not raw:
            warnings.append(f"Web search returned no sources for topic '{topic}'.")
            return {"sources": [], "fact_sheet": "", "full_page_chunks": [],
                    "credibility_scores": {}, "warnings": warnings}

    # 2) Deep read — fetch + chunk the top-N pages concurrently.
    top = sorted(raw, key=lambda r: r.get("score") or 0.0, reverse=True)[: config.PAGE_FETCH_TOP_N]
    fetched = await asyncio.gather(
        *[_fetch_and_chunk(r.get("url", "")) for r in top], return_exceptions=True
    )
    chunks: list[PageChunk] = []
    published: dict[str, datetime] = {}
    for r, result in zip(top, fetched):
        url = r.get("url", "")
        if isinstance(result, Exception):
            warnings.append(f"Could not fetch full page for {url}: {result}")
            continue
        page_chunks, pub = result
        chunks.extend(page_chunks)
        if pub:
            published[url] = pub

    terms = _terms(f"{topic} {keyword}")
    for chunk in chunks:
        chunk.relevance_score = _relevance(chunk.chunk_text, terms)

    # 3) Structured extraction — grounded in full-page excerpts where available.
    chunk_text_by_url: dict[str, str] = {}
    for chunk in chunks:
        chunk_text_by_url[chunk.url] = (chunk_text_by_url.get(chunk.url, "") + " " + chunk.chunk_text).strip()
    catalog = "\n".join(
        f"[{i}] {r.get('title', 'Untitled')}\n    URL: {r.get('url', '')}\n"
        f"    Excerpt: {(chunk_text_by_url.get(r.get('url', '')) or (r.get('content') or ''))[:_EXCERPT_CHARS]}"
        for i, r in enumerate(raw, 1)
    )
    model = llm.get_llm("research").with_structured_output(_ResearchOutput).with_retry(
        stop_after_attempt=config.LLM_MAX_RETRIES
    )
    result: _ResearchOutput = await model.ainvoke(
        [SystemMessage(content=_SYSTEM), HumanMessage(content=f"TOPIC: {topic}\n\nSOURCES:\n{catalog}")]
    )

    # 4) Build typed sources + constrain citations to retrieved URLs.
    retrieved_at = datetime.now(timezone.utc).isoformat()
    valid_urls = {(r.get("url") or "").strip() for r in raw}
    sources = [
        Source(url=a.url, title=a.title, retrieved_at=retrieved_at,
               snippet=a.snippet[:_SNIPPET_CHARS], credibility_note=a.credibility_note)
        for a in result.sources if a.url in valid_urls
    ]
    untagged = 0
    for fact in result.facts:
        fact.source_urls = [u for u in fact.source_urls if u in valid_urls]
        if not fact.source_urls:
            untagged += 1
    if untagged:
        warnings.append(f"{untagged} fact(s) could not be tied to a retrieved source.")
    if not sources:
        warnings.append("No usable sources survived validation; fact sheet may be thin.")

    # 5) Credibility scoring (domain authority + recency + corroboration).
    credibility_scores: dict[str, float] = {}
    for src in sources:
        score = round(
            0.4 * _domain_authority(src.url)
            + 0.3 * _recency_score(published.get(src.url))
            + 0.3 * _corroboration(src.url, result.facts),
            3,
        )
        credibility_scores[src.url] = score
        if score < config.CREDIBILITY_MIN:
            warnings.append(f"Low-credibility source flagged: {src.url} (score {score:.2f}).")

    fact_sheet = _render_fact_sheet(topic, sources, result.facts, credibility_scores)
    logger.info(
        "research: %d sources, %d chunks, %d facts (avg credibility %.2f)",
        len(sources), len(chunks), len(result.facts),
        sum(credibility_scores.values()) / len(credibility_scores) if credibility_scores else 0.0,
    )
    return {
        "sources": sources,
        "fact_sheet": fact_sheet,
        "full_page_chunks": chunks,
        "credibility_scores": credibility_scores,
        "warnings": warnings,
    }
