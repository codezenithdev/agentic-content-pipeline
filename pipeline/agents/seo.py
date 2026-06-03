"""seo_agent (V2, async) — SEO scoring with NLP checks + prioritized fixes.

Input  (state): ``draft``, ``target_keyword``
Output (state update): ``seo_score`` (0-100), ``seo_feedback`` (dict), ``warnings``

Computes objective metrics in code (keyword density, heading structure, textstat readability,
title-tag keyword presence, entity repetition), then the LLM returns structured feedback split
into ``fix_now`` (passed to the editor) and ``fix_later`` (deferred), plus internal-link ideas.
Deterministic metrics override the model's so the numbers are always honest. Async.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import textstat
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pipeline import config, llm
from pipeline.state import V2PipelineState

logger = logging.getLogger("pipeline.seo")

_MD_MARKUP = re.compile(r"(^#{1,6}\s+)|([*_`>])|(\[(.*?)\]\(.*?\))", re.MULTILINE)


class _SeoFeedback(BaseModel):
    keyword_issues: str
    heading_issues: str
    meta_description: str = Field(..., description="Compelling <=160-char meta description with the keyword.")
    readability_score: float = Field(default=0.0, description="Overwritten with the computed value.")
    entity_density: str = Field(default="", description="Assessment of whether key entities recur enough.")
    title_tag: str = Field(default="", description="Overwritten in code from the H1 check.")
    internal_link_opportunities: list[str] = Field(default_factory=list)
    fix_now: list[str] = Field(default_factory=list, description="High-priority fixes for the editor.")
    fix_later: list[str] = Field(default_factory=list, description="Deferred, nice-to-have fixes.")
    suggestions: list[str] = Field(default_factory=list, description="Back-compat: all suggestions.")


class _SeoResult(BaseModel):
    seo_score: int = Field(..., description="Overall SEO quality 0-100.")
    feedback: _SeoFeedback


llm.MOCK_RESPONSES["seo"] = {
    "seo_score": 82,
    "feedback": {
        "keyword_issues": "Keyword usage looks balanced.",
        "heading_issues": "Heading structure is clear.",
        "meta_description": "A concise mock meta description about the topic.",
        "readability_score": 55.0,
        "entity_density": "Key entities recur appropriately.",
        "title_tag": "OK",
        "internal_link_opportunities": ["Link the benefits section to a methodology page."],
        "fix_now": ["Tighten the introduction."],
        "fix_later": ["Add an FAQ section."],
        "suggestions": ["Tighten the introduction.", "Add an FAQ section."],
    },
}

_SYSTEM = (
    "You are an SEO analyst. Given an ARTICLE (markdown), a TARGET KEYWORD, and pre-computed "
    "METRICS, assess keyword usage, heading structure, entity density (do key entities recur "
    "enough to signal topical focus?), and internal-link opportunities, and write a <=160-char "
    "meta description containing the keyword. Split your recommendations into fix_now (must-fix "
    "now, will be sent to the editor) and fix_later (deferred). Return an overall seo_score 0-100."
)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = _MD_MARKUP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keyword_density(draft: str, keyword: str) -> tuple[int, float, int]:
    words = re.findall(r"\b\w+\b", draft.lower())
    total = len(words)
    kw = keyword.lower().strip()
    occurrences = len(re.findall(re.escape(kw), draft.lower())) if kw else 0
    density = (occurrences * max(1, len(kw.split())) / total * 100) if total else 0.0
    return occurrences, round(density, 2), total


def _heading_counts(draft: str) -> dict[str, int]:
    return {
        "h1": len(re.findall(r"^# .+$", draft, re.MULTILINE)),
        "h2": len(re.findall(r"^## .+$", draft, re.MULTILINE)),
        "h3": len(re.findall(r"^### .+$", draft, re.MULTILINE)),
    }


def _title_has_keyword(draft: str, keyword: str) -> bool:
    for line in draft.splitlines():
        if line.strip().startswith("# "):
            return keyword.lower().strip() in line.lower()
    return False


def _entity_repetition(draft: str) -> dict[str, int]:
    """Crude named-entity proxy: capitalized words (len>3) that recur."""

    words = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", draft)
    return {w: c for w, c in Counter(words).items() if c >= 2}


async def seo_agent(state: V2PipelineState) -> dict:
    """Score SEO with NLP checks and a fix_now/fix_later split (async node)."""

    draft = state.get("draft", "")
    keyword = state.get("target_keyword", "")
    warnings = list(state.get("warnings", []))
    if not draft.strip():
        warnings.append("SEO analysis skipped: empty draft.")
        return {"seo_score": 0, "seo_feedback": None, "warnings": warnings}

    occurrences, density, total_words = _keyword_density(draft, keyword)
    headings = _heading_counts(draft)
    title_ok = _title_has_keyword(draft, keyword)
    entities = _entity_repetition(draft)
    try:
        readability = round(float(textstat.flesch_reading_ease(_strip_markdown(draft))), 1)
    except Exception:
        readability = 0.0

    top_entities = ", ".join(f"{w}×{c}" for w, c in sorted(entities.items(), key=lambda x: -x[1])[:6]) or "none"
    metrics = (
        f"- target keyword: {keyword!r}\n"
        f"- keyword occurrences: {occurrences} (density {density}%)\n"
        f"- total words: {total_words}\n"
        f"- headings: H1={headings['h1']}, H2={headings['h2']}, H3={headings['h3']}\n"
        f"- H1 contains keyword: {title_ok}\n"
        f"- recurring entities: {top_entities}\n"
        f"- flesch reading ease: {readability}"
    )

    model = llm.get_llm("seo").with_structured_output(_SeoResult).with_retry(
        stop_after_attempt=config.LLM_MAX_RETRIES
    )
    result: _SeoResult = await model.ainvoke(
        [SystemMessage(content=_SYSTEM),
         HumanMessage(content=f"TARGET KEYWORD: {keyword}\n\nMETRICS:\n{metrics}\n\nARTICLE:\n{draft}")]
    )

    score = max(0, min(100, int(result.seo_score)))
    feedback = result.feedback.model_dump()
    # Keep computed metrics authoritative.
    feedback["readability_score"] = readability
    feedback["keyword_density_pct"] = density
    feedback["keyword_occurrences"] = occurrences
    feedback["headings"] = headings
    feedback["title_tag"] = (
        "OK: the H1 contains the target keyword" if title_ok
        else "Fix: the H1 does not contain the target keyword"
    )
    feedback["title_tag_ok"] = title_ok
    feedback["entity_repetition"] = entities
    if not title_ok and "H1" not in " ".join(feedback.get("fix_now", [])):
        feedback.setdefault("fix_now", []).append("Add the target keyword to the H1 title.")
    if not feedback.get("suggestions"):
        feedback["suggestions"] = list(feedback.get("fix_now", [])) + list(feedback.get("fix_later", []))

    logger.info("seo: score=%d (density %.2f%%, readability %.1f, title_kw=%s, %d fix_now)",
                score, density, readability, title_ok, len(feedback.get("fix_now", [])))
    return {"seo_score": score, "seo_feedback": feedback, "warnings": warnings}
