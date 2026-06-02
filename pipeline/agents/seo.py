"""seo_agent — score the draft's SEO and return structured feedback.

Input  (state): ``draft``, ``target_keyword``
Output (state update): ``seo_score`` (0-100), ``seo_feedback`` (dict), ``warnings``

Objective metrics are computed in code so they are honest and reproducible:
- keyword density vs the target keyword,
- heading structure (H1/H2/H3 counts),
- readability via ``textstat`` Flesch reading ease.
These metrics are handed to the LLM, which returns a structured judgement (issues, a written
meta description, suggestions) and an overall 0-100 score. The deterministic readability value
overrides whatever the model reports, so the number always matches ``textstat``.
"""

from __future__ import annotations

import re

import textstat
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pipeline import config, llm
from pipeline.state import PipelineState

_MD_MARKUP = re.compile(r"(^#{1,6}\s+)|([*_`>])|(\[(.*?)\]\(.*?\))", re.MULTILINE)


class _SeoFeedback(BaseModel):
    keyword_issues: str = Field(..., description="Assessment of keyword usage/density vs target.")
    heading_issues: str = Field(..., description="Assessment of heading structure (H1/H2/H3).")
    meta_description: str = Field(..., description="A compelling <=160-char meta description with the keyword.")
    readability_score: float = Field(..., description="Flesch reading ease (overwritten with the computed value).")
    suggestions: list[str] = Field(default_factory=list, description="Concrete, prioritized SEO fixes.")


class _SeoResult(BaseModel):
    seo_score: int = Field(..., description="Overall SEO quality 0-100.")
    feedback: _SeoFeedback


# Deterministic mock-mode response: a clean, passing score.
llm.MOCK_RESPONSES["seo"] = {
    "seo_score": 82,
    "feedback": {
        "keyword_issues": "Keyword usage looks balanced.",
        "heading_issues": "Heading structure is clear.",
        "meta_description": "A concise mock meta description about the topic.",
        "readability_score": 55.0,
        "suggestions": ["Mock suggestion: add an FAQ section."],
    },
}

_SYSTEM = (
    "You are an SEO analyst. Given an ARTICLE (markdown), a TARGET KEYWORD, and pre-computed "
    "METRICS, assess: (1) keyword usage/density (ideal ~0.5-2.5%; flag stuffing or under-use), "
    "(2) heading structure (exactly one H1, logical H2/H3 nesting), (3) write a compelling meta "
    "description <=160 chars containing the keyword, and (4) give prioritized suggestions. "
    "Return an overall seo_score 0-100 reflecting these factors and the provided readability."
)


def _strip_markdown(text: str) -> str:
    """Light markdown -> plain text for readability scoring."""

    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # links -> link text
    text = _MD_MARKUP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keyword_density(draft: str, keyword: str) -> tuple[int, float, int]:
    """Return (occurrences, density_pct, total_words) for the keyword phrase."""

    words = re.findall(r"\b\w+\b", draft.lower())
    total = len(words)
    kw = keyword.lower().strip()
    occurrences = len(re.findall(re.escape(kw), draft.lower())) if kw else 0
    kw_word_len = max(1, len(kw.split()))
    density = (occurrences * kw_word_len / total * 100) if total else 0.0
    return occurrences, round(density, 2), total


def _heading_counts(draft: str) -> dict[str, int]:
    return {
        "h1": len(re.findall(r"^# .+$", draft, re.MULTILINE)),
        "h2": len(re.findall(r"^## .+$", draft, re.MULTILINE)),
        "h3": len(re.findall(r"^### .+$", draft, re.MULTILINE)),
    }


def seo_agent(state: PipelineState) -> dict:
    """Score SEO and return structured feedback (LangGraph node)."""

    draft = state.get("draft", "")
    keyword = state.get("target_keyword", "")
    warnings = list(state.get("warnings", []))

    if not draft.strip():
        warnings.append("SEO analysis skipped: empty draft.")
        return {"seo_score": 0, "seo_feedback": None, "warnings": warnings}

    # --- deterministic metrics ---
    occurrences, density, total_words = _keyword_density(draft, keyword)
    headings = _heading_counts(draft)
    try:
        readability = round(float(textstat.flesch_reading_ease(_strip_markdown(draft))), 1)
    except Exception:
        readability = 0.0

    metrics = (
        f"- target keyword: {keyword!r}\n"
        f"- keyword occurrences: {occurrences}\n"
        f"- keyword density: {density}%\n"
        f"- total words: {total_words}\n"
        f"- headings: H1={headings['h1']}, H2={headings['h2']}, H3={headings['h3']}\n"
        f"- flesch reading ease: {readability}"
    )

    model = llm.get_llm("seo").with_structured_output(_SeoResult).with_retry(
        stop_after_attempt=config.LLM_MAX_RETRIES
    )
    result: _SeoResult = model.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"TARGET KEYWORD: {keyword}\n\nMETRICS:\n{metrics}\n\nARTICLE:\n{draft}"),
        ]
    )

    score = max(0, min(100, int(result.seo_score)))
    feedback = result.feedback.model_dump()
    # Keep the computed metrics authoritative.
    feedback["readability_score"] = readability
    feedback["keyword_density_pct"] = density
    feedback["keyword_occurrences"] = occurrences
    feedback["headings"] = headings

    return {"seo_score": score, "seo_feedback": feedback, "warnings": warnings}
