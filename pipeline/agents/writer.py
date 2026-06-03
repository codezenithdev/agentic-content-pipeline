"""writer_agent (V2, async) — persona-shaped drafting with a self-critique pass.

Input  (state): ``fact_sheet``, ``outline`` (from outline_agent), ``target_keyword``, ``style_persona``
Output (state update): ``outline``, ``draft`` (markdown), ``self_critique``, ``warnings``

Pipeline: use the provided outline (or self-generate one if absent, preserving the V1 flow where
no outline_agent runs) -> draft in the chosen persona's voice -> harshly self-critique the draft
-> revise to address the critique. Uses the strong writer model. Facts come ONLY from the fact
sheet. Async so the V2 graph can stream; V1's graph drives it via a sync adapter.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pipeline import config, llm
from pipeline.state import V2PipelineState

logger = logging.getLogger("pipeline.writer")

WORD_MIN, WORD_MAX = 900, 1300

_PERSONA_GUIDE = {
    "technical deep-dive": "Precise and rigorous. Assume an expert reader. Use exact figures, "
    "mechanisms, and correct terminology. No hand-holding or filler.",
    "beginner-friendly": "Warm and accessible. Define terms, use analogies, keep sentences short. "
    "Assume no prior knowledge.",
    "op-ed": "Persuasive with a clear point of view. First person is allowed. Argue a thesis and "
    "marshal the facts to support it.",
}


# --- self-generated outline (V1 fallback when no outline_agent ran) -------------------
class _OutlineSection(BaseModel):
    heading: str
    subsections: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)


class _Outline(BaseModel):
    title: str
    sections: list[_OutlineSection]


llm.MOCK_RESPONSES["writer"] = {
    "title": "Mock Article Title",
    "sections": [
        {"heading": "Introduction", "subsections": [], "key_points": ["Mock point A"]},
        {"heading": "Key Benefits", "subsections": ["Detail one"], "key_points": ["Mock point B"]},
        {"heading": "Conclusion", "subsections": [], "key_points": ["Mock wrap-up"]},
    ],
}
llm.MOCK_TEXT["writer"] = (
    "# Mock Article Title\n\n"
    "## Introduction\nThis is a mock introduction grounded in the mock fact sheet.\n\n"
    "## Key Benefits\nThis is mock body text about the topic.\n\n"
    "### Detail one\nMore mock detail.\n\n"
    "## Conclusion\nA mock conclusion.\n"
)


def _persona(state: V2PipelineState) -> str:
    return _PERSONA_GUIDE.get(state.get("style_persona", ""), _PERSONA_GUIDE["technical deep-dive"])


def _render_outline(outline: _Outline) -> str:
    lines = [f"# {outline.title}", ""]
    for section in outline.sections:
        lines.append(f"## {section.heading}")
        for sub in section.subsections:
            lines.append(f"### {sub}")
        for point in section.key_points:
            lines.append(f"- {point}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def _self_outline(state: V2PipelineState) -> str:
    model = llm.get_llm("writer").with_structured_output(_Outline).with_retry(
        stop_after_attempt=config.LLM_MAX_RETRIES
    )
    result: _Outline = await model.ainvoke(
        [
            SystemMessage(content="Produce an H2/H3 outline (title + sections + key points) drawn "
                          "only from the fact sheet. The title must include the target keyword."),
            HumanMessage(content=f"TARGET KEYWORD: {state.get('target_keyword','')}\n\n"
                         f"FACT SHEET:\n{state.get('fact_sheet','')}"),
        ]
    )
    return _render_outline(result)


async def _text(role: str, system: str, human: str) -> str:
    model = llm.get_llm(role).with_retry(stop_after_attempt=config.LLM_MAX_RETRIES)
    response = await model.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
    return (response.content if hasattr(response, "content") else str(response)).strip()


async def writer_agent(state: V2PipelineState) -> dict:
    """Draft the article in persona voice, then self-critique and revise (async node)."""

    fact_sheet = state.get("fact_sheet", "")
    warnings = list(state.get("warnings", []))
    if not fact_sheet.strip():
        warnings.append("Writer received an empty fact sheet; skipping draft.")
        return {"outline": "", "draft": "", "self_critique": "", "warnings": warnings}

    keyword = state.get("target_keyword", "")
    persona = _persona(state)
    outline = (state.get("outline") or "").strip() or await _self_outline(state)

    # 1) First draft in persona voice.
    draft = await _text(
        "writer",
        "You are an expert writer. Write a complete article in MARKDOWN that follows the OUTLINE.\n"
        f"Voice/persona: {persona}\n"
        f"Length: {WORD_MIN}-{WORD_MAX} words. Use '#' for the title, '##'/'###' for sections.\n"
        "Use ONLY facts present in the FACT SHEET — never invent statistics, studies, or names. "
        "Weave the TARGET KEYWORD in naturally (no stuffing).",
        f"TARGET KEYWORD: {keyword}\n\nOUTLINE:\n{outline}\n\nFACT SHEET:\n{fact_sheet}",
    )

    # 2) Harsh self-critique.
    critique = await _text(
        "writer",
        "You are a ruthless editor. Critique the draft for logical gaps, unsupported or vague "
        "claims, weak sections, and persona mismatches. Be specific and harsh. Bullet points.",
        f"PERSONA: {persona}\n\nFACT SHEET:\n{fact_sheet}\n\nDRAFT:\n{draft}",
    )

    # 3) Revise to address the critique (still facts-only).
    revised = await _text(
        "writer",
        "Revise the DRAFT to fix every issue in the CRITIQUE while preserving what works. "
        f"Keep the persona ({persona}), markdown structure, and {WORD_MIN}-{WORD_MAX} word target. "
        "Use ONLY fact-sheet facts. Return the full revised article.",
        f"TARGET KEYWORD: {keyword}\n\nCRITIQUE:\n{critique}\n\nFACT SHEET:\n{fact_sheet}\n\nDRAFT:\n{draft}",
    )
    revised = revised or draft

    word_count = len(revised.split())
    if not revised:
        warnings.append("Writer produced an empty draft.")
    elif not (WORD_MIN <= word_count <= WORD_MAX):
        warnings.append(f"Draft length {word_count} words is outside the {WORD_MIN}-{WORD_MAX} target.")

    logger.info("writer: drafted %d words (persona=%r, critique=%d chars)",
                word_count, state.get("style_persona", ""), len(critique))
    return {"outline": outline, "draft": revised, "self_critique": critique, "warnings": warnings}
