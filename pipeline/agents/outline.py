"""outline_agent (V2, async) — turn the fact sheet into a structured, source-aware outline.

Input  (state): ``fact_sheet``, ``target_keyword``, ``style_persona``
Output (state update): ``outline`` (markdown str), ``warnings``

Produces a structured ``OutlineSection`` list (section title, which facts/sources each section
draws on, estimated word count, subsections) and renders it to markdown. Lightweight node — uses
the cheap outline model. The writer consumes this outline instead of self-generating one.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pipeline import config, llm
from pipeline.state import OutlineSection, V2PipelineState

logger = logging.getLogger("pipeline.outline")


class _OutlineModel(BaseModel):
    title: str = Field(..., description="A compelling article title containing the target keyword.")
    sections: list[OutlineSection] = Field(..., description="Ordered H2 sections.")


llm.MOCK_RESPONSES["outline"] = {
    "title": "Mock Outline Title",
    "sections": [
        {"title": "Introduction", "facts_to_cover": ["mock fact"],
         "source_urls": ["https://example.com/1"], "estimated_words": 150, "subsections": []},
        {"title": "Key Benefits", "facts_to_cover": ["mock fact 2"], "source_urls": [],
         "estimated_words": 400, "subsections": ["Detail one"]},
        {"title": "Conclusion", "facts_to_cover": [], "source_urls": [],
         "estimated_words": 120, "subsections": []},
    ],
}

_SYSTEM = (
    "You are a content strategist. Given a FACT SHEET, a TARGET KEYWORD, and a STYLE PERSONA, "
    "produce a structured article outline. For each H2 section give: a title, the specific facts "
    "it should cover (drawn ONLY from the fact sheet), the source URLs backing those facts, an "
    "estimated word count, and any H3 subsections. The article title must read naturally and "
    "include the target keyword. Shape the structure to suit the persona."
)


def _render(outline: _OutlineModel) -> str:
    lines = [f"# {outline.title}", ""]
    for section in outline.sections:
        wc = f"  _(~{section.estimated_words} words)_" if section.estimated_words else ""
        lines.append(f"## {section.title}{wc}")
        for sub in section.subsections:
            lines.append(f"### {sub}")
        for fact in section.facts_to_cover:
            srcs = f"  [{', '.join(section.source_urls)}]" if section.source_urls else ""
            lines.append(f"- {fact}{srcs}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def outline_agent(state: V2PipelineState) -> dict:
    """Generate a structured, persona-aware outline from the fact sheet (async node)."""

    fact_sheet = state.get("fact_sheet", "")
    warnings = list(state.get("warnings", []))
    if not fact_sheet.strip():
        warnings.append("Outline agent received an empty fact sheet; skipping outline.")
        return {"outline": "", "warnings": warnings}

    persona = state.get("style_persona", "technical deep-dive")
    keyword = state.get("target_keyword", "")
    model = llm.get_llm("outline").with_structured_output(_OutlineModel).with_retry(
        stop_after_attempt=config.LLM_MAX_RETRIES
    )
    result: _OutlineModel = await model.ainvoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(
                content=f"TARGET KEYWORD: {keyword}\nSTYLE PERSONA: {persona}\n\nFACT SHEET:\n{fact_sheet}"
            ),
        ]
    )
    logger.info("outline: %d sections for persona %r", len(result.sections), persona)
    return {"outline": _render(result), "warnings": warnings}
