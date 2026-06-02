"""writer_agent — outline, then draft, using only the researched facts.

Input  (state): ``topic``, ``target_keyword``, ``fact_sheet``
Output (state update): ``outline`` (markdown str), ``draft`` (markdown str), ``warnings``

Two LLM calls:
1. A *structured* outline (H2/H3 sections + key points) derived from the fact sheet.
2. A free-form markdown draft (~900-1300 words) that follows the outline and uses ONLY
   facts present in the fact sheet. The draft is prose, so it is returned as markdown text
   rather than a structured object (structured outputs are reserved for scores/flags).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pipeline import config, llm
from pipeline.state import PipelineState

WORD_MIN, WORD_MAX = 900, 1300


# --- structured outline schema --------------------------------------------------------
class _OutlineSection(BaseModel):
    heading: str = Field(..., description="An H2 section heading.")
    subsections: list[str] = Field(default_factory=list, description="Optional H3 subheadings.")
    key_points: list[str] = Field(
        default_factory=list, description="Facts (from the fact sheet) this section should cover."
    )


class _Outline(BaseModel):
    title: str = Field(..., description="A compelling article title containing the target keyword.")
    sections: list[_OutlineSection] = Field(..., description="Ordered H2 sections.")


# Deterministic mock-mode outputs.
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

_OUTLINE_SYSTEM = (
    "You are an expert content strategist. Given a TOPIC, a TARGET KEYWORD, and a FACT SHEET, "
    "produce a clear article outline with H2 sections (and H3 subsections where helpful). "
    "Each section lists the key points it should cover, drawn ONLY from the fact sheet. "
    "The title must read naturally and include the target keyword."
)

_DRAFT_SYSTEM = (
    "You are an expert writer. Write a complete article in MARKDOWN that follows the given OUTLINE.\n"
    "Hard rules:\n"
    f"- Length: {WORD_MIN}-{WORD_MAX} words.\n"
    "- Use ONLY facts present in the FACT SHEET. Do NOT introduce any statistic, study, date, "
    "name, or claim that is not in the fact sheet. You may rephrase and connect facts.\n"
    "- Use '#' for the title, '##' for sections, '###' for subsections.\n"
    "- Weave the TARGET KEYWORD naturally into the title, an early paragraph, and a few headings "
    "(no keyword stuffing).\n"
    "- Open with a short hook and close with a concise conclusion. Do not fabricate sources or a "
    "meta description."
)


def _render_outline(outline: _Outline) -> str:
    """Render the structured outline to a markdown string for state + display."""

    lines = [f"# {outline.title}", ""]
    for section in outline.sections:
        lines.append(f"## {section.heading}")
        for sub in section.subsections:
            lines.append(f"### {sub}")
        for point in section.key_points:
            lines.append(f"- {point}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def writer_agent(state: PipelineState) -> dict:
    """Generate an outline and a fact-grounded markdown draft (LangGraph node)."""

    topic = state["topic"]
    target_keyword = state["target_keyword"]
    fact_sheet = state.get("fact_sheet", "")
    warnings = list(state.get("warnings", []))

    if not fact_sheet.strip():
        warnings.append("Writer received an empty fact sheet; skipping draft.")
        return {"outline": "", "draft": "", "warnings": warnings}

    # 1) Structured outline.
    outline_model = llm.get_llm("writer").with_structured_output(_Outline).with_retry(
        stop_after_attempt=config.LLM_MAX_RETRIES
    )
    outline: _Outline = outline_model.invoke(
        [
            SystemMessage(content=_OUTLINE_SYSTEM),
            HumanMessage(
                content=f"TOPIC: {topic}\nTARGET KEYWORD: {target_keyword}\n\nFACT SHEET:\n{fact_sheet}"
            ),
        ]
    )
    outline_md = _render_outline(outline)

    # 2) Full markdown draft (prose -> plain text invoke).
    draft_model = llm.get_llm("writer").with_retry(stop_after_attempt=config.LLM_MAX_RETRIES)
    response = draft_model.invoke(
        [
            SystemMessage(content=_DRAFT_SYSTEM),
            HumanMessage(
                content=(
                    f"TARGET KEYWORD: {target_keyword}\n\nOUTLINE:\n{outline_md}\n\n"
                    f"FACT SHEET:\n{fact_sheet}"
                )
            ),
        ]
    )
    draft = (response.content if hasattr(response, "content") else str(response)).strip()

    word_count = len(draft.split())
    if not draft:
        warnings.append("Writer produced an empty draft.")
    elif not (WORD_MIN <= word_count <= WORD_MAX):
        warnings.append(
            f"Draft length {word_count} words is outside the {WORD_MIN}-{WORD_MAX} target."
        )

    return {"outline": outline_md, "draft": draft, "warnings": warnings}
