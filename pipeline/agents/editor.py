"""editor_agent — targeted rewrite driven by the fact-check + SEO findings.

Input  (state): ``draft``, ``flagged_claims``, ``seo_feedback``, ``fact_sheet``, ``target_keyword``
Output (state update): ``draft`` (revised), ``revision_count`` (+1), ``warnings``

The editor fixes ONLY the reported problems (remove/repair unsupported claims using facts that
exist in the fact sheet, reduce keyword stuffing, improve readability) and leaves the rest intact.
It returns a structured object — the revised draft plus human-readable change notes — so each
revision is observable. Incrementing ``revision_count`` is what the loop guard counts against.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pipeline import config, llm
from pipeline.state import FlaggedClaim, PipelineState

logger = logging.getLogger("pipeline.editor")


class _EditorOutput(BaseModel):
    """Structured editor result (so edits are loggable, not a bare string)."""

    revised_draft: str = Field(..., description="The full revised article in markdown.")
    change_notes: list[str] = Field(
        default_factory=list, description="Short notes describing each change made."
    )


# Deterministic mock-mode response.
llm.MOCK_RESPONSES["editor"] = {
    "revised_draft": (
        "# Mock Revised Article\n\n## Introduction\nA revised mock introduction.\n\n"
        "## Body\nRevised mock body text with balanced keyword usage.\n\n"
        "## Conclusion\nA revised mock conclusion.\n"
    ),
    "change_notes": ["Mock: removed an unsupported claim", "Mock: reduced keyword density"],
}

_SYSTEM = (
    "You are a senior editor. Improve the ARTICLE by addressing ONLY the listed problems; keep "
    "everything else intact.\n"
    "- For each FLAGGED CLAIM: fix or remove it. If you rewrite it, use ONLY facts present in the "
    "FACT SHEET — never invent new facts or statistics.\n"
    "- Apply the SEO FEEDBACK: if keyword density is too high, reduce exact-match repetitions and "
    "use synonyms; improve readability by simplifying long sentences; keep one H1 and a clean H2/H3 "
    "structure.\n"
    "- Preserve the article's markdown structure, voice, and ~900-1300 word length.\n"
    "Return the FULL revised draft and a short list of change notes."
)


def _format_flagged(claims: list[FlaggedClaim]) -> str:
    if not claims:
        return "(none)"
    return "\n".join(
        f"- CLAIM: {c.claim}\n  REASON: {c.reason}\n  FIX: {c.suggested_fix}\n  SOURCE: {c.source_url or 'none'}"
        for c in claims
    )


def _format_seo(seo_feedback: dict | None) -> str:
    if not seo_feedback:
        return "(none)"
    lines = [
        f"- keyword_issues: {seo_feedback.get('keyword_issues', '')}",
        f"- heading_issues: {seo_feedback.get('heading_issues', '')}",
        f"- readability (Flesch): {seo_feedback.get('readability_score', '')}",
        f"- keyword_density_pct: {seo_feedback.get('keyword_density_pct', '')}",
    ]
    for s in seo_feedback.get("suggestions", []):
        lines.append(f"- suggestion: {s}")
    return "\n".join(lines)


def editor_agent(state: PipelineState) -> dict:
    """Rewrite only the problem sections and bump ``revision_count`` (LangGraph node)."""

    draft = state.get("draft", "")
    warnings = list(state.get("warnings", []))
    next_revision = state.get("revision_count", 0) + 1

    if not draft.strip():
        warnings.append("Editor skipped: empty draft.")
        return {"revision_count": next_revision, "warnings": warnings}

    model = llm.get_llm("editor").with_structured_output(_EditorOutput).with_retry(
        stop_after_attempt=config.LLM_MAX_RETRIES
    )
    result: _EditorOutput = model.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(
                content=(
                    f"TARGET KEYWORD: {state.get('target_keyword', '')}\n\n"
                    f"FLAGGED CLAIMS:\n{_format_flagged(state.get('flagged_claims', []))}\n\n"
                    f"SEO FEEDBACK:\n{_format_seo(state.get('seo_feedback'))}\n\n"
                    f"FACT SHEET:\n{state.get('fact_sheet', '')}\n\n"
                    f"ARTICLE:\n{draft}"
                )
            ),
        ]
    )

    logger.info(
        "editor_agent: revision %d applied (%d change(s))", next_revision, len(result.change_notes)
    )
    for note in result.change_notes:
        logger.info("  - %s", note)

    return {
        "draft": result.revised_draft.strip() or draft,
        "revision_count": next_revision,
        "warnings": warnings,
    }
