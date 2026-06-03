"""editor_agent (V2, async) — targeted rewrites + a scored RevisionRound.

Input  (state): ``draft``, ``flagged_claims_trace``, ``seo_feedback.fix_now``, ``style_persona``,
                ``fact_check_score``, ``seo_score``, ``revision_count``
Output (state update): ``draft`` (revised), ``revision_count`` (+1), ``revision_history``, ``warnings``

Rewrites ONLY the sections implicated by the flagged claims or the SEO ``fix_now`` list (not a full
rewrite), then computes a :class:`RevisionRound` via ``difflib``: which H2 sections changed and an
``edit_distance`` (1 - SequenceMatcher ratio). The router uses that distance as a stall gate. Async.
"""

from __future__ import annotations

import difflib
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from pipeline import config, llm
from pipeline.state import RevisionRound, V2PipelineState

logger = logging.getLogger("pipeline.editor")

llm.MOCK_TEXT["editor"] = (
    "# Mock Article Title\n\n"
    "## Introduction\nA revised mock introduction grounded in the fact sheet.\n\n"
    "## Key Benefits\nRevised mock body text with balanced keyword usage.\n\n"
    "## Conclusion\nA revised mock conclusion.\n"
)


def _split_sections(md: str) -> dict[str, str]:
    """Map H2 heading -> section body (with a synthetic '_preamble_' for text before the first H2)."""

    sections: dict[str, str] = {}
    current, buf = "_preamble_", []
    for line in md.splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(buf)
            current, buf = line[3:].strip(), []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf)
    return sections


def _changed_sections(old: str, new: str) -> list[str]:
    old_s, new_s = _split_sections(old), _split_sections(new)
    changed = [h for h, body in new_s.items() if old_s.get(h) != body and h != "_preamble_"]
    changed += [f"{h} (removed)" for h in old_s if h not in new_s and h != "_preamble_"]
    return changed


async def editor_agent(state: V2PipelineState) -> dict:
    """Apply targeted fixes and record a scored revision round (async node)."""

    draft = state.get("draft", "")
    warnings = list(state.get("warnings", []))
    next_revision = state.get("revision_count", 0) + 1
    if not draft.strip():
        warnings.append("Editor skipped: empty draft.")
        return {"revision_count": next_revision, "warnings": warnings}

    persona = state.get("style_persona", "technical deep-dive")
    flagged = [t for t in state.get("flagged_claims_trace", []) if t.status.lower() != "verified"]
    fix_now = (state.get("seo_feedback") or {}).get("fix_now", [])
    flagged_block = "\n".join(f"- {t.claim}  (source: {t.source_url or 'none'})" for t in flagged) or "(none)"
    fix_block = "\n".join(f"- {item}" for item in fix_now) or "(none)"
    human_feedback = (state.get("human_feedback") or "").strip()
    feedback_block = f"\n\nHUMAN REVIEWER FEEDBACK (address this directly):\n{human_feedback}" if human_feedback else ""

    model = llm.get_llm("editor").with_retry(stop_after_attempt=config.LLM_MAX_RETRIES)
    response = await model.ainvoke(
        [
            SystemMessage(content=(
                "You are a senior editor. Rewrite ONLY the sections needed to fix the FLAGGED CLAIMS "
                "(repair or remove them using ONLY facts in the FACT SHEET) and the SEO FIX_NOW items. "
                f"Leave every other section untouched. Keep the persona ({persona}), the markdown "
                "structure, and the length. Return the FULL revised article.")),
            HumanMessage(content=(
                f"FLAGGED CLAIMS:\n{flagged_block}\n\nSEO FIX_NOW:\n{fix_block}{feedback_block}\n\n"
                f"FACT SHEET:\n{state.get('fact_sheet', '')}\n\nARTICLE:\n{draft}")),
        ]
    )
    revised = (response.content if hasattr(response, "content") else str(response)).strip() or draft

    ratio = difflib.SequenceMatcher(None, draft, revised).ratio()
    edit_distance = round(1.0 - ratio, 4)
    changed = _changed_sections(draft, revised)
    total_sections = max(1, len(_split_sections(draft)) - 1)
    revision_round = RevisionRound(
        round_number=next_revision,
        fact_check_score=float(state.get("fact_check_score") or 0.0),
        seo_score=float(state.get("seo_score") or 0),
        diff_summary=f"{len(changed)} of {total_sections} section(s) changed (edit distance {edit_distance:.3f})",
        sections_changed=changed,
        edit_distance=edit_distance,
    )
    history = list(state.get("revision_history", [])) + [revision_round]
    logger.info("editor: revision %d | edit_distance=%.4f | %d section(s) changed",
                next_revision, edit_distance, len(changed))
    return {
        "draft": revised,
        "revision_count": next_revision,
        "revision_history": history,
        "warnings": warnings,
    }
