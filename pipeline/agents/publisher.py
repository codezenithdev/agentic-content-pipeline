"""publisher_agent — the human-in-the-loop publication gate.

Input  (state): final ``draft`` + scores + ``sources`` + ``revision_count``
Output (state update): ``approved``, ``warnings`` (and, in M6, the exported files)

The graph is compiled with ``interrupt_before=["publisher_agent"]``, so execution PAUSES
before this node until a human resumes it (Streamlit "Approve & Publish" button or a CLI
yes/no prompt). Reaching this node therefore means a human approved publication.

NOTE: this is the M5 version — it handles the loop-guard warning and marks approval. The
``.md`` / ``.html`` / ``.meta.json`` export is implemented in M6.
"""

from __future__ import annotations

import logging

from pipeline import config
from pipeline.state import PipelineState

logger = logging.getLogger("pipeline.publisher")


def publisher_agent(state: PipelineState) -> dict:
    """Finalize the article after human approval (LangGraph node)."""

    warnings = list(state.get("warnings", []))
    revision_count = state.get("revision_count", 0)
    fact_check_score = state.get("fact_check_score") or 0.0
    seo_score = state.get("seo_score") or 0

    capped = revision_count >= config.MAX_REVISIONS and (
        fact_check_score < config.FACT_CHECK_THRESHOLD or seo_score < config.SEO_THRESHOLD
    )
    if capped:
        msg = (
            f"Max revisions reached ({revision_count}/{config.MAX_REVISIONS}) — needs human review: "
            f"fact_check={fact_check_score:.2f} (<{config.FACT_CHECK_THRESHOLD}) or "
            f"seo={seo_score} (<{config.SEO_THRESHOLD})."
        )
        warnings.append(msg)
        logger.warning(msg)

    logger.info("publisher_agent: approved and finalized (revision %d).", revision_count)
    return {"approved": True, "warnings": warnings}
