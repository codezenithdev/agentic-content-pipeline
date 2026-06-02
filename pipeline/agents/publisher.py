"""publisher_agent — the human-in-the-loop publication gate + Medium-ready export.

Input  (state): final ``draft`` + scores + ``sources`` + ``revision_count`` + ``seo_feedback``
Output (state update): ``approved``, ``title``, ``meta_description``, ``slug``, ``output_files``, ``warnings``

The graph is compiled with ``interrupt_before=["publisher_agent"]``, so execution PAUSES before
this node until a human resumes it (Streamlit "Approve & Publish" button or the CLI yes/no
prompt). Reaching this node therefore means a human approved publication; on a rejection the
caller simply never resumes and nothing is written.

On approval it writes three files to ``OUTPUT_DIR``:
- ``<slug>.md``         clean markdown,
- ``<slug>.html``       clean semantic HTML for Medium's "Import a story" feature,
- ``<slug>.meta.json``  title, meta description, target keyword, sources, final scores, revisions.
"""

from __future__ import annotations

import html
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import markdown as md

from pipeline import config
from pipeline.state import PipelineState

logger = logging.getLogger("pipeline.publisher")

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <meta name="description" content="{description}">
</head>
<body>
<article>
{body}
</article>
</body>
</html>
"""


def _extract_title(draft: str, fallback: str) -> str:
    """Use the draft's first H1 as the title, falling back to the topic."""

    for line in draft.splitlines():
        if line.strip().startswith("# "):
            return line.strip()[2:].strip()
    return fallback.strip().title()


def _slugify(title: str) -> str:
    """Turn a title into a filesystem-safe slug."""

    slug = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:80] or "article"


def _to_html(draft: str, title: str, description: str) -> str:
    """Render markdown to a clean, semantic HTML document suitable for Medium import."""

    body = md.markdown(draft, extensions=["extra", "sane_lists"])
    return _HTML_TEMPLATE.format(
        title=html.escape(title),
        description=html.escape(description, quote=True),
        body=body,
    )


def publisher_agent(state: PipelineState) -> dict:
    """Finalize and export the article after human approval (LangGraph node)."""

    warnings = list(state.get("warnings", []))
    draft = state.get("draft", "")
    revision_count = state.get("revision_count", 0)
    fact_check_score = state.get("fact_check_score") or 0.0
    seo_score = state.get("seo_score") or 0
    seo_feedback = state.get("seo_feedback") or {}

    # Loop-guard transparency: flag if we published only because revisions were capped.
    if revision_count >= config.MAX_REVISIONS and (
        fact_check_score < config.FACT_CHECK_THRESHOLD or seo_score < config.SEO_THRESHOLD
    ):
        msg = (
            f"Max revisions reached ({revision_count}/{config.MAX_REVISIONS}) — needs human review: "
            f"fact_check={fact_check_score:.2f} (<{config.FACT_CHECK_THRESHOLD}) or "
            f"seo={seo_score} (<{config.SEO_THRESHOLD})."
        )
        warnings.append(msg)
        logger.warning(msg)

    if not draft.strip():
        warnings.append("Nothing to publish: the draft is empty.")
        return {"approved": False, "warnings": warnings}

    title = _extract_title(draft, state.get("topic", "article"))
    slug = _slugify(title)
    meta_description = (seo_feedback.get("meta_description") or "").strip()

    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{slug}.md"
    html_path = out_dir / f"{slug}.html"
    meta_path = out_dir / f"{slug}.meta.json"

    md_path.write_text(draft, encoding="utf-8")
    html_path.write_text(_to_html(draft, title, meta_description), encoding="utf-8")

    meta = {
        "title": title,
        "slug": slug,
        "meta_description": meta_description,
        "target_keyword": state.get("target_keyword", ""),
        "fact_check_score": fact_check_score,
        "seo_score": seo_score,
        "revision_count": revision_count,
        "sources": [s.model_dump() for s in state.get("sources", [])],
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    output_files = {"md": str(md_path), "html": str(html_path), "meta": str(meta_path)}
    logger.info("publisher_agent: published '%s' -> %s", title, ", ".join(output_files.values()))

    return {
        "approved": True,
        "title": title,
        "meta_description": meta_description,
        "slug": slug,
        "output_files": output_files,
        "warnings": warnings,
    }
