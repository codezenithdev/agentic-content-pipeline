"""publisher_agent (V2) — HITL gate, Medium-ready export, audio summary, and memory write-back.

Input  (state): final ``draft`` + scores + ``sources`` + ``revision_history`` + ``flagged_claims_trace``
Output (state update): ``approved``, ``title``, ``meta_description``, ``slug``, ``output_files``,
                       ``audio_summary_path``, ``warnings``

Reaching this node means a human approved (the graph interrupts before it). On approval it:
- writes ``<slug>.md`` / ``<slug>.html`` / ``<slug>.meta.json`` (meta now carries the full
  ``revision_history`` and ``flagged_claims_trace`` so the UI can render the timeline),
- if the run started from voice input, synthesizes an audio summary -> ``<slug>_summary.mp3``,
- stores the topic embedding + sources in ChromaDB so future similar runs hit memory.

Stays synchronous (the V2 graph runs sync nodes fine); shared with the V1 graph unchanged.
"""

from __future__ import annotations

import html
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import markdown as md

from memory import store
from pipeline import config
from pipeline.state import V2PipelineState
from voice import tts_output

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
    for line in draft.splitlines():
        if line.strip().startswith("# "):
            return line.strip()[2:].strip()
    return fallback.strip().title()


def _slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:80] or "article"


def _to_html(draft: str, title: str, description: str) -> str:
    body = md.markdown(draft, extensions=["extra", "sane_lists"])
    return _HTML_TEMPLATE.format(
        title=html.escape(title), description=html.escape(description, quote=True), body=body
    )


def _summary_text(draft: str, meta_description: str) -> str:
    """First 3 sentences of the article body, prefixed with the meta description."""

    text = re.sub(r"[#*_`>]", " ", draft)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)[:3]
    body = " ".join(sentences)
    return f"{meta_description} {body}".strip() if meta_description else body


def publisher_agent(state: V2PipelineState) -> dict:
    """Finalize, export, optionally narrate, and remember the article (LangGraph node)."""

    warnings = list(state.get("warnings", []))
    draft = state.get("draft", "")
    revision_count = state.get("revision_count", 0)
    fact_check_score = state.get("fact_check_score") or 0.0
    seo_score = state.get("seo_score") or 0
    seo_feedback = state.get("seo_feedback") or {}
    history = state.get("revision_history", []) or []
    below_threshold = fact_check_score < config.FACT_CHECK_THRESHOLD or seo_score < config.SEO_THRESHOLD

    # Loop-guard transparency: capped, or the editor stalled.
    if revision_count >= config.MAX_REVISIONS and below_threshold:
        msg = (f"Max revisions reached ({revision_count}/{config.MAX_REVISIONS}) — needs human "
               f"review: fact_check={fact_check_score:.2f}, seo={seo_score}.")
        warnings.append(msg)
        logger.warning(msg)
    elif below_threshold and history and history[-1].edit_distance < config.EDIT_DISTANCE_STALL:
        msg = (f"Editor stalled (last edit distance {history[-1].edit_distance:.3f} < "
               f"{config.EDIT_DISTANCE_STALL}) — needs human review.")
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
    output_files = {"md": str(md_path), "html": str(html_path), "meta": str(meta_path)}

    # Audio summary — only if the run started from voice input.
    audio_summary_path = None
    if state.get("voice_input_path"):
        try:
            audio_path = out_dir / f"{slug}_summary.mp3"
            tts_output.synthesize(_summary_text(draft, meta_description), str(audio_path))
            audio_summary_path = str(audio_path)
            output_files["audio"] = audio_summary_path
        except Exception as exc:  # TTS failure shouldn't block publication
            warnings.append(f"Audio summary generation failed: {exc}")
            logger.warning("publisher: TTS failed (%s)", exc)

    meta = {
        "title": title,
        "slug": slug,
        "meta_description": meta_description,
        "target_keyword": state.get("target_keyword", ""),
        "style_persona": state.get("style_persona", ""),
        "fact_check_score": fact_check_score,
        "seo_score": seo_score,
        "revision_count": revision_count,
        "sources": [s.model_dump() for s in state.get("sources", [])],
        "credibility_scores": state.get("credibility_scores", {}),
        "revision_history": [r.model_dump() for r in history],
        "flagged_claims_trace": [t.model_dump() for t in state.get("flagged_claims_trace", [])],
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Memory write-back so future similar topics get a memory hit.
    try:
        store.save_run(slug, state.get("topic", ""), state.get("fact_sheet", ""),
                       state.get("sources", []), final_score=float(fact_check_score))
    except Exception as exc:  # memory is best-effort
        warnings.append(f"Memory write-back failed: {exc}")
        logger.warning("publisher: memory save failed (%s)", exc)

    logger.info("publisher: published '%s' (revision %d, audio=%s)",
                title, revision_count, bool(audio_summary_path))
    return {
        "approved": True,
        "title": title,
        "meta_description": meta_description,
        "slug": slug,
        "output_files": output_files,
        "audio_summary_path": audio_summary_path,
        "warnings": warnings,
    }
