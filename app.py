"""Streamlit demo for the agentic content pipeline.

Run with:  streamlit run app.py

The UI: enter a topic + target keyword, watch the LangGraph pipeline step through each agent
live (with a badge whenever the router loops back to the editor), inspect the flagged claims and
SEO feedback, then approve at the human-in-the-loop gate to export the Medium-ready files.

State is held in ``st.session_state`` across Streamlit reruns; the compiled graph (and its
in-memory checkpointer) is kept there too so the HITL interrupt can be resumed on approval.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from pipeline import config
from pipeline.graph import build_graph
from pipeline.state import initial_state

NODE_LABELS = {
    "research_agent": "🔎 Research",
    "writer_agent": "✍️ Writer",
    "fact_check_agent": "✅ Fact-check",
    "seo_agent": "📈 SEO",
    "editor_agent": "🛠️ Editor",
    "publisher_agent": "🚀 Publisher",
}

st.set_page_config(page_title="Agentic Content Pipeline", page_icon="📝", layout="wide")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def render_mermaid(mermaid_text: str, height: int = 540) -> None:
    """Render LangGraph's mermaid text using mermaid.js from a CDN (no system deps)."""

    html = f"""
    <div class="mermaid">{mermaid_text}</div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, securityLevel: 'loose', theme: 'neutral' }});
    </script>
    """
    components.html(html, height=height, scrolling=True)


def reset_run() -> None:
    """Clear per-run state so the user can start a fresh pipeline."""

    for key in ("phase", "graph", "cfg", "pre_state", "final_state", "trace"):
        st.session_state.pop(key, None)


def render_scores(state: dict) -> None:
    c1, c2, c3 = st.columns(3)
    fc = state.get("fact_check_score")
    c1.metric(
        "Fact-check score",
        f"{fc:.2f}" if fc is not None else "—",
        help=f"Threshold ≥ {config.FACT_CHECK_THRESHOLD}",
    )
    c2.metric(
        "SEO score",
        state.get("seo_score") if state.get("seo_score") is not None else "—",
        help=f"Threshold ≥ {config.SEO_THRESHOLD}",
    )
    c3.metric(
        "Revisions", state.get("revision_count", 0), help=f"Cap: {config.MAX_REVISIONS}"
    )


def render_findings(state: dict) -> None:
    flagged = state.get("flagged_claims", []) or []
    with st.expander(f"🚩 Flagged claims ({len(flagged)})", expanded=bool(flagged)):
        if not flagged:
            st.success("No unsupported claims — every claim is backed by a source.")
        for fc in flagged:
            st.markdown(
                f"**Claim:** {fc.claim}\n\n"
                f"- **Why:** {fc.reason}\n"
                f"- **Fix:** {fc.suggested_fix}\n"
                f"- **Source:** {fc.source_url or '_none_'}"
            )
            st.divider()

    fb = state.get("seo_feedback") or {}
    with st.expander("📈 SEO feedback", expanded=False):
        if not fb:
            st.info("No SEO feedback yet.")
        else:
            s1, s2, s3 = st.columns(3)
            s1.metric("Keyword density", f"{fb.get('keyword_density_pct', '—')}%")
            s2.metric("Readability (Flesch)", fb.get("readability_score", "—"))
            h = fb.get("headings", {})
            s3.metric(
                "Headings",
                f"H1:{h.get('h1', '?')} H2:{h.get('h2', '?')} H3:{h.get('h3', '?')}",
            )
            st.markdown(f"**Keyword issues:** {fb.get('keyword_issues', '')}")
            st.markdown(f"**Heading issues:** {fb.get('heading_issues', '')}")
            st.markdown(f"**Meta description:** {fb.get('meta_description', '')}")
            if fb.get("suggestions"):
                st.markdown("**Suggestions:**")
                for s in fb["suggestions"]:
                    st.markdown(f"- {s}")


def render_warnings(state: dict) -> None:
    for w in state.get("warnings", []) or []:
        (st.warning if "needs human review" in w else st.info)(w)


# --------------------------------------------------------------------------------------
# Sidebar — configuration
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    config.MOCK_MODE = st.toggle(
        "Mock mode (no API calls)",
        value=config.MOCK_MODE,
        help="Run the whole pipeline with deterministic fakes — no API keys or network needed.",
    )
    st.caption(f"**Writer/Editor model:** `{config.MODEL_STRONG}`")
    st.caption(f"**Grader model:** `{config.MODEL_MINI}`")
    st.caption(
        f"**Thresholds:** fact-check ≥ {config.FACT_CHECK_THRESHOLD}, "
        f"SEO ≥ {config.SEO_THRESHOLD}, max revisions = {config.MAX_REVISIONS}"
    )
    st.divider()
    if st.button("🔄 Reset", use_container_width=True):
        reset_run()
        st.rerun()


# --------------------------------------------------------------------------------------
# Header + inputs
# --------------------------------------------------------------------------------------
st.title("📝 Agentic Content Pipeline")
st.caption(
    "Research → Write → Fact-check → SEO → (Edit loop) → Human approval → Publish"
)

phase = st.session_state.get("phase", "idle")
disabled = phase in {"approve", "publishing"}

col_topic, col_kw = st.columns([3, 2])
topic = col_topic.text_input(
    "Topic", value="Mediterranean diet health benefits", disabled=disabled
)
keyword = col_kw.text_input(
    "Target keyword", value="Mediterranean diet", disabled=disabled
)
run_clicked = st.button(
    "▶️ Run pipeline", type="primary", disabled=disabled or not (topic and keyword)
)

# Graph diagram (always visible).
with st.expander("🗺️ Pipeline graph", expanded=True):
    try:
        diagram = build_graph().get_graph().draw_mermaid()
        render_mermaid(diagram)
    except Exception as exc:  # pragma: no cover - diagram is best-effort
        st.info(f"Graph diagram unavailable: {exc}")


# --------------------------------------------------------------------------------------
# Run the pipeline up to the HITL interrupt, streaming live progress
# --------------------------------------------------------------------------------------
if run_clicked:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
    st.session_state.graph = graph
    st.session_state.cfg = cfg

    st.subheader("⏳ Live progress")
    timeline_ph = st.empty()
    metrics_ph = st.empty()
    badges_ph = st.empty()

    completed: list[str] = []
    badges: list[str] = []
    metrics = {"fact_check_score": None, "seo_score": None, "revision_count": 0}

    try:
        for event in graph.stream(
            initial_state(topic, keyword), cfg, stream_mode="updates"
        ):
            for node, update in event.items():
                completed.append(node)
                update = update or {}
                metrics["fact_check_score"] = update.get(
                    "fact_check_score", metrics["fact_check_score"]
                )
                metrics["seo_score"] = update.get("seo_score", metrics["seo_score"])
                metrics["revision_count"] = update.get(
                    "revision_count", metrics["revision_count"]
                )
                if node == "editor_agent":
                    badges.append(
                        f"🔁 Router looped back → **Editor** (revision {metrics['revision_count']})"
                    )

                timeline_ph.markdown(
                    "  ".join(f"{NODE_LABELS.get(n, n)} ✓" for n in completed)
                    or "_starting…_"
                )
                with metrics_ph.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric(
                        "Fact-check",
                        metrics["fact_check_score"]
                        if metrics["fact_check_score"] is not None
                        else "—",
                    )
                    m2.metric(
                        "SEO",
                        metrics["seo_score"]
                        if metrics["seo_score"] is not None
                        else "—",
                    )
                    m3.metric("Revisions", metrics["revision_count"])
                if badges:
                    badges_ph.warning("\n\n".join(badges))
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.stop()

    st.session_state.pre_state = graph.get_state(cfg).values
    st.session_state.trace = {"completed": completed, "badges": badges}
    st.session_state.phase = "approve"
    st.rerun()


# --------------------------------------------------------------------------------------
# HITL gate — review and approve
# --------------------------------------------------------------------------------------
if phase == "approve":
    state = st.session_state.pre_state
    trace = st.session_state.get("trace", {})

    st.subheader("🧪 Quality report")
    render_scores(state)
    if trace.get("badges"):
        st.warning("\n\n".join(trace["badges"]))
    render_warnings(state)
    render_findings(state)

    with st.expander("📄 Final draft (preview)", expanded=True):
        st.markdown(state.get("draft", "_(empty draft)_"))

    st.subheader("🚦 Human approval gate")
    st.info("Review the draft above, then approve to export the Medium-ready files.")
    col_ok, col_no = st.columns(2)
    if col_ok.button("✅ Approve & Publish", type="primary", use_container_width=True):
        st.session_state.final_state = st.session_state.graph.invoke(
            None, st.session_state.cfg
        )
        st.session_state.phase = "published"
        st.rerun()
    if col_no.button("✗ Reject (discard)", use_container_width=True):
        reset_run()
        st.rerun()


# --------------------------------------------------------------------------------------
# Published — show the article + downloads
# --------------------------------------------------------------------------------------
if phase == "published":
    state = st.session_state.final_state
    st.success(f"Published: **{state.get('title', '(untitled)')}**")
    render_scores(state)
    render_warnings(state)

    files = state.get("output_files") or {}
    cols = st.columns(len(files) or 1)
    for col, (kind, path) in zip(cols, files.items()):
        p = Path(path)
        if p.exists():
            mime = {
                "md": "text/markdown",
                "html": "text/html",
                "meta": "application/json",
            }.get(kind, "text/plain")
            col.download_button(
                f"⬇️ Download .{'json' if kind == 'meta' else kind}",
                data=p.read_bytes(),
                file_name=p.name,
                mime=mime,
                use_container_width=True,
            )

    with st.expander("📄 Final article", expanded=True):
        st.markdown(state.get("draft", ""))

    if st.button("➕ Write another"):
        reset_run()
        st.rerun()
