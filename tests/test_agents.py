"""Smoke tests for each agent, running in deterministic mock mode (no API keys).

research / outline / writer / fact_check / seo are async (V2) and driven via ``asyncio.run``;
editor and publisher remain synchronous until V2-M6.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pipeline import config
from pipeline.agents.editor import editor_agent
from pipeline.agents.fact_check import fact_check_agent
from pipeline.agents.outline import outline_agent
from pipeline.agents.publisher import publisher_agent
from pipeline.agents.research import research_agent
from pipeline.agents.seo import seo_agent
from pipeline.agents.writer import writer_agent
from pipeline.state import Source, initial_state, initial_v2_state


@pytest.fixture
def drafted_state() -> dict:
    """A state advanced through research -> writer -> fact_check -> seo (mock mode)."""

    st = initial_v2_state("AI agents", "langgraph")
    st.update(asyncio.run(research_agent(st)))
    st.update(asyncio.run(writer_agent(st)))
    st.update(asyncio.run(fact_check_agent(st)))
    st.update(asyncio.run(seo_agent(st)))
    return st


def test_research_builds_sources_and_fact_sheet():
    out = asyncio.run(research_agent(initial_v2_state("AI agents", "langgraph")))
    assert out["sources"] and all(isinstance(s, Source) for s in out["sources"])
    assert "Fact Sheet" in out["fact_sheet"]
    assert out["full_page_chunks"] and out["credibility_scores"]
    assert any(tag in out["fact_sheet"] for tag in ("[HIGH]", "[MEDIUM]", "[LOW]"))


def test_outline_agent_produces_outline():
    st = initial_v2_state("AI agents", "langgraph", style_persona="op-ed")
    st.update(asyncio.run(research_agent(st)))
    out = asyncio.run(outline_agent(st))
    assert out["outline"].lstrip().startswith("#")


def test_writer_uses_outline_and_self_critiques():
    st = initial_v2_state("AI agents", "langgraph")
    st.update(asyncio.run(research_agent(st)))
    st.update(asyncio.run(outline_agent(st)))
    out = asyncio.run(writer_agent(st))
    assert out["draft"].lstrip().startswith("#")
    assert out["self_critique"]  # the self-critique pass ran


def test_writer_handles_empty_fact_sheet():
    out = asyncio.run(writer_agent(initial_v2_state("x", "y")))
    assert out["draft"] == ""
    assert any("empty fact sheet" in w.lower() for w in out["warnings"])


def test_fact_check_returns_bounded_score_and_trace(drafted_state):
    out = asyncio.run(fact_check_agent(drafted_state))
    assert 0.0 <= out["fact_check_score"] <= 1.0
    assert isinstance(out["flagged_claims"], list)
    assert isinstance(out["flagged_claims_trace"], list)


def test_seo_returns_score_and_fix_lists(drafted_state):
    out = asyncio.run(seo_agent(drafted_state))
    assert 0 <= out["seo_score"] <= 100
    fb = out["seo_feedback"]
    required = {"keyword_issues", "heading_issues", "meta_description", "readability_score",
                "suggestions", "fix_now", "fix_later", "title_tag"}
    assert required <= set(fb.keys())


def test_editor_increments_revision_count(drafted_state):
    before = drafted_state["revision_count"]
    out = asyncio.run(editor_agent(drafted_state))
    assert out["revision_count"] == before + 1
    assert out["revision_history"] and out["revision_history"][-1].round_number == before + 1
    assert out["draft"]


def test_publisher_writes_three_files(tmp_path, monkeypatch, drafted_state):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    out = publisher_agent(drafted_state)
    assert out["approved"] is True
    files = out["output_files"]
    for kind in ("md", "html", "meta"):
        assert Path(files[kind]).exists()
    meta = json.loads(Path(files["meta"]).read_text(encoding="utf-8"))
    assert meta["title"] and "sources" in meta
    assert "<article>" in Path(files["html"]).read_text(encoding="utf-8")
