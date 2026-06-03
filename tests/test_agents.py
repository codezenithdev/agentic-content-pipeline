"""Smoke tests for each agent, running in deterministic mock mode (no API keys)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pipeline import config
from pipeline.agents.editor import editor_agent
from pipeline.agents.fact_check import fact_check_agent
from pipeline.agents.publisher import publisher_agent
from pipeline.agents.research import research_agent
from pipeline.agents.seo import seo_agent
from pipeline.agents.writer import writer_agent
from pipeline.state import Source, initial_state


@pytest.fixture
def drafted_state() -> dict:
    """A state advanced through research -> writer -> fact_check -> seo (mock mode)."""

    st = initial_state("AI agents", "langgraph")
    st.update(asyncio.run(research_agent(st)))
    st.update(writer_agent(st))
    st.update(fact_check_agent(st))
    st.update(seo_agent(st))
    return st


def test_research_builds_sources_and_fact_sheet():
    out = asyncio.run(research_agent(initial_state("AI agents", "langgraph")))
    assert out["sources"] and all(isinstance(s, Source) for s in out["sources"])
    assert "Fact Sheet" in out["fact_sheet"]
    # V2: deep-research artifacts present
    assert out["full_page_chunks"] and out["credibility_scores"]
    assert any(tag in out["fact_sheet"] for tag in ("[HIGH]", "[MEDIUM]", "[LOW]"))


def test_writer_builds_outline_and_draft():
    st = initial_state("AI agents", "langgraph")
    st.update(asyncio.run(research_agent(st)))
    out = writer_agent(st)
    assert out["outline"].lstrip().startswith("#")
    assert out["draft"].lstrip().startswith("#")


def test_writer_handles_empty_fact_sheet():
    out = writer_agent(initial_state("x", "y"))  # no research => empty fact sheet
    assert out["draft"] == ""
    assert any("empty fact sheet" in w.lower() for w in out["warnings"])


def test_fact_check_returns_bounded_score(drafted_state):
    out = fact_check_agent(drafted_state)
    assert 0.0 <= out["fact_check_score"] <= 1.0
    assert isinstance(out["flagged_claims"], list)


def test_seo_returns_score_and_required_feedback_keys(drafted_state):
    out = seo_agent(drafted_state)
    assert 0 <= out["seo_score"] <= 100
    required = {"keyword_issues", "heading_issues", "meta_description", "readability_score", "suggestions"}
    assert required <= set(out["seo_feedback"].keys())


def test_editor_increments_revision_count(drafted_state):
    before = drafted_state["revision_count"]
    out = editor_agent(drafted_state)
    assert out["revision_count"] == before + 1
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
