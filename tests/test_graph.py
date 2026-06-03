"""Router, loop-guard, and full-graph tests (mock mode).

The router tests are pure-function assertions; the full-graph tests prove the editor loop
fires on low scores and that the guard always terminates the loop at MAX_REVISIONS.
"""

from __future__ import annotations

import uuid

from pipeline import config, llm
from pipeline.graph import build_graph, route_after_seo
from pipeline.state import RevisionRound, initial_state, initial_v2_state


def _state(revision_count: int, fact_check: float, seo: int) -> dict:
    st = initial_state("AI agents", "langgraph")
    st["revision_count"] = revision_count
    st["fact_check_score"] = fact_check
    st["seo_score"] = seo
    return st


def _cfg() -> dict:
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


# --- router (pure function) -----------------------------------------------------------
def test_router_passthrough_when_scores_pass():
    assert route_after_seo(_state(0, 0.95, 82)) == "publisher_agent"


def test_router_loops_on_low_fact_check():
    assert route_after_seo(_state(0, 0.50, 90)) == "editor_agent"


def test_router_loops_on_low_seo():
    assert route_after_seo(_state(0, 0.95, 40)) == "editor_agent"


def test_router_guard_caps_at_max_revisions():
    # Even with terrible scores, hitting the revision cap routes to the publisher.
    assert route_after_seo(_state(config.MAX_REVISIONS, 0.10, 10)) == "publisher_agent"


# --- full graph -----------------------------------------------------------------------
def test_full_graph_passthrough(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    graph = build_graph()
    cfg = _cfg()
    res = graph.invoke(initial_state("AI agents", "langgraph"), cfg)
    assert graph.get_state(cfg).next == ("publisher_agent",)  # paused at HITL gate
    assert res["revision_count"] == 0
    final = graph.invoke(None, cfg)  # resume = approve
    assert final["approved"] is True
    assert final["output_files"]


def test_full_graph_loops_then_guard_terminates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "EDIT_DISTANCE_STALL", -1.0)  # disable stall gate -> test the cap
    # Force the graders to keep failing so the editor loop must engage every pass.
    monkeypatch.setitem(llm.MOCK_RESPONSES, "fact_check", {"fact_check_score": 0.5, "flagged_claims": []})
    monkeypatch.setitem(
        llm.MOCK_RESPONSES,
        "seo",
        {"seo_score": 40, "feedback": {"keyword_issues": "x", "heading_issues": "y",
                                       "meta_description": "d", "readability_score": 20.0,
                                       "suggestions": ["s"]}},
    )
    graph = build_graph()
    cfg = _cfg()
    res = graph.invoke(initial_v2_state("AI agents", "langgraph"), cfg)
    assert res["revision_count"] == config.MAX_REVISIONS  # guard capped the loop
    final = graph.invoke(None, cfg)
    assert any("Max revisions reached" in w for w in final["warnings"])


def test_router_stall_gate_escalates():
    st = _state(1, 0.5, 40)  # scores would normally loop to the editor
    st["revision_history"] = [
        RevisionRound(round_number=1, fact_check_score=0.5, seo_score=40.0, edit_distance=0.01)
    ]
    assert route_after_seo(st) == "publisher_agent"  # near-zero edit distance -> stall


def test_full_graph_stall_escalates_before_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setitem(llm.MOCK_RESPONSES, "fact_check", {"fact_check_score": 0.5, "verdicts": []})
    monkeypatch.setitem(
        llm.MOCK_RESPONSES, "seo",
        {"seo_score": 40, "feedback": {"keyword_issues": "x", "heading_issues": "y",
                                       "meta_description": "d", "readability_score": 20.0,
                                       "suggestions": ["s"]}},
    )
    graph = build_graph()
    cfg = _cfg()
    res = graph.invoke(initial_v2_state("AI agents", "langgraph"), cfg)
    # The mock editor emits identical text on repeat -> edit distance ~0 -> stall before the cap.
    assert 0 < res["revision_count"] < config.MAX_REVISIONS
    assert res["revision_history"]
    final = graph.invoke(None, cfg)
    assert any("stalled" in w.lower() for w in final["warnings"])
