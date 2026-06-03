"""V2 end-to-end integration tests against the 8-node graph_v2 (mock mode).

Covers: voice -> full run -> HITL approve -> exported files (+ audio); the memory loop (same
topic twice -> second run hits memory and reuses sources). Batch is covered in test_batch.py.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pipeline import config
from pipeline.graph_v2 import build_graph_v2
from pipeline.state import initial_v2_state


def _cfg(tid: str) -> dict:
    return {"configurable": {"thread_id": tid}}


def test_v2_voice_run_approve_exports_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

    async def run():
        graph = build_graph_v2()
        cfg = _cfg("v2-voice")
        pre = await graph.ainvoke(
            initial_v2_state("AI agents", "langgraph", voice_input_path="/tmp/topic.mp3"), cfg
        )
        # paused at the HITL gate with an outline + draft produced
        assert graph.get_state(cfg).next == ("publisher_agent",)
        assert pre.get("outline") and pre.get("draft")
        return await graph.ainvoke(None, cfg)  # approve

    final = asyncio.run(run())
    assert final["approved"] is True
    files = final["output_files"]
    for kind in ("md", "html", "meta", "audio"):
        assert kind in files and Path(files[kind]).exists()
    assert final["audio_summary_path"] and Path(final["audio_summary_path"]).exists()


def test_v2_memory_loop_second_run_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

    async def run_once(graph, tid):
        cfg = _cfg(tid)
        pre = await graph.ainvoke(initial_v2_state("AI agents", "langgraph"), cfg)
        await graph.ainvoke(None, cfg)  # approve -> publisher writes the run to memory
        return pre

    async def run():
        graph = build_graph_v2()
        first = await run_once(graph, "mem-1")
        second = await run_once(graph, "mem-2")
        return first, second

    first, second = asyncio.run(run())
    assert first["memory_hit"] is False
    assert second["memory_hit"] is True
    assert len(second["reused_sources"]) > 0


def test_v2_revision_history_recorded_on_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "EDIT_DISTANCE_STALL", -1.0)  # let it run to the cap
    from pipeline import llm

    monkeypatch.setitem(llm.MOCK_RESPONSES, "fact_check", {"fact_check_score": 0.4, "verdicts": []})
    monkeypatch.setitem(
        llm.MOCK_RESPONSES, "seo",
        {"seo_score": 30, "feedback": {"keyword_issues": "x", "heading_issues": "y",
                                       "meta_description": "d", "readability_score": 10.0, "suggestions": ["s"]}},
    )

    async def run():
        graph = build_graph_v2()
        cfg = _cfg("v2-loop")
        pre = await graph.ainvoke(initial_v2_state("AI agents", "langgraph"), cfg)
        return pre

    pre = asyncio.run(run())
    assert pre["revision_count"] == config.MAX_REVISIONS
    assert len(pre["revision_history"]) == config.MAX_REVISIONS
    assert all(hasattr(r, "edit_distance") for r in pre["revision_history"])
