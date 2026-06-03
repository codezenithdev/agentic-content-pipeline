"""Multi-topic batch tests (Send API, mock mode)."""

from __future__ import annotations

import asyncio

from pipeline import config
from pipeline.graph_batch import build_batch_graph
from pipeline.state import BatchResult


def test_batch_runs_all_topics(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    graph = build_batch_graph()

    async def run():
        return await graph.ainvoke(
            {"batch_topics": ["AI agents", "Mediterranean diet", "Quantum computing"],
             "target_keyword": "kw", "style_persona": "technical deep-dive"},
            {"configurable": {"thread_id": "batch-test"}},
        )

    res = asyncio.run(run())
    results = res["batch_results"]
    assert len(results) == 3
    assert all(isinstance(r, BatchResult) for r in results)
    assert all(r.status == "complete" for r in results)
    assert {r.topic for r in results} == {"AI agents", "Mediterranean diet", "Quantum computing"}
    assert all(r.output_path for r in results)


def test_batch_dispatch_emits_one_send_per_topic():
    from pipeline.graph_batch import _dispatch

    sends = _dispatch({"batch_topics": ["a", "b"], "target_keyword": "k", "style_persona": "op-ed"})
    assert len(sends) == 2
    assert all(s.node == "run_topic" for s in sends)
    assert {s.arg["topic"] for s in sends} == {"a", "b"}
    assert all(s.arg["style_persona"] == "op-ed" for s in sends)
