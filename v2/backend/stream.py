"""In-memory run-state store + SSE event drivers for the V2 FastAPI backend.

A run (or batch) is started as a background ``asyncio`` task that drives the LangGraph graph with
``astream_events`` and pushes JSON events onto an ``asyncio.Queue``. The SSE endpoints drain that
queue. Single-topic runs pause at the HITL interrupt (``hitl_required``); ``approve``/``reject``
spawn a follow-up task that resumes the graph and pushes the remaining events.

In-memory state is fine for a demo (single process). Keyed by run_id / batch_id.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pipeline.graph_batch import build_batch_graph
from pipeline.graph_v2 import build_graph_v2
from pipeline.state import initial_v2_state

logger = logging.getLogger("v2.stream")

NODES = {
    "memory_check", "research_agent", "outline_agent", "writer_agent",
    "fact_check_agent", "seo_agent", "editor_agent", "publisher_agent",
}


# ======================================================================================
# Single-topic runs
# ======================================================================================
class RunState:
    def __init__(self, run_id: str, topic: str, target_keyword: str,
                 style_persona: str, voice_input_path: str | None) -> None:
        self.run_id = run_id
        self.graph = build_graph_v2()  # interrupt before publisher (HITL)
        self.cfg = {"configurable": {"thread_id": run_id}}
        self.initial = initial_v2_state(topic, target_keyword, style_persona=style_persona,
                                        voice_input_path=voice_input_path)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.status = "pending"
        self.task: asyncio.Task | None = None


RUNS: dict[str, RunState] = {}


def _dump_list(items: list[Any]) -> list[dict]:
    return [i.model_dump() if hasattr(i, "model_dump") else i for i in items]


def _summarize(node: str, output: Any) -> dict:
    """Per-node, JSON-serializable slice of the node's state update (for the frontend)."""

    output = output or {}
    if not isinstance(output, dict):
        return {}
    if node == "memory_check":
        return {"memory_hit": output.get("memory_hit"),
                "reused_sources": len(output.get("reused_sources", []))}
    if node == "research_agent":
        return {"sources": _dump_list(output.get("sources", [])),
                "credibility_scores": output.get("credibility_scores", {}),
                "chunk_count": len(output.get("full_page_chunks", []))}
    if node == "outline_agent":
        return {"outline": output.get("outline", "")}
    if node == "writer_agent":
        return {"draft": output.get("draft", ""), "self_critique": output.get("self_critique", "")}
    if node == "fact_check_agent":
        return {"fact_check_score": output.get("fact_check_score"),
                "flagged_claims_trace": _dump_list(output.get("flagged_claims_trace", []))}
    if node == "seo_agent":
        return {"seo_score": output.get("seo_score"), "seo_feedback": output.get("seo_feedback")}
    if node == "editor_agent":
        return {"revision_count": output.get("revision_count"),
                "revision_history": _dump_list(output.get("revision_history", []))}
    if node == "publisher_agent":
        return {"approved": output.get("approved"), "title": output.get("title"),
                "output_files": output.get("output_files", {}),
                "audio_summary_path": output.get("audio_summary_path")}
    return {}


def _state_summary(values: dict) -> dict:
    return {
        "title": values.get("title", ""),
        "draft": values.get("draft", ""),
        "fact_check_score": values.get("fact_check_score"),
        "seo_score": values.get("seo_score"),
        "revision_count": values.get("revision_count", 0),
        "seo_feedback": values.get("seo_feedback"),
        "flagged_claims_trace": _dump_list(values.get("flagged_claims_trace", [])),
        "revision_history": _dump_list(values.get("revision_history", [])),
        "warnings": values.get("warnings", []),
    }


async def _emit(handle: Any, **event: Any) -> None:
    await handle.queue.put(event)


async def _drive(rs: RunState, *, resume: bool = False) -> None:
    """Run the graph to the next interrupt (or completion), emitting SSE events."""

    rs.status = "running"
    inp = None if resume else rs.initial
    try:
        async for ev in rs.graph.astream_events(inp, rs.cfg, version="v2"):
            etype, name = ev.get("event"), ev.get("name")
            if name in NODES and etype == "on_chain_start":
                await _emit(rs, event="node_start", node=name, run_id=rs.run_id)
            elif name in NODES and etype == "on_chain_end":
                data = _summarize(name, (ev.get("data") or {}).get("output"))
                await _emit(rs, event="node_complete", node=name, data=data, run_id=rs.run_id)
                if name == "editor_agent" and data.get("revision_history"):
                    await _emit(rs, event="revision_round",
                                data=data["revision_history"][-1], run_id=rs.run_id)

        snapshot = rs.graph.get_state(rs.cfg)
        if snapshot.next == ("publisher_agent",):
            rs.status = "awaiting_approval"
            await _emit(rs, event="hitl_required", data=_state_summary(snapshot.values), run_id=rs.run_id)
        else:
            rs.status = "complete"
            await _emit(rs, event="complete",
                        output_paths=(snapshot.values.get("output_files") or {}), run_id=rs.run_id)
    except Exception as exc:  # pragma: no cover - surfaced to the client
        rs.status = "error"
        logger.exception("run %s failed", rs.run_id)
        await _emit(rs, event="error", message=str(exc), run_id=rs.run_id)


def start_run(run_id: str, topic: str, target_keyword: str, style_persona: str,
              voice_input_path: str | None = None) -> RunState:
    rs = RunState(run_id, topic, target_keyword, style_persona, voice_input_path)
    RUNS[run_id] = rs
    rs.task = asyncio.create_task(_drive(rs))
    return rs


async def approve(rs: RunState) -> None:
    rs.task = asyncio.create_task(_drive(rs, resume=True))


async def reject(rs: RunState, note: str) -> None:
    # Re-route from the SEO node with a forced-low fact-check score so the router sends the draft
    # back to the editor, carrying the human feedback note for the editor to address.
    rs.graph.update_state(rs.cfg, {"human_feedback": note, "fact_check_score": 0.0},
                          as_node="seo_agent")
    rs.task = asyncio.create_task(_drive(rs, resume=True))


async def event_stream(rs: RunState):
    """Yield SSE-formatted events for a run until it completes or errors."""

    import json
    while True:
        event = await rs.queue.get()
        yield {"event": event.get("event", "message"), "data": json.dumps(event)}
        if event.get("event") in ("complete", "error"):
            break


# ======================================================================================
# Multi-topic batches
# ======================================================================================
class BatchRun:
    def __init__(self, batch_id: str, topics: list[str], target_keyword: str, style_persona: str) -> None:
        self.batch_id = batch_id
        self.graph = build_batch_graph()
        self.input = {"batch_topics": topics, "target_keyword": target_keyword,
                      "style_persona": style_persona}
        self.cfg = {"configurable": {"thread_id": batch_id}}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.status = "pending"
        self.task: asyncio.Task | None = None


BATCHES: dict[str, BatchRun] = {}


async def _drive_batch(bs: BatchRun) -> None:
    bs.status = "running"
    await _emit(bs, event="batch_start", topics=bs.input["batch_topics"], batch_id=bs.batch_id)
    try:
        async for ev in bs.graph.astream_events(bs.input, bs.cfg, version="v2"):
            etype, name = ev.get("event"), ev.get("name")
            if name == "run_topic" and etype == "on_chain_start":
                topic = ((ev.get("data") or {}).get("input") or {}).get("topic")
                await _emit(bs, event="topic_start", topic=topic, batch_id=bs.batch_id)
            elif name == "run_topic" and etype == "on_chain_end":
                results = ((ev.get("data") or {}).get("output") or {}).get("batch_results", [])
                payload = results[0].model_dump() if results else {}
                await _emit(bs, event="topic_complete", data=payload, batch_id=bs.batch_id)
        bs.status = "complete"
        await _emit(bs, event="complete", batch_id=bs.batch_id)
    except Exception as exc:  # pragma: no cover
        bs.status = "error"
        logger.exception("batch %s failed", bs.batch_id)
        await _emit(bs, event="error", message=str(exc), batch_id=bs.batch_id)


def start_batch(batch_id: str, topics: list[str], target_keyword: str, style_persona: str) -> BatchRun:
    bs = BatchRun(batch_id, topics, target_keyword, style_persona)
    BATCHES[batch_id] = bs
    bs.task = asyncio.create_task(_drive_batch(bs))
    return bs


async def batch_event_stream(bs: BatchRun):
    import json
    while True:
        event = await bs.queue.get()
        yield {"event": event.get("event", "message"), "data": json.dumps(event)}
        if event.get("event") in ("complete", "error"):
            break
