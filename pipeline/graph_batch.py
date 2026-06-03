"""Multi-topic batch graph using the LangGraph Send API.

A dispatch node fans out one ``Send`` per topic to ``run_topic``, which runs the full single-topic
V2 pipeline (auto-publishing — no HITL) and returns a :class:`BatchResult`. Results accumulate via
an ``operator.add`` reducer and are summarized by ``results_aggregator``. Concurrency is capped at
``config.MAX_PARALLEL_SUBGRAPHS`` with a per-event-loop semaphore.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import uuid
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from pipeline import config
from pipeline.graph_v2 import build_graph_v2
from pipeline.state import BatchResult, initial_v2_state

logger = logging.getLogger("pipeline.batch")

RUN_TOPIC = "run_topic"
AGGREGATOR = "results_aggregator"


class BatchState(TypedDict, total=False):
    batch_topics: list[str]
    target_keyword: str
    style_persona: str
    batch_results: Annotated[list[BatchResult], operator.add]


_subgraph = None
_semaphores: dict[int, asyncio.Semaphore] = {}


def _get_subgraph():
    """A single auto-publishing (no-interrupt) single-topic graph, reused across topics."""

    global _subgraph
    if _subgraph is None:
        _subgraph = build_graph_v2(interrupt=False)
    return _subgraph


def _semaphore() -> asyncio.Semaphore:
    """One semaphore per running event loop (so repeated asyncio.run calls don't share state)."""

    loop_id = id(asyncio.get_running_loop())
    sem = _semaphores.get(loop_id)
    if sem is None:
        sem = asyncio.Semaphore(config.MAX_PARALLEL_SUBGRAPHS)
        _semaphores[loop_id] = sem
    return sem


def _dispatch(state: BatchState) -> list[Send]:
    topics = state.get("batch_topics", [])
    keyword = state.get("target_keyword", "")
    persona = state.get("style_persona", "technical deep-dive")
    logger.info("[BATCH] dispatching %d topic(s) (max parallel %d)",
                len(topics), config.MAX_PARALLEL_SUBGRAPHS)
    return [
        Send(RUN_TOPIC, {"topic": t, "target_keyword": keyword, "style_persona": persona})
        for t in topics
    ]


async def run_topic(payload: dict) -> dict:
    """Run the full single-topic pipeline for one topic and return its BatchResult."""

    topic = payload["topic"]
    async with _semaphore():
        logger.info("[BATCH] start topic %r", topic)
        cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
        try:
            result = await _get_subgraph().ainvoke(
                initial_v2_state(topic, payload.get("target_keyword", ""),
                                 style_persona=payload.get("style_persona", "technical deep-dive")),
                cfg,
            )
            files = result.get("output_files") or {}
            batch_result = BatchResult(topic=topic, status="complete",
                                       final_score=result.get("fact_check_score"),
                                       output_path=files.get("md"))
        except Exception as exc:
            logger.warning("[BATCH] topic %r failed: %s", topic, exc)
            batch_result = BatchResult(topic=topic, status="failed", final_score=None, output_path=None)
    logger.info("[BATCH] done topic %r -> %s", topic, batch_result.status)
    return {"batch_results": [batch_result]}


async def results_aggregator(state: BatchState) -> dict:
    results = state.get("batch_results", [])
    complete = sum(1 for r in results if r.status == "complete")
    logger.info("[BATCH] aggregated %d/%d complete", complete, len(results))
    return {}


def build_batch_graph():
    """Build and compile the multi-topic batch graph (Send fan-out -> aggregator)."""

    builder = StateGraph(BatchState)
    builder.add_node(RUN_TOPIC, run_topic)
    builder.add_node(AGGREGATOR, results_aggregator)
    builder.add_conditional_edges(START, _dispatch, [RUN_TOPIC])
    builder.add_edge(RUN_TOPIC, AGGREGATOR)
    builder.add_edge(AGGREGATOR, END)
    return builder.compile()
