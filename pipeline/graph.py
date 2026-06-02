"""build_graph() — assemble the multi-agent pipeline as a LangGraph state machine.

Topology::

    START -> research -> writer -> fact_check -> seo -> [router] ─▶ editor ─┐
                                                          │                 │
                                                          │   (editor loops back through
                                                          │    fact_check -> seo to re-validate)
                                                          └────────────▶ publisher -> END

The conditional **router** runs after the SEO agent. Its checks are ordered so the loop can
never run forever:

1. ``revision_count >= MAX_REVISIONS``  -> publisher (the publisher attaches a "needs human
   review" warning). This is checked FIRST.
2. ``fact_check_score < FACT_CHECK_THRESHOLD``  -> editor
3. ``seo_score < SEO_THRESHOLD``  -> editor
4. otherwise  -> publisher

Every decision is logged so the loop behaviour is observable. The graph is compiled with a
``MemorySaver`` checkpointer and ``interrupt_before=["publisher_agent"]`` so it pauses for human
approval before publishing (HITL).
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from pipeline import config
from pipeline.agents.editor import editor_agent
from pipeline.agents.fact_check import fact_check_agent
from pipeline.agents.publisher import publisher_agent
from pipeline.agents.research import research_agent
from pipeline.agents.seo import seo_agent
from pipeline.agents.writer import writer_agent
from pipeline.state import FlaggedClaim, PipelineState, Source

logger = logging.getLogger("pipeline.router")

#: Node names (also used as the labels in the rendered graph diagram).
RESEARCH = "research_agent"
WRITER = "writer_agent"
FACT_CHECK = "fact_check_agent"
SEO = "seo_agent"
EDITOR = "editor_agent"
PUBLISHER = "publisher_agent"


def route_after_seo(state: PipelineState) -> Literal["editor_agent", "publisher_agent"]:
    """Decide whether to loop back to the editor or proceed to the publisher.

    Pure routing function (it does not mutate state); it logs each decision for observability.
    The loop guard (``revision_count >= MAX_REVISIONS``) is checked first.
    """

    revision_count = state.get("revision_count", 0)
    fact_check_score = state.get("fact_check_score")
    seo_score = state.get("seo_score")
    fc = fact_check_score if fact_check_score is not None else 0.0
    seo = seo_score if seo_score is not None else 0

    if revision_count >= config.MAX_REVISIONS:
        logger.info(
            "router: revision_count %d >= MAX_REVISIONS %d -> %s (loop guard)",
            revision_count, config.MAX_REVISIONS, PUBLISHER,
        )
        return PUBLISHER
    if fc < config.FACT_CHECK_THRESHOLD:
        logger.info(
            "router: fact_check_score %.2f < %.2f -> %s (revision %d)",
            fc, config.FACT_CHECK_THRESHOLD, EDITOR, revision_count + 1,
        )
        return EDITOR
    if seo < config.SEO_THRESHOLD:
        logger.info(
            "router: seo_score %d < %d -> %s (revision %d)",
            seo, config.SEO_THRESHOLD, EDITOR, revision_count + 1,
        )
        return EDITOR
    logger.info(
        "router: fact_check_score %.2f >= %.2f and seo_score %d >= %d -> %s",
        fc, config.FACT_CHECK_THRESHOLD, seo, config.SEO_THRESHOLD, PUBLISHER,
    )
    return PUBLISHER


def _default_checkpointer() -> MemorySaver:
    """In-memory checkpointer whose serializer explicitly allows our Pydantic state types.

    Without the allowlist, LangGraph's msgpack serializer warns ("Deserializing unregistered
    type ...") whenever the checkpointed state holds ``Source`` / ``FlaggedClaim`` objects, and
    threatens to block it in a future version. Registering the types keeps the demo output clean.
    """

    serde = JsonPlusSerializer(allowed_msgpack_modules=[Source, FlaggedClaim])
    return MemorySaver(serde=serde)


def build_graph(checkpointer=None):
    """Build and compile the pipeline graph.

    Args:
        checkpointer: a LangGraph checkpointer; defaults to an in-memory ``MemorySaver`` whose
            serializer allows our Pydantic state types. A checkpointer is required so the graph
            can pause at the HITL interrupt and resume.

    Returns:
        A compiled graph. Invoke with ``config={"configurable": {"thread_id": "<id>"}}``; it
        runs to the interrupt before ``publisher_agent``, then resume with ``invoke(None, config)``.
    """

    builder = StateGraph(PipelineState)
    builder.add_node(RESEARCH, research_agent)
    builder.add_node(WRITER, writer_agent)
    builder.add_node(FACT_CHECK, fact_check_agent)
    builder.add_node(SEO, seo_agent)
    builder.add_node(EDITOR, editor_agent)
    builder.add_node(PUBLISHER, publisher_agent)

    builder.add_edge(START, RESEARCH)
    builder.add_edge(RESEARCH, WRITER)
    builder.add_edge(WRITER, FACT_CHECK)
    builder.add_edge(FACT_CHECK, SEO)
    builder.add_conditional_edges(SEO, route_after_seo, {EDITOR: EDITOR, PUBLISHER: PUBLISHER})
    builder.add_edge(EDITOR, FACT_CHECK)  # re-validate every revision
    builder.add_edge(PUBLISHER, END)

    return builder.compile(
        checkpointer=checkpointer or _default_checkpointer(),
        interrupt_before=[PUBLISHER],
    )
