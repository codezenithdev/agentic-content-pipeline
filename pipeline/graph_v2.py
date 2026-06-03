"""V2 graph — the 8-node async pipeline with memory + loop visibility.

    START -> memory_check -> research -> outline -> writer -> fact_check -> seo -> [router]
                                                                                     |
                                              editor (loops back to fact_check) <----+----> publisher -> END

All agent nodes are async (so the FastAPI layer can ``astream`` them); the publisher is sync.
The conditional router (shared with V1, in :mod:`pipeline.graph`) adds the edit-distance stall
gate on top of the threshold + revision-cap guards. Compiled with a ``MemorySaver`` checkpointer
and ``interrupt_before=["publisher_agent"]`` for the human-in-the-loop gate.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from pipeline.agents.editor import editor_agent
from pipeline.agents.fact_check import fact_check_agent
from pipeline.agents.memory_check import memory_check
from pipeline.agents.outline import outline_agent
from pipeline.agents.publisher import publisher_agent
from pipeline.agents.research import research_agent
from pipeline.agents.seo import seo_agent
from pipeline.agents.writer import writer_agent
from pipeline.graph import _default_checkpointer, route_after_seo
from pipeline.state import V2PipelineState

MEMORY = "memory_check"
RESEARCH = "research_agent"
OUTLINE = "outline_agent"
WRITER = "writer_agent"
FACT_CHECK = "fact_check_agent"
SEO = "seo_agent"
EDITOR = "editor_agent"
PUBLISHER = "publisher_agent"


def build_graph_v2(checkpointer=None, *, interrupt: bool = True):
    """Build and compile the 8-node async V2 graph.

    Invoke with ``ainvoke``/``astream`` and ``config={"configurable": {"thread_id": ...}}``; it
    runs to the interrupt before ``publisher_agent``, then resume with ``ainvoke(None, config)``.

    Args:
        interrupt: when True (default) pause before ``publisher_agent`` for the HITL gate; set
            False for batch runs that should auto-publish without a human in the loop.
    """

    builder = StateGraph(V2PipelineState)
    builder.add_node(MEMORY, memory_check)
    builder.add_node(RESEARCH, research_agent)
    builder.add_node(OUTLINE, outline_agent)
    builder.add_node(WRITER, writer_agent)
    builder.add_node(FACT_CHECK, fact_check_agent)
    builder.add_node(SEO, seo_agent)
    builder.add_node(EDITOR, editor_agent)
    builder.add_node(PUBLISHER, publisher_agent)

    builder.add_edge(START, MEMORY)
    builder.add_edge(MEMORY, RESEARCH)
    builder.add_edge(RESEARCH, OUTLINE)
    builder.add_edge(OUTLINE, WRITER)
    builder.add_edge(WRITER, FACT_CHECK)
    builder.add_edge(FACT_CHECK, SEO)
    builder.add_conditional_edges(SEO, route_after_seo, {EDITOR: EDITOR, PUBLISHER: PUBLISHER})
    builder.add_edge(EDITOR, FACT_CHECK)  # re-validate every revision
    builder.add_edge(PUBLISHER, END)

    compile_kwargs = {"checkpointer": checkpointer or _default_checkpointer()}
    if interrupt:
        compile_kwargs["interrupt_before"] = [PUBLISHER]
    return builder.compile(**compile_kwargs)
