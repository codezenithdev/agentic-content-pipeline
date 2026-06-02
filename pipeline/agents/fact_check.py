"""fact_check_agent — verify the draft's claims against the researched sources.

Input  (state): ``draft``, ``fact_sheet``, ``sources``
Output (state update): ``fact_check_score`` (0.0-1.0), ``flagged_claims`` (list[FlaggedClaim]), ``warnings``

The agent extracts factual claims from the draft and cross-checks each against the fact
sheet and source snippets (which together represent what the sources actually support).
Claims with no support are likely hallucinations and are flagged with a concrete fix.
Returns a single structured object via ``with_structured_output`` — no regex parsing.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pipeline import config, llm
from pipeline.state import FlaggedClaim, PipelineState, Source


class _FactCheckResult(BaseModel):
    """Structured fact-check output."""

    fact_check_score: float = Field(
        ..., description="Overall support confidence 0.0-1.0 (1.0 = every claim is supported)."
    )
    flagged_claims: list[FlaggedClaim] = Field(
        default_factory=list, description="Claims that are unsupported, contradicted, or exaggerated."
    )


# Deterministic mock-mode response: a clean, passing check.
llm.MOCK_RESPONSES["fact_check"] = {"fact_check_score": 0.95, "flagged_claims": []}

_SYSTEM = (
    "You are a rigorous fact-checker. You are given an ARTICLE, a FACT SHEET (facts tagged with "
    "the source URLs that support them), and a list of SOURCES (title, URL, excerpt).\n"
    "Task:\n"
    "1. Extract every factual claim made in the article (statistics, study findings, dates, named "
    "entities, cause-effect assertions).\n"
    "2. For each claim, decide whether it is SUPPORTED by the fact sheet/sources, or "
    "UNSUPPORTED / CONTRADICTED / EXAGGERATED.\n"
    "3. Flag every claim that is not clearly supported. For each flagged claim give: the claim, the "
    "reason it is flagged, a concrete suggested_fix (rewrite or removal), and source_url (a partially "
    "relevant source URL, or null if none).\n"
    "Scoring: fact_check_score is your overall confidence in [0.0, 1.0] that the whole article is "
    "supported by the sources. 1.0 means every claim is supported; subtract more for each "
    "unsupported claim, and subtract heavily for fabricated statistics or studies."
)


def _format_sources(sources: list[Source]) -> str:
    if not sources:
        return "(no sources available)"
    return "\n".join(f"- {s.title} | {s.url}\n  {s.snippet}" for s in sources)


def fact_check_agent(state: PipelineState) -> dict:
    """Cross-check the draft against sources and return a score + flagged claims (LangGraph node)."""

    draft = state.get("draft", "")
    warnings = list(state.get("warnings", []))

    if not draft.strip():
        warnings.append("Fact-check skipped: empty draft.")
        return {"fact_check_score": 0.0, "flagged_claims": [], "warnings": warnings}

    model = llm.get_llm("fact_check").with_structured_output(_FactCheckResult).with_retry(
        stop_after_attempt=config.LLM_MAX_RETRIES
    )
    result: _FactCheckResult = model.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(
                content=(
                    f"ARTICLE:\n{draft}\n\n"
                    f"FACT SHEET:\n{state.get('fact_sheet', '')}\n\n"
                    f"SOURCES:\n{_format_sources(state.get('sources', []))}"
                )
            ),
        ]
    )

    # Defensive clamp (structured outputs don't strictly enforce numeric bounds).
    score = max(0.0, min(1.0, float(result.fact_check_score)))
    return {
        "fact_check_score": score,
        "flagged_claims": result.flagged_claims,
        "warnings": warnings,
    }
