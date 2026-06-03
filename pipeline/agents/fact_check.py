"""fact_check_agent (V2, async) — RAG-based verification against full-page chunks.

Input  (state): ``draft``, ``full_page_chunks``, ``sources``, ``revision_count``
Output (state update): ``fact_check_score`` (0-1), ``flagged_claims`` (V1 compat),
                       ``flagged_claims_trace`` (list[ClaimTrace]), ``warnings``

Flow: extract every factual claim from the draft; for each claim embed it and retrieve the top-3
most relevant page chunks (cosine over our embeddings); have the LLM verify each claim against its
retrieved evidence. Much more accurate than V1's snippet-only check. Falls back to source snippets
when no chunks are available. Async so the V2 graph can stream.
"""

from __future__ import annotations

import asyncio
import logging
import math

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pipeline import config, llm
from pipeline.state import ClaimTrace, FlaggedClaim, PageChunk, Source, V2PipelineState

logger = logging.getLogger("pipeline.fact_check")

_MAX_CLAIMS = 20
_TOP_K = 3


class _ClaimList(BaseModel):
    claims: list[str] = Field(..., description="Every distinct factual claim made in the article.")


class _ClaimVerdict(BaseModel):
    claim: str
    status: str = Field(..., description='"verified" | "flagged" (unsupported/contradicted).')
    reason: str = Field(..., description="Why verified or flagged.")
    suggested_fix: str = Field(default="", description="Concrete fix if flagged.")
    source_url: str | None = Field(default=None, description="Best supporting source URL, or null.")


class _FactCheckResult(BaseModel):
    fact_check_score: float = Field(..., description="Overall support confidence 0.0-1.0.")
    verdicts: list[_ClaimVerdict] = Field(default_factory=list)


# Schema-specific mock keys so the claim-extraction call and the verify call don't collide;
# the plain "fact_check" key (verify result) stays overridable by tests via monkeypatch.
llm.MOCK_RESPONSES["fact_check:_ClaimList"] = {"claims": ["Mock claim one", "Mock claim two"]}
llm.MOCK_RESPONSES["fact_check"] = {
    "fact_check_score": 0.95,
    "verdicts": [
        {"claim": "Mock claim one", "status": "verified", "reason": "supported by evidence",
         "suggested_fix": "", "source_url": "https://example.com/1"},
        {"claim": "Mock claim two", "status": "flagged", "reason": "no supporting evidence",
         "suggested_fix": "Remove or attribute.", "source_url": None},
    ],
}

_EXTRACT_SYSTEM = (
    "You are a fact-checker. Extract every distinct factual claim from the ARTICLE (statistics, "
    "study findings, dates, named entities, cause-effect assertions). Return them verbatim-ish."
)
_VERIFY_SYSTEM = (
    "You are a rigorous fact-checker. For each CLAIM you are given the top retrieved EVIDENCE "
    "chunks from the source pages. Decide whether each claim is SUPPORTED by its evidence.\n"
    "For each claim return: status ('verified' or 'flagged'), a reason, a suggested_fix if flagged, "
    "and the best supporting source_url (or null).\n"
    "fact_check_score = overall confidence in [0,1] that the whole article is supported; subtract "
    "heavily for fabricated statistics or studies."
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def _retrieve(claims: list[str], chunks: list[PageChunk]) -> dict[str, list[PageChunk]]:
    """Embed claims + chunks and return the top-K chunks per claim (cosine)."""

    if not claims or not chunks:
        return {claim: [] for claim in claims}
    chunk_embeddings = await asyncio.to_thread(llm.embed_texts, [c.chunk_text for c in chunks])
    claim_embeddings = await asyncio.to_thread(llm.embed_texts, claims)
    retrieved: dict[str, list[PageChunk]] = {}
    for claim, claim_emb in zip(claims, claim_embeddings):
        ranked = sorted(
            ((_cosine(claim_emb, ce), i) for i, ce in enumerate(chunk_embeddings)),
            reverse=True,
        )
        retrieved[claim] = [chunks[i] for _, i in ranked[:_TOP_K]]
    return retrieved


def _evidence_block(claims, retrieved, sources: list[Source]) -> str:
    blocks = []
    for i, claim in enumerate(claims, 1):
        ev = retrieved.get(claim, [])
        if ev:
            lines = "\n".join(f"    - ({c.url}) {c.chunk_text[:400]}" for c in ev)
        else:  # fallback to source snippets
            lines = "\n".join(f"    - ({s.url}) {s.snippet[:300]}" for s in sources[:_TOP_K]) or "    (no evidence)"
        blocks.append(f"CLAIM {i}: {claim}\n  EVIDENCE:\n{lines}")
    return "\n\n".join(blocks)


async def fact_check_agent(state: V2PipelineState) -> dict:
    """Verify the draft's claims against retrieved page chunks (async node)."""

    draft = state.get("draft", "")
    warnings = list(state.get("warnings", []))
    round_no = state.get("revision_count", 0)
    if not draft.strip():
        warnings.append("Fact-check skipped: empty draft.")
        return {"fact_check_score": 0.0, "flagged_claims": [], "flagged_claims_trace": [], "warnings": warnings}

    # 1) Extract claims.
    extractor = llm.get_llm("fact_check").with_structured_output(_ClaimList).with_retry(
        stop_after_attempt=config.LLM_MAX_RETRIES
    )
    claim_list: _ClaimList = await extractor.ainvoke(
        [SystemMessage(content=_EXTRACT_SYSTEM), HumanMessage(content=f"ARTICLE:\n{draft}")]
    )
    claims = [c for c in claim_list.claims if c.strip()][:_MAX_CLAIMS]

    # 2) Retrieve evidence per claim (RAG).
    chunks = state.get("full_page_chunks", []) or []
    retrieved = await _retrieve(claims, chunks)

    # 3) Verify against retrieved evidence.
    verifier = llm.get_llm("fact_check").with_structured_output(_FactCheckResult).with_retry(
        stop_after_attempt=config.LLM_MAX_RETRIES
    )
    result: _FactCheckResult = await verifier.ainvoke(
        [
            SystemMessage(content=_VERIFY_SYSTEM),
            HumanMessage(content=_evidence_block(claims, retrieved, state.get("sources", []))),
        ]
    )

    score = max(0.0, min(1.0, float(result.fact_check_score)))
    trace = [
        ClaimTrace(claim=v.claim, status=v.status, original_text=v.claim, revised_text=None,
                   source_url=v.source_url, round_number=round_no)
        for v in result.verdicts
    ]
    flagged = [
        FlaggedClaim(claim=v.claim, reason=v.reason, suggested_fix=v.suggested_fix, source_url=v.source_url)
        for v in result.verdicts if v.status.lower() != "verified"
    ]
    logger.info("fact_check: score=%.2f, %d claims, %d flagged (round %d, %d chunks)",
                score, len(claims), len(flagged), round_no, len(chunks))
    return {
        "fact_check_score": score,
        "flagged_claims": flagged,
        "flagged_claims_trace": trace,
        "warnings": warnings,
    }
