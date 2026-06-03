"""FastAPI application for the V2 pipeline — SSE streaming, HITL, batch, voice, memory.

Run with:  uvicorn v2.backend.main:app --port 8000 --reload   (from the project root)

Routes (all under /api):
  POST   /run                      start a single-topic run -> {run_id}
  GET    /run/{id}/stream          SSE: node_start / node_complete / revision_round / hitl_required / complete
  POST   /run/{id}/approve         resume past the HITL gate -> publish
  POST   /run/{id}/reject          send back to the editor with a human feedback note
  GET    /run/{id}/output          {md_url, html_url, audio_url, meta_json}
  GET    /output/{slug}/audio      stream the summary mp3
  GET    /output/{slug}/{filename} serve a generated file
  POST   /batch                    start a multi-topic batch -> {batch_id}
  GET    /batch/{id}/stream        SSE: batch_start / topic_start / topic_complete / complete
  POST   /voice/transcribe         multipart audio -> {transcript}
  GET    /memory/search?topic=...  past similar runs (list[MemoryHit])
  DELETE /memory/{slug}            remove a run from memory

The routes are consolidated here for the demo; the ``routers/`` package holds placeholders for a
future split. In-flight run state lives in :mod:`v2.backend.stream` (in-memory, keyed by id).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from memory import store
from pipeline import config
from v2.backend import stream as S
from voice import whisper_input

app = FastAPI(title="Agentic Content Pipeline V2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_or_404(run_id: str) -> S.RunState:
    rs = S.RUNS.get(run_id)
    if rs is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id {run_id}")
    return rs


# --------------------------------------------------------------------------------------
# Single-topic run
# --------------------------------------------------------------------------------------
class RunRequest(BaseModel):
    topic: str
    target_keyword: str
    style_persona: str = "technical deep-dive"
    voice_input_path: str | None = None


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "mock_mode": config.MOCK_MODE}


@app.post("/api/run")
async def start_run(req: RunRequest) -> dict:
    run_id = uuid.uuid4().hex
    S.start_run(run_id, req.topic, req.target_keyword, req.style_persona, req.voice_input_path)
    return {"run_id": run_id}


@app.get("/api/run/{run_id}/stream")
async def run_stream(run_id: str) -> EventSourceResponse:
    return EventSourceResponse(S.event_stream(_run_or_404(run_id)))


@app.post("/api/run/{run_id}/approve")
async def approve_run(run_id: str) -> dict:
    await S.approve(_run_or_404(run_id))
    return {"status": "approved"}


class RejectRequest(BaseModel):
    note: str = ""


@app.post("/api/run/{run_id}/reject")
async def reject_run(run_id: str, body: RejectRequest) -> dict:
    await S.reject(_run_or_404(run_id), body.note)
    return {"status": "revising"}


@app.get("/api/run/{run_id}/output")
async def run_output(run_id: str) -> dict:
    rs = _run_or_404(run_id)
    values = rs.graph.get_state(rs.cfg).values
    files = values.get("output_files") or {}
    slug = values.get("slug", "")

    def url(kind: str) -> str | None:
        path = files.get(kind)
        return f"/api/output/{slug}/{Path(path).name}" if path else None

    meta_json = None
    if files.get("meta") and Path(files["meta"]).exists():
        meta_json = json.loads(Path(files["meta"]).read_text(encoding="utf-8"))
    return {"slug": slug, "md_url": url("md"), "html_url": url("html"),
            "audio_url": url("audio"), "meta_json": meta_json}


# --------------------------------------------------------------------------------------
# Output file serving (specific /audio route registered before the generic one)
# --------------------------------------------------------------------------------------
@app.get("/api/output/{slug}/audio")
async def serve_audio(slug: str) -> FileResponse:
    path = Path(config.OUTPUT_DIR) / f"{slug}_summary.mp3"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no audio summary for this slug")
    return FileResponse(str(path), media_type="audio/mpeg")


@app.get("/api/output/{slug}/{filename}")
async def serve_output(slug: str, filename: str) -> FileResponse:
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    path = Path(config.OUTPUT_DIR) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(path))


# --------------------------------------------------------------------------------------
# Multi-topic batch
# --------------------------------------------------------------------------------------
class BatchRequest(BaseModel):
    topics: list[str]
    target_keyword: str = ""
    style_persona: str = "technical deep-dive"


@app.post("/api/batch")
async def start_batch(req: BatchRequest) -> dict:
    batch_id = uuid.uuid4().hex
    S.start_batch(batch_id, req.topics, req.target_keyword, req.style_persona)
    return {"batch_id": batch_id}


@app.get("/api/batch/{batch_id}/stream")
async def batch_stream(batch_id: str) -> EventSourceResponse:
    bs = S.BATCHES.get(batch_id)
    if bs is None:
        raise HTTPException(status_code=404, detail=f"unknown batch_id {batch_id}")
    return EventSourceResponse(S.batch_event_stream(bs))


# --------------------------------------------------------------------------------------
# Voice
# --------------------------------------------------------------------------------------
@app.post("/api/voice/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"
    tmp = Path(tempfile.gettempdir()) / f"upload_{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(await file.read())
    try:
        transcript = await asyncio.to_thread(whisper_input.transcribe, str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    return {"transcript": transcript}


# --------------------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------------------
@app.get("/api/memory/search")
async def memory_search(topic: str) -> dict:
    hits = await asyncio.to_thread(store.query_similar, topic, 0.0)  # ranked, unthresholded
    return {"results": [h.model_dump() for h in hits]}


@app.delete("/api/memory/{slug}")
async def memory_delete(slug: str) -> dict:
    await asyncio.to_thread(store.delete_run, slug)
    return {"status": "deleted", "slug": slug}
