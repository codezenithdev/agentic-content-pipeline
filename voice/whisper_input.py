"""voice.whisper_input — transcribe an audio file into a topic string (OpenAI Whisper).

``transcribe(path) -> str`` runs the configured Whisper model over an audio file. Mic capture
is intentionally out of scope (headless-unfriendly); file transcription only. Deterministic in
mock mode. The FastAPI ``/api/voice/transcribe`` route wraps this for the frontend.
"""

from __future__ import annotations

import logging

from pipeline import config, llm

logger = logging.getLogger("voice.whisper")


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file to text (mock returns a deterministic transcript)."""

    client = llm.get_openai_client()
    with open(audio_path, "rb") as handle:
        result = client.audio.transcriptions.create(model=config.WHISPER_MODEL, file=handle)
    text = (getattr(result, "text", "") or "").strip()
    logger.info("whisper: transcribed %s (%d chars)", audio_path, len(text))
    return text
