"""voice.tts_output — synthesize an article summary to an mp3 (OpenAI TTS).

``synthesize(text, out_path) -> str`` generates speech with the configured TTS model/voice and
writes it to ``out_path``. Mock mode writes a tiny deterministic stub so the publisher + tests
run offline. The publisher calls this on approval when the run started from voice input.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline import config, llm

logger = logging.getLogger("voice.tts")


def synthesize(text: str, out_path: str) -> str:
    """Generate speech for ``text`` and write it to ``out_path`` (returns the path)."""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    client = llm.get_openai_client()
    response = client.audio.speech.create(
        model=config.TTS_MODEL, voice=config.TTS_VOICE, input=text
    )
    if hasattr(response, "stream_to_file"):
        response.stream_to_file(out_path)
    elif hasattr(response, "write_to_file"):
        response.write_to_file(out_path)
    else:  # pragma: no cover - fallback for SDK shape changes
        with open(out_path, "wb") as handle:
            handle.write(response.content)
    logger.info("tts: wrote %s", out_path)
    return out_path
