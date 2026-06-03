"""Voice layer tests (mock mode): Whisper transcription, TTS synthesis, publisher audio hook."""

from __future__ import annotations

from pathlib import Path

from pipeline import config
from pipeline.agents.publisher import publisher_agent
from pipeline.state import Source, initial_v2_state
from voice import tts_output, whisper_input


def test_transcribe_returns_text(tmp_path):
    audio = tmp_path / "topic.mp3"
    audio.write_bytes(b"fake-audio-bytes")
    text = whisper_input.transcribe(str(audio))
    assert isinstance(text, str) and text


def test_synthesize_writes_mp3(tmp_path):
    out = tmp_path / "summary.mp3"
    path = tts_output.synthesize("This is a spoken summary of the article.", str(out))
    assert Path(path).exists() and out.stat().st_size > 0


def _drafted_voice_state(voice_path: str) -> dict:
    st = initial_v2_state("AI agents", "langgraph", voice_input_path=voice_path)
    st["draft"] = "# Title\n\n## Intro\nSentence one. Sentence two. Sentence three. Sentence four."
    st["sources"] = [Source(url="https://example.com/1", title="S", retrieved_at="now",
                            snippet="snippet", credibility_note="ok")]
    st["seo_feedback"] = {"meta_description": "A concise summary."}
    st["fact_check_score"] = 0.9
    st["seo_score"] = 80
    return st


def test_publisher_generates_audio_on_voice_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    out = publisher_agent(_drafted_voice_state("/tmp/in.mp3"))
    assert out["approved"] is True
    assert out["audio_summary_path"] and Path(out["audio_summary_path"]).exists()
    assert out["output_files"].get("audio")


def test_publisher_skips_audio_without_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    st = _drafted_voice_state("/tmp/in.mp3")
    st["voice_input_path"] = None
    out = publisher_agent(st)
    assert out["audio_summary_path"] is None
    assert "audio" not in out["output_files"]
