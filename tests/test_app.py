"""Streamlit UI tests using the official ``AppTest`` harness (headless, mock mode).

Drives the same Run -> approve -> publish flow a user would, asserting on session_state phase
transitions, so the app's wiring (streaming, the HITL gate, the resume) is covered without a
browser.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from pipeline import config

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_initial_render():
    at = AppTest.from_file(APP).run(timeout=60)
    assert not at.exception
    assert any("Run pipeline" in b.label for b in at.button)


def test_app_run_approve_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    at = AppTest.from_file(APP).run(timeout=120)

    next(b for b in at.button if "Run pipeline" in b.label).click()
    at.run(timeout=120)  # executes the pipeline; st.rerun() sets phase -> "approve"
    at.run(timeout=120)  # render the approval gate
    assert at.session_state["phase"] == "approve"

    next(b for b in at.button if "Approve" in b.label).click()
    at.run(timeout=120)  # publisher writes files; st.rerun() sets phase -> "published"
    at.run(timeout=120)  # render the published view
    assert at.session_state["phase"] == "published"
    assert at.session_state["final_state"]["approved"] is True
