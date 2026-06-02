"""Command-line runner for the content pipeline, with a human-in-the-loop approval gate.

Usage:
    python run_cli.py "<topic>" "<target keyword>"        # interactive y/N prompt
    python run_cli.py "<topic>" "<keyword>" --yes          # auto-approve (non-interactive)
    python run_cli.py "<topic>" "<keyword>" --no           # auto-reject (writes nothing)

Runs the graph up to the HITL interrupt (before the publisher), prints the draft and the
final scores, then asks whether to publish. On approval it resumes the graph so the publisher
exports the .md / .html / .meta.json files; on rejection it stops and writes nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid

from pipeline.graph import build_graph
from pipeline.state import initial_state


def _print_report(state: dict) -> None:
    """Print the draft plus the fact-check / SEO results gathered before the gate."""

    print("\n" + "=" * 70)
    print("DRAFT")
    print("=" * 70)
    print(state.get("draft", "(empty)"))

    print("\n" + "=" * 70)
    print("QUALITY REPORT")
    print("=" * 70)
    print(f"revision_count   : {state.get('revision_count', 0)}")
    print(f"fact_check_score : {state.get('fact_check_score')}")
    print(f"seo_score        : {state.get('seo_score')}")

    flagged = state.get("flagged_claims", [])
    print(f"flagged_claims   : {len(flagged)}")
    for fc in flagged:
        print(f"  - {fc.claim[:80]} | {fc.reason[:60]}")

    fb = state.get("seo_feedback") or {}
    if fb:
        print(f"keyword density  : {fb.get('keyword_density_pct')}%  "
              f"(occurrences {fb.get('keyword_occurrences')})")
        print(f"readability      : {fb.get('readability_score')} (Flesch)")
        print(f"meta description : {fb.get('meta_description', '')}")

    for w in state.get("warnings", []):
        print(f"  ! warning: {w}")


def _decide(args: argparse.Namespace) -> bool:
    """Resolve the approval decision from flags or an interactive prompt."""

    if args.yes:
        return True
    if args.no:
        return False
    try:
        answer = input("\nApprove & publish? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agentic content pipeline (CLI).")
    parser.add_argument("topic", help="The topic to research and write about.")
    parser.add_argument("target_keyword", help="The SEO target keyword.")
    parser.add_argument("--yes", action="store_true", help="Auto-approve publication.")
    parser.add_argument("--no", action="store_true", help="Auto-reject publication.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    graph = build_graph()
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}

    # Run up to the HITL interrupt (before the publisher node).
    state = graph.invoke(initial_state(args.topic, args.target_keyword), cfg)
    _print_report(state)

    snapshot = graph.get_state(cfg)
    if snapshot.next != ("publisher_agent",):
        print("\nPipeline did not reach the publication gate; nothing to approve.")
        return 1

    if not _decide(args):
        print("\nRejected. No files written.")
        return 0

    final = graph.invoke(None, cfg)  # resume past the interrupt = approval
    files = final.get("output_files") or {}
    print("\nPublished. Files written:")
    for kind, path in files.items():
        print(f"  {kind:>4}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
