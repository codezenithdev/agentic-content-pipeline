"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { RunData } from "@/lib/types";

export default function HitlGate({
  data, onApprove, onReject,
}: {
  data: RunData;
  onApprove: () => void;
  onReject: (note: string) => void;
}) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const flagged = (data.warnings || []).find(
    (w) => w.toLowerCase().includes("needs human review") || w.toLowerCase().includes("stalled"),
  );

  return (
    <div className="rounded-lg border border-amber/40 bg-panel p-4">
      <h3 className="font-mono text-amber">⏸ Awaiting approval</h3>
      {flagged && (
        <div className="mt-2 rounded border border-amber/40 bg-amber/10 p-2 text-xs text-amber">{flagged}</div>
      )}
      <div className="mt-2 flex gap-4 font-mono text-xs text-muted">
        <span>fact-check <b className="text-paper">{data.fact_check_score?.toFixed(2) ?? "—"}</b></span>
        <span>seo <b className="text-paper">{data.seo_score ?? "—"}</b></span>
        <span>revisions <b className="text-paper">{data.revision_count}</b></span>
      </div>
      <div className="prose-editorial mt-3 max-h-72 overflow-auto rounded border border-line bg-ink p-3 text-sm">
        <ReactMarkdown>{data.draft || "_(no draft)_"}</ReactMarkdown>
      </div>
      <div className="mt-3 flex flex-col gap-2 sm:flex-row">
        <button
          disabled={busy}
          onClick={() => { setBusy(true); onApprove(); }}
          className="rounded bg-ok/90 px-4 py-2 font-mono text-sm text-ink hover:bg-ok disabled:opacity-50"
        >
          ✓ Approve &amp; Publish
        </button>
        <input
          value={note} onChange={(e) => setNote(e.target.value)}
          placeholder="feedback for the editor…"
          className="flex-1 rounded border border-line bg-ink px-3 py-2 text-sm outline-none focus:border-amber"
        />
        <button
          disabled={busy}
          onClick={() => { setBusy(true); onReject(note); }}
          className="rounded border border-amber px-4 py-2 font-mono text-sm text-amber hover:bg-amber/10 disabled:opacity-50"
        >
          ↩ Send back for revision
        </button>
      </div>
    </div>
  );
}
