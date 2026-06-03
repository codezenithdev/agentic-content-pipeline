"use client";

import { RevisionRound } from "@/lib/types";

export default function RevisionTimeline({ rounds }: { rounds: RevisionRound[] }) {
  if (!rounds.length)
    return (
      <p className="text-sm text-muted">
        No revisions yet — the draft either passed on the first pass or hasn&apos;t looped.
      </p>
    );
  return (
    <div className="space-y-3">
      {rounds.map((r) => (
        <div key={r.round_number} className="rounded border border-line bg-elevated p-3">
          <div className="flex items-center justify-between font-mono text-xs">
            <span className="text-teal">Round {r.round_number}</span>
            <span className="text-muted">edit distance {r.edit_distance.toFixed(3)}</span>
          </div>
          <div className="mt-2 h-1.5 w-full rounded bg-line">
            <div className="h-1.5 rounded bg-teal" style={{ width: `${Math.min(100, r.edit_distance * 100)}%` }} />
          </div>
          <div className="mt-2 flex gap-4 font-mono text-xs">
            <span>fact-check <b className="text-paper">{r.fact_check_score.toFixed(2)}</b></span>
            <span>seo <b className="text-paper">{r.seo_score}</b></span>
          </div>
          {r.sections_changed.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {r.sections_changed.map((s) => (
                <span key={s} className="rounded bg-amber/15 px-1.5 py-0.5 font-mono text-[10px] text-amber">{s}</span>
              ))}
            </div>
          )}
          <p className="mt-2 text-xs text-muted">{r.diff_summary}</p>
        </div>
      ))}
    </div>
  );
}
