"use client";

import { useParams } from "next/navigation";
import { useState } from "react";
import AgentGraph from "@/components/AgentGraph";
import HitlGate from "@/components/HitlGate";
import StatePanel from "@/components/StatePanel";
import { useRunStream } from "@/hooks/useRunStream";
import { api } from "@/lib/api";

const PHASE_COLOR: Record<string, string> = {
  running: "#2DD4BF", awaiting_approval: "#F59E0B", complete: "#34D399", error: "#F87171", idle: "#9A968C",
};

function slugFromPath(p?: string): string {
  return (p || "").split(/[\\/]/).pop()?.replace(/\.md$/, "") || "";
}

export default function RunPage() {
  const id = String(useParams().id);
  const { nodeStatus, loopCounts, data, phase, error } = useRunStream(id);
  const [, setSelected] = useState<string | null>(null);

  return (
    <main className="mx-auto max-w-6xl p-6">
      <header className="mb-4 flex items-center justify-between">
        <h1 className="font-mono text-lg">run <span className="text-muted">{id.slice(0, 8)}</span></h1>
        <span className="font-mono text-xs" style={{ color: PHASE_COLOR[phase] }}>{phase.replace("_", " ")}</span>
      </header>

      {error && <div className="mb-4 rounded border border-coral/50 bg-coral/10 p-3 text-sm text-coral">{error}</div>}

      <div className="grid gap-4 lg:grid-cols-2">
        <AgentGraph nodeStatus={nodeStatus} loopCounts={loopCounts} onSelect={setSelected} />
        <StatePanel data={data} />
      </div>

      {phase === "awaiting_approval" && (
        <div className="mt-4">
          <HitlGate data={data} onApprove={() => api.approve(id)} onReject={(note) => api.reject(id, note)} />
        </div>
      )}

      {phase === "complete" && (
        <div className="mt-4 rounded-lg border border-ok/40 bg-panel p-4">
          <p className="font-mono text-ok">✓ published{data.title ? `: ${data.title}` : ""}</p>
          {data.output_paths?.md && (
            <a href={`/output/${slugFromPath(data.output_paths.md)}`} className="font-mono text-xs text-teal">open article →</a>
          )}
        </div>
      )}
    </main>
  );
}
