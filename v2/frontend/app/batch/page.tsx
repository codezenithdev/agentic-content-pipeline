"use client";

import { useState } from "react";
import BatchDashboard from "@/components/BatchDashboard";
import { api } from "@/lib/api";

export default function BatchPage() {
  const [raw, setRaw] = useState("AI agents\nMediterranean diet\nQuantum computing");
  const [keyword, setKeyword] = useState("");
  const [batchId, setBatchId] = useState<string | null>(null);
  const [topics, setTopics] = useState<string[]>([]);

  async function start() {
    const ts = raw.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!ts.length) return;
    setTopics(ts);
    const { batch_id } = await api.startBatch({ topics: ts, target_keyword: keyword, style_persona: "technical deep-dive" });
    setBatchId(batch_id);
  }

  return (
    <main className="mx-auto max-w-5xl p-6">
      <header className="mb-4 flex items-center justify-between">
        <h1 className="font-mono text-lg">multi-topic batch <span className="text-muted">(max 3 parallel)</span></h1>
        <a href="/" className="font-mono text-xs text-teal">← dashboard</a>
      </header>

      {!batchId ? (
        <section className="rounded-lg border border-line bg-panel p-5">
          <label className="text-xs text-muted">Topics (one per line)
            <textarea value={raw} onChange={(e) => setRaw(e.target.value)} rows={5}
              className="mt-1 w-full rounded border border-line bg-ink p-2 text-sm outline-none focus:border-teal" />
          </label>
          <input value={keyword} onChange={(e) => setKeyword(e.target.value)} placeholder="target keyword (optional)"
            className="mt-3 w-full rounded border border-line bg-ink p-2 text-sm outline-none focus:border-teal" />
          <button onClick={start} className="mt-3 rounded bg-teal px-4 py-2 font-mono text-sm text-ink hover:opacity-90">▶ Run batch</button>
        </section>
      ) : (
        <BatchDashboard batchId={batchId} topics={topics} />
      )}
    </main>
  );
}
