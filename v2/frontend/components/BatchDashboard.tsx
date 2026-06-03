"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { BatchTopic } from "@/lib/types";

function slugOf(path: string | null): string | null {
  if (!path) return null;
  const file = path.split(/[\\/]/).pop() || "";
  return file.replace(/\.md$/, "") || null;
}

export default function BatchDashboard({ batchId, topics }: { batchId: string; topics: string[] }) {
  const [cards, setCards] = useState<Record<string, BatchTopic>>(() =>
    Object.fromEntries(topics.map((t) => [t, { topic: t, status: "pending", final_score: null, output_path: null }])),
  );
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!batchId) return;
    const es = new EventSource(api.batchStreamUrl(batchId));
    es.addEventListener("topic_start", (e) => {
      const ev = JSON.parse((e as MessageEvent).data);
      setCards((c) => (ev.topic ? { ...c, [ev.topic]: { ...c[ev.topic], status: "running" } } : c));
    });
    es.addEventListener("topic_complete", (e) => {
      const d = JSON.parse((e as MessageEvent).data).data;
      setCards((c) => ({ ...c, [d.topic]: { topic: d.topic, status: d.status, final_score: d.final_score, output_path: d.output_path } }));
    });
    es.addEventListener("complete", () => { setDone(true); es.close(); });
    es.addEventListener("error", (e) => { if ((e as MessageEvent).data) { setDone(true); es.close(); } });
    return () => es.close();
  }, [batchId]);

  return (
    <div>
      <div className="mb-3 font-mono text-xs text-muted">{done ? "batch complete" : "running…"}</div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {Object.values(cards).map((c) => {
          const color =
            c.status === "complete" ? "#34D399" : c.status === "failed" ? "#F87171" : c.status === "running" ? "#2DD4BF" : "#26262C";
          const width = c.status === "complete" ? "100%" : c.status === "running" ? "60%" : "8%";
          const slug = slugOf(c.output_path);
          return (
            <div key={c.topic} className="rounded-lg border bg-panel p-3" style={{ borderColor: color }}>
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm">{c.topic}</span>
                <span className="font-mono text-xs" style={{ color }}>{c.status}</span>
              </div>
              <div className="mt-2 h-1 rounded bg-line">
                <div className="h-1 rounded transition-all" style={{ width, background: color }} />
              </div>
              {c.final_score != null && (
                <div className="mt-2 font-mono text-xs text-muted">fact-check {c.final_score.toFixed(2)}</div>
              )}
              {c.status === "complete" && slug && (
                <a href={`/output/${slug}`} className="mt-2 inline-block font-mono text-xs text-teal">view article →</a>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
