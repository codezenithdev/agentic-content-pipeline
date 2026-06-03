"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { NodeStatus, RunData } from "@/lib/types";

const EMPTY: RunData = {
  sources: [], credibility_scores: {}, outline: "", draft: "", self_critique: "",
  fact_check_score: null, flagged_claims_trace: [], seo_score: null, seo_feedback: null,
  revision_history: [], revision_count: 0, warnings: [],
};

export type Phase = "idle" | "running" | "awaiting_approval" | "complete" | "error";

export interface RunStreamState {
  nodeStatus: Record<string, NodeStatus>;
  loopCounts: Record<string, number>;
  data: RunData;
  phase: Phase;
  error?: string;
}

function mergeNode(d: RunData, node: string, p: Record<string, unknown>): RunData {
  const nd: RunData = { ...d };
  if (node === "memory_check") { nd.memory_hit = p.memory_hit as boolean; nd.reused_sources = p.reused_sources as number; }
  if (node === "research_agent") { nd.sources = (p.sources as RunData["sources"]) || []; nd.credibility_scores = (p.credibility_scores as Record<string, number>) || {}; }
  if (node === "outline_agent") nd.outline = (p.outline as string) || "";
  if (node === "writer_agent") { nd.draft = (p.draft as string) || nd.draft; nd.self_critique = (p.self_critique as string) || ""; }
  if (node === "fact_check_agent") { nd.fact_check_score = (p.fact_check_score as number) ?? nd.fact_check_score; nd.flagged_claims_trace = (p.flagged_claims_trace as RunData["flagged_claims_trace"]) || []; }
  if (node === "seo_agent") { nd.seo_score = (p.seo_score as number) ?? nd.seo_score; nd.seo_feedback = (p.seo_feedback as RunData["seo_feedback"]) || nd.seo_feedback; }
  if (node === "editor_agent") { nd.revision_count = (p.revision_count as number) ?? nd.revision_count; nd.revision_history = (p.revision_history as RunData["revision_history"]) || nd.revision_history; }
  if (node === "publisher_agent") { nd.title = p.title as string; nd.output_paths = (p.output_files as Record<string, string>) || nd.output_paths; }
  return nd;
}

export function useRunStream(runId: string | null): RunStreamState {
  const [state, setState] = useState<RunStreamState>({ nodeStatus: {}, loopCounts: {}, data: EMPTY, phase: "idle" });
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!runId) return;
    const es = new EventSource(api.runStreamUrl(runId));
    esRef.current = es;
    setState((s) => ({ ...s, phase: "running" }));

    es.addEventListener("node_start", (e) => {
      const ev = JSON.parse((e as MessageEvent).data);
      setState((s) => {
        const ns = { ...s.nodeStatus };
        const lc = { ...s.loopCounts };
        if (ns[ev.node] === "complete") lc[ev.node] = (lc[ev.node] || 1) + 1; // re-entry => loop
        ns[ev.node] = "running";
        return { ...s, nodeStatus: ns, loopCounts: lc };
      });
    });

    es.addEventListener("node_complete", (e) => {
      const ev = JSON.parse((e as MessageEvent).data);
      setState((s) => ({
        ...s,
        nodeStatus: { ...s.nodeStatus, [ev.node]: "complete" as NodeStatus },
        data: mergeNode(s.data, ev.node, ev.data || {}),
      }));
    });

    es.addEventListener("hitl_required", (e) => {
      const sm = JSON.parse((e as MessageEvent).data).data || {};
      setState((s) => ({
        ...s, phase: "awaiting_approval",
        nodeStatus: { ...s.nodeStatus, publisher_agent: "hitl" as NodeStatus },
        data: { ...s.data, draft: sm.draft ?? s.data.draft, fact_check_score: sm.fact_check_score ?? s.data.fact_check_score,
          seo_score: sm.seo_score ?? s.data.seo_score, revision_count: sm.revision_count ?? s.data.revision_count,
          seo_feedback: sm.seo_feedback ?? s.data.seo_feedback, flagged_claims_trace: sm.flagged_claims_trace ?? s.data.flagged_claims_trace,
          revision_history: sm.revision_history ?? s.data.revision_history, warnings: sm.warnings ?? s.data.warnings,
          title: sm.title ?? s.data.title },
      }));
    });

    es.addEventListener("complete", (e) => {
      const ev = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, phase: "complete",
        nodeStatus: { ...s.nodeStatus, publisher_agent: "complete" as NodeStatus },
        data: { ...s.data, output_paths: ev.output_paths || s.data.output_paths } }));
      es.close();
    });

    es.addEventListener("error", (e) => {
      const data = (e as MessageEvent).data; // custom error events carry data; connection blips don't
      if (!data) return;
      setState((s) => ({ ...s, phase: "error", error: JSON.parse(data).message }));
      es.close();
    });

    return () => es.close();
  }, [runId]);

  return state;
}
