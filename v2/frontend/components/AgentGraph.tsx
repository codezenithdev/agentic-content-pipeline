"use client";

import { useMemo } from "react";
import ReactFlow, {
  Background, Controls, Handle, Position, type Edge, type Node, type NodeProps,
} from "reactflow";
import "reactflow/dist/style.css";
import { motion } from "framer-motion";
import { NodeStatus } from "@/lib/types";

const LABELS: Record<string, string> = {
  memory_check: "Memory", research_agent: "Research", outline_agent: "Outline",
  writer_agent: "Writer", fact_check_agent: "Fact-check", seo_agent: "SEO",
  editor_agent: "Editor", publisher_agent: "Publisher",
};
const ACCENT: Record<string, string> = {
  memory_check: "#9A968C", research_agent: "#2DD4BF", outline_agent: "#2DD4BF",
  writer_agent: "#A78BFA", fact_check_agent: "#F87171", seo_agent: "#F87171",
  editor_agent: "#F59E0B", publisher_agent: "#34D399",
};
const POS: Record<string, { x: number; y: number }> = {
  memory_check: { x: 250, y: 0 }, research_agent: { x: 40, y: 95 },
  outline_agent: { x: 250, y: 95 }, writer_agent: { x: 460, y: 95 },
  fact_check_agent: { x: 40, y: 200 }, seo_agent: { x: 250, y: 200 },
  editor_agent: { x: 460, y: 200 }, publisher_agent: { x: 250, y: 305 },
};
const EDGES: [string, string][] = [
  ["memory_check", "research_agent"], ["research_agent", "outline_agent"],
  ["outline_agent", "writer_agent"], ["writer_agent", "fact_check_agent"],
  ["fact_check_agent", "seo_agent"], ["seo_agent", "editor_agent"],
  ["editor_agent", "fact_check_agent"], ["seo_agent", "publisher_agent"],
];

type AData = { label: string; status: NodeStatus; accent: string; loop: number };

function AgentNodeView({ data }: NodeProps<AData>) {
  const cls = data.status === "running" ? "node-running" : data.status === "hitl" ? "node-hitl" : "";
  const border =
    data.status === "complete" ? data.accent
    : data.status === "hitl" ? "#F59E0B"
    : data.status === "running" ? data.accent
    : "#26262C";
  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0.5 }} animate={{ scale: 1, opacity: 1 }}
      className={`rounded-lg border bg-elevated px-3 py-2 text-xs ${cls}`}
      style={{ borderColor: border, minWidth: 124 }}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono" style={{ color: data.accent }}>{data.label}</span>
        {data.status === "complete" && <span className="text-ok">✓</span>}
        {data.status === "running" && <span className="text-teal">●</span>}
        {data.status === "hitl" && <span className="text-amber">⏸</span>}
      </div>
      {data.loop > 1 && (
        <div className="mt-1 inline-block rounded bg-amber/20 px-1 font-mono text-[10px] text-amber">↻ {data.loop}</div>
      )}
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </motion.div>
  );
}
const nodeTypes = { agent: AgentNodeView };

export default function AgentGraph({
  nodeStatus, loopCounts, onSelect,
}: {
  nodeStatus: Record<string, NodeStatus>;
  loopCounts: Record<string, number>;
  onSelect: (id: string) => void;
}) {
  const nodes: Node<AData>[] = useMemo(
    () => Object.keys(POS).map((id) => ({
      id, type: "agent", position: POS[id],
      data: { label: LABELS[id], status: nodeStatus[id] || "idle", accent: ACCENT[id], loop: loopCounts[id] || 0 },
    })),
    [nodeStatus, loopCounts],
  );
  const edges: Edge[] = useMemo(
    () => EDGES.map(([s, t], i) => {
      const active = nodeStatus[s] === "complete" && nodeStatus[t] === "running";
      return { id: `${s}-${t}-${i}`, source: s, target: t, animated: active,
        className: active ? "edge-active" : "", style: { stroke: "#26262C" } };
    }),
    [nodeStatus],
  );
  return (
    <div style={{ height: 430 }} className="rounded-lg border border-line bg-panel">
      <ReactFlow
        nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView
        onNodeClick={(_, n) => onSelect(n.id)} proOptions={{ hideAttribution: true }}
        nodesDraggable={false} nodesConnectable={false}
      >
        <Background color="#1C1C22" gap={18} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
