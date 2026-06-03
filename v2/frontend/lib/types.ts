export type NodeStatus = "idle" | "running" | "complete" | "looped" | "hitl";

export const NODE_IDS = [
  "memory_check",
  "research_agent",
  "outline_agent",
  "writer_agent",
  "fact_check_agent",
  "seo_agent",
  "editor_agent",
  "publisher_agent",
] as const;
export type NodeId = (typeof NODE_IDS)[number];

export interface Source {
  url: string;
  title: string;
  retrieved_at: string;
  snippet: string;
  credibility_note: string;
}

export interface ClaimTrace {
  claim: string;
  status: string;
  original_text: string;
  revised_text: string | null;
  source_url: string | null;
  round_number: number;
}

export interface RevisionRound {
  round_number: number;
  fact_check_score: number;
  seo_score: number;
  diff_summary: string;
  sections_changed: string[];
  edit_distance: number;
}

export interface SeoFeedback {
  keyword_issues?: string;
  heading_issues?: string;
  meta_description?: string;
  readability_score?: number;
  keyword_density_pct?: number;
  title_tag?: string;
  fix_now?: string[];
  fix_later?: string[];
  suggestions?: string[];
  [k: string]: unknown;
}

export interface RunData {
  memory_hit?: boolean;
  reused_sources?: number;
  sources: Source[];
  credibility_scores: Record<string, number>;
  outline: string;
  draft: string;
  self_critique: string;
  fact_check_score: number | null;
  flagged_claims_trace: ClaimTrace[];
  seo_score: number | null;
  seo_feedback: SeoFeedback | null;
  revision_history: RevisionRound[];
  revision_count: number;
  warnings: string[];
  title?: string;
  output_paths?: Record<string, string>;
}

export interface BatchTopic {
  topic: string;
  status: "pending" | "running" | "complete" | "failed";
  final_score: number | null;
  output_path: string | null;
}
