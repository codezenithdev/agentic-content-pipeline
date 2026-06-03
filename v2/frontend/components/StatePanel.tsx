"use client";

import * as Tabs from "@radix-ui/react-tabs";
import ReactMarkdown from "react-markdown";
import { RunData } from "@/lib/types";
import RevisionTimeline from "./RevisionTimeline";

function credColor(s: number) {
  return s >= 0.7 ? "#34D399" : s >= 0.4 ? "#F59E0B" : "#F87171";
}

const tabCls =
  "px-3 py-2 font-mono text-xs text-muted data-[state=active]:text-teal data-[state=active]:border-b-2 data-[state=active]:border-teal";

export default function StatePanel({ data }: { data: RunData }) {
  return (
    <Tabs.Root defaultValue="sources" className="rounded-lg border border-line bg-panel">
      <Tabs.List className="flex gap-1 border-b border-line px-2">
        <Tabs.Trigger value="sources" className={tabCls}>Sources</Tabs.Trigger>
        <Tabs.Trigger value="draft" className={tabCls}>Draft</Tabs.Trigger>
        <Tabs.Trigger value="fact" className={tabCls}>Fact Check</Tabs.Trigger>
        <Tabs.Trigger value="seo" className={tabCls}>SEO</Tabs.Trigger>
        <Tabs.Trigger value="rev" className={tabCls}>Revisions</Tabs.Trigger>
      </Tabs.List>
      <div className="max-h-[430px] overflow-auto p-3">
        <Tabs.Content value="sources">
          {data.memory_hit && (
            <div className="mb-2 rounded bg-teal/10 px-2 py-1 font-mono text-xs text-teal">
              memory hit — reused {data.reused_sources} source(s)
            </div>
          )}
          {data.sources.length === 0 ? (
            <p className="text-sm text-muted">No sources yet.</p>
          ) : (
            <table className="w-full text-xs">
              <tbody>
                {data.sources.map((s) => {
                  const c = data.credibility_scores[s.url];
                  return (
                    <tr key={s.url} className="border-b border-line/60 align-top">
                      <td className="py-1.5 pr-2">
                        <a href={s.url} target="_blank" rel="noreferrer" className="text-paper hover:text-teal">{s.title}</a>
                        <div className="text-muted">{s.credibility_note}</div>
                      </td>
                      <td className="py-1.5 text-right">
                        {c != null && (
                          <span className="rounded px-1.5 py-0.5 font-mono"
                            style={{ background: `${credColor(c)}22`, color: credColor(c) }}>
                            {c.toFixed(2)}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </Tabs.Content>

        <Tabs.Content value="draft" className="prose-editorial text-sm">
          {data.draft ? <ReactMarkdown>{data.draft}</ReactMarkdown> : <p className="text-muted">No draft yet.</p>}
        </Tabs.Content>

        <Tabs.Content value="fact">
          <p className="mb-2 font-mono text-xs text-muted">
            score <b className="text-paper">{data.fact_check_score?.toFixed(2) ?? "—"}</b>
          </p>
          {data.flagged_claims_trace.length === 0 ? (
            <p className="text-sm text-muted">No claims traced yet.</p>
          ) : (
            data.flagged_claims_trace.map((t, i) => {
              const col = t.status === "verified" ? "#34D399" : t.status === "flagged" ? "#F87171" : "#F59E0B";
              return (
                <div key={i} className="mb-2 rounded border border-line bg-elevated p-2 text-xs">
                  <span className="font-mono" style={{ color: col }}>{t.status}</span> · <span>{t.claim}</span>
                  {t.source_url && (
                    <a href={t.source_url} target="_blank" rel="noreferrer" className="ml-1 text-teal">[src]</a>
                  )}
                </div>
              );
            })
          )}
        </Tabs.Content>

        <Tabs.Content value="seo">
          {!data.seo_feedback ? (
            <p className="text-sm text-muted">No SEO analysis yet.</p>
          ) : (
            <div className="text-xs">
              <div className="mb-2 flex flex-wrap gap-4 font-mono">
                <span>score <b className="text-paper">{data.seo_score}</b></span>
                <span>density <b className="text-paper">{String(data.seo_feedback.keyword_density_pct ?? "—")}%</b></span>
                <span>readability <b className="text-paper">{String(data.seo_feedback.readability_score ?? "—")}</b></span>
              </div>
              {(data.seo_feedback.fix_now || []).length > 0 && (
                <>
                  <div className="font-mono text-coral">fix now</div>
                  <div className="mb-2 mt-1 flex flex-wrap gap-1">
                    {(data.seo_feedback.fix_now || []).map((f, i) => (
                      <span key={i} className="rounded bg-coral/15 px-1.5 py-0.5 text-coral">{f}</span>
                    ))}
                  </div>
                </>
              )}
              {(data.seo_feedback.fix_later || []).length > 0 && (
                <>
                  <div className="font-mono text-muted">fix later</div>
                  <ul className="ml-4 list-disc text-muted">
                    {(data.seo_feedback.fix_later || []).map((f, i) => <li key={i}>{f}</li>)}
                  </ul>
                </>
              )}
              {data.seo_feedback.meta_description && (
                <p className="mt-2 text-muted"><b className="text-paper">meta:</b> {data.seo_feedback.meta_description}</p>
              )}
            </div>
          )}
        </Tabs.Content>

        <Tabs.Content value="rev">
          <RevisionTimeline rounds={data.revision_history} />
        </Tabs.Content>
      </div>
    </Tabs.Root>
  );
}
