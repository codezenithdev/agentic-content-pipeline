"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { API_BASE } from "@/lib/api";

function Row({ k, v }: { k: string; v: unknown }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-muted">{k}</dt>
      <dd className="font-mono text-paper">{String(v ?? "—")}</dd>
    </div>
  );
}

export default function OutputViewer({ slug }: { slug: string }) {
  const [meta, setMeta] = useState<Record<string, unknown> | null>(null);
  const [md, setMd] = useState("");
  const [hasAudio, setHasAudio] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const m = await fetch(`${API_BASE}/api/output/${slug}/${slug}.meta.json`);
        if (m.ok) setMeta(await m.json());
      } catch { /* ignore */ }
      try {
        const t = await fetch(`${API_BASE}/api/output/${slug}/${slug}.md`);
        if (t.ok) setMd(await t.text());
      } catch { /* ignore */ }
    })();
  }, [slug]);

  const fileUrl = (ext: string) => `${API_BASE}/api/output/${slug}/${slug}.${ext}`;

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
      <div className="rounded-lg border border-line bg-panel p-5">
        <div className="prose-editorial">
          {md ? <ReactMarkdown>{md}</ReactMarkdown> : <p className="text-muted">Loading article…</p>}
        </div>
      </div>
      <aside className="space-y-3">
        <div className="rounded-lg border border-line bg-panel p-4">
          <div className="mb-3 flex flex-wrap gap-2">
            <a href={fileUrl("md")} download className="rounded border border-line px-3 py-1.5 font-mono text-xs hover:border-teal">⬇ .md</a>
            <a href={fileUrl("html")} download className="rounded border border-line px-3 py-1.5 font-mono text-xs hover:border-teal">⬇ .html</a>
          </div>
          {hasAudio && (
            <audio controls src={`${API_BASE}/api/output/${slug}/audio`} onError={() => setHasAudio(false)} className="w-full" />
          )}
        </div>
        {meta && (
          <div className="rounded-lg border border-line bg-panel p-4 text-xs">
            <h3 className="mb-2 font-mono text-teal">meta</h3>
            <dl className="space-y-1">
              <Row k="keyword" v={meta.target_keyword} />
              <Row k="persona" v={meta.style_persona} />
              <Row k="fact-check" v={meta.fact_check_score} />
              <Row k="seo" v={meta.seo_score} />
              <Row k="revisions" v={meta.revision_count} />
              <Row k="sources" v={(meta.sources as unknown[] | undefined)?.length} />
              <Row k="run date" v={String(meta.generated_at ?? "").slice(0, 10)} />
            </dl>
          </div>
        )}
        <div className="rounded-lg border border-line bg-panel p-4 text-xs text-muted">
          <h3 className="mb-1 font-mono text-paper">Import to Medium</h3>
          <p>Host the <code>.html</code> at a public URL, then use Medium&apos;s &quot;Import a story&quot; at <code>medium.com/p/import</code>.</p>
          <button
            onClick={() => navigator.clipboard?.writeText("https://medium.com/p/import")}
            className="mt-2 rounded border border-line px-2 py-1 hover:border-teal"
          >
            copy import URL
          </button>
        </div>
      </aside>
    </div>
  );
}
