"use client";

import { useRouter } from "next/navigation";
import { useState, type ChangeEvent } from "react";
import { api } from "@/lib/api";

const PERSONAS = ["technical deep-dive", "beginner-friendly", "op-ed"];

interface MemHit { slug: string; topic: string; similarity: number }

export default function Dashboard() {
  const router = useRouter();
  const [topic, setTopic] = useState("Mediterranean diet health benefits");
  const [keyword, setKeyword] = useState("Mediterranean diet");
  const [persona, setPersona] = useState(PERSONAS[0]);
  const [voicePath, setVoicePath] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [mem, setMem] = useState<MemHit[]>([]);

  async function onVoice(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    try {
      const { transcript } = await api.transcribe(file);
      if (transcript) setTopic(transcript);
      setVoicePath(file.name);
    } finally {
      setBusy(false);
    }
  }

  async function start() {
    setBusy(true);
    try {
      const { run_id } = await api.startRun({ topic, target_keyword: keyword, style_persona: persona, voice_input_path: voicePath });
      router.push(`/run/${run_id}`);
    } catch {
      setBusy(false);
      alert("Failed to start — is the backend running on :8000?");
    }
  }

  async function searchMem() {
    try {
      const r = await api.memorySearch(topic);
      setMem((r.results as unknown as MemHit[]) || []);
    } catch { /* ignore */ }
  }

  const field = "mt-1 w-full rounded border border-line bg-ink p-2 text-sm text-paper outline-none focus:border-teal";

  return (
    <main className="mx-auto max-w-4xl p-6">
      <header className="mb-6">
        <h1 className="font-mono text-2xl text-paper">Agentic Content Pipeline <span className="text-teal">/ ops</span></h1>
        <p className="text-sm text-muted">Research → outline → write → fact-check → SEO → edit loop → human approval → publish.</p>
      </header>

      <section className="rounded-lg border border-line bg-panel p-5">
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="text-xs text-muted">Topic
            <textarea value={topic} onChange={(e) => setTopic(e.target.value)} rows={2} className={field} />
          </label>
          <div className="space-y-3">
            <label className="block text-xs text-muted">Target keyword
              <input value={keyword} onChange={(e) => setKeyword(e.target.value)} className={field} />
            </label>
            <label className="block text-xs text-muted">Style persona
              <select value={persona} onChange={(e) => setPersona(e.target.value)} className={field}>
                {PERSONAS.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </label>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button onClick={start} disabled={busy || !topic || !keyword}
            className="rounded bg-teal px-4 py-2 font-mono text-sm text-ink hover:opacity-90 disabled:opacity-40">▶ Run pipeline</button>
          <label className="cursor-pointer rounded border border-line px-3 py-2 font-mono text-xs text-muted hover:border-teal">
            🎙 voice topic<input type="file" accept="audio/*" onChange={onVoice} className="hidden" />
          </label>
          {voicePath && <span className="font-mono text-xs text-teal">{voicePath}</span>}
          <button onClick={searchMem} className="rounded border border-line px-3 py-2 font-mono text-xs text-muted hover:border-teal">⟲ memory</button>
          <a href="/batch" className="ml-auto font-mono text-xs text-teal">multi-topic batch →</a>
        </div>
      </section>

      {mem.length > 0 && (
        <section className="mt-4 rounded-lg border border-line bg-panel p-4 text-xs">
          <h3 className="mb-2 font-mono text-teal">memory — similar past runs</h3>
          {mem.map((h) => (
            <div key={h.slug} className="flex justify-between border-b border-line/60 py-1">
              <a href={`/output/${h.slug}`} className="text-paper hover:text-teal">{h.topic}</a>
              <span className="font-mono text-muted">sim {Number(h.similarity).toFixed(2)}</span>
            </div>
          ))}
        </section>
      )}
    </main>
  );
}
