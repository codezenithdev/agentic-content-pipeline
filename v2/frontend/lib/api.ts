export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function jpost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

async function jget<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export interface StartRunBody {
  topic: string;
  target_keyword: string;
  style_persona: string;
  voice_input_path?: string | null;
}

export const api = {
  startRun: (b: StartRunBody) => jpost<{ run_id: string }>("/api/run", b),
  approve: (id: string) => jpost<{ status: string }>(`/api/run/${id}/approve`, {}),
  reject: (id: string, note: string) => jpost<{ status: string }>(`/api/run/${id}/reject`, { note }),
  output: (id: string) => jget<Record<string, unknown>>(`/api/run/${id}/output`),
  startBatch: (b: { topics: string[]; target_keyword: string; style_persona: string }) =>
    jpost<{ batch_id: string }>("/api/batch", b),
  memorySearch: (topic: string) =>
    jget<{ results: Array<Record<string, unknown>> }>(`/api/memory/search?topic=${encodeURIComponent(topic)}`),
  transcribe: async (file: File): Promise<{ transcript: string }> => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${API_BASE}/api/voice/transcribe`, { method: "POST", body: fd });
    return res.json();
  },
  runStreamUrl: (id: string) => `${API_BASE}/api/run/${id}/stream`,
  batchStreamUrl: (id: string) => `${API_BASE}/api/batch/${id}/stream`,
  abs: (rel: string) => (rel.startsWith("http") ? rel : `${API_BASE}${rel}`),
};
