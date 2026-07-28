export const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function createJob(file, options) {
  const fd = new FormData();
  fd.append("video", file);
  fd.append("options", JSON.stringify(options));
  const r = await fetch(`${API}/api/jobs`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed (${r.status})`);
  return r.json();
}

export async function getJob(id) {
  const r = await fetch(`${API}/api/jobs/${id}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`job fetch failed (${r.status})`);
  return r.json();
}

export const fileUrl = (id, name) => `${API}/api/jobs/${id}/${name}`;
