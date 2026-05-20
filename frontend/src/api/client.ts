/**
 * API client for the MedSight Python backend.
 *
 * All endpoints are prefixed with /api and proxied to the FastAPI
 * server running at http://localhost:8000 (configured in vite.config.ts).
 *
 * Wire these functions to your actual backend routes.
 */

const BASE = "/api";

/* ── System ──────────────────────────────────────────── */

export async function fetchSystemStatus() {
  const res = await fetch(`${BASE}/status`);
  if (!res.ok) throw new Error("Failed to fetch system status");
  return res.json();
}

/* ── Image Upload & Inference ────────────────────────── */

export async function uploadImageForInference(
  file: File,
  confidence: number
) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE}/inference/image?confidence=${confidence}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error("Image inference failed");
  return res.json();
}

/* ── Video Upload ────────────────────────────────────── */

export async function uploadVideoForInference(
  file: File,
  confidence: number
) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE}/inference/video?confidence=${confidence}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error("Video upload failed");
  return res.json();
}

/* ── Stream Control ──────────────────────────────────── */

export async function startStream(confidence: number) {
  const res = await fetch(`${BASE}/stream/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confidence }),
  });
  if (!res.ok) throw new Error("Failed to start stream");
  return res.json();
}

export async function stopStream() {
  const res = await fetch(`${BASE}/stream/stop`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to stop stream");
  return res.json();
}

export async function fetchStreamFrame() {
  const res = await fetch(`${BASE}/stream/frame`);
  if (res.status === 410) throw new Error("Stream ended");
  if (!res.ok) throw new Error("Failed to fetch stream frame");
  return res.json();
}

/* ── Confidence Update ───────────────────────────────── */

export async function updateConfidence(confidence: number) {
  const res = await fetch(`${BASE}/config/confidence`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confidence }),
  });
  if (!res.ok) throw new Error("Failed to update confidence");
  return res.json();
}
