/**
 * API client for the MedSight Python backend.
 *
 * All endpoints are prefixed with /api and proxied to the FastAPI
 * server running at http://localhost:8000 (configured in vite.config.ts).
 */

const BASE = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? "/api" : "https://medsight-backend-0ue8.onrender.com/api");

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

export async function sendClientFrame(blob: Blob, confidence: number) {
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");

  const res = await fetch(`${BASE}/stream/client-frame?confidence=${confidence}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error("Failed to send client frame");
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

/* ── Clinical Report ─────────────────────────────────── */

export async function fetchReport() {
  const res = await fetch(`${BASE}/report`);
  if (!res.ok) throw new Error("Failed to fetch report");
  return res.json();
}

/* ── XAI Saliency Map ────────────────────────────────── */

export async function fetchSaliencyMap(detectionIndex: number = 0) {
  const res = await fetch(`${BASE}/xai/saliency?detection_index=${detectionIndex}`);
  if (!res.ok) throw new Error("Failed to generate saliency map");
  return res.json();
}

/* ── Audit Trail ─────────────────────────────────────── */

export async function fetchAuditTrail(limit: number = 50) {
  const res = await fetch(`${BASE}/audit?limit=${limit}`);
  if (!res.ok) throw new Error("Failed to fetch audit trail");
  return res.json();
}
