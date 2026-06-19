/**
 * API client for the MedSight Python backend.
 *
 * All endpoints are prefixed with /api and proxied to the FastAPI
 * server running at http://localhost:8000 (configured in vite.config.ts).
 */

const BASE = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? "/api" : "https://medsight-backend-0ue8.onrender.com/api");
console.log(BASE);

async function extractErrorMessage(
  res: Response,
  fallback: string
): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") {
      return body.detail;
    }
    return fallback;
  } catch {
    return fallback;
  }
}

/* ── System ──────────────────────────────────────────── */

export async function fetchSystemStatus() {
  const res = await fetch(`${BASE}/status`);
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to fetch system status");
    throw new Error(msg);
  }
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
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Image inference failed");
    throw new Error(msg);
  }
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
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Video upload failed");
    throw new Error(msg);
  }
  return res.json();
}

/* ── Stream Control ──────────────────────────────────── */

export async function startStream(confidence: number) {
  const res = await fetch(`${BASE}/stream/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confidence }),
  });
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to start stream");
    throw new Error(msg);
  }
  return res.json();
}

export async function stopStream() {
  const res = await fetch(`${BASE}/stream/stop`, { method: "POST" });
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to stop stream");
    throw new Error(msg);
  }
  return res.json();
}

export async function fetchStreamFrame() {
  const res = await fetch(`${BASE}/stream/frame`);
  if (res.status === 410) throw new Error("Stream ended");
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to fetch stream frame");
    throw new Error(msg);
  }
  return res.json();
}

export async function sendClientFrame(blob: Blob, confidence: number) {
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");

  const res = await fetch(`${BASE}/stream/client-frame?confidence=${confidence}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to send client frame");
    throw new Error(msg);
  }
  return res.json();
}

/* ── Confidence Update ───────────────────────────────── */

export async function updateConfidence(confidence: number) {
  const res = await fetch(`${BASE}/config/confidence`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confidence }),
  });
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to update confidence");
    throw new Error(msg);
  }
  return res.json();
}

/* ── Clinical Report ─────────────────────────────────── */

export async function fetchReport() {
  const res = await fetch(`${BASE}/report`);
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to fetch report");
    throw new Error(msg);
  }
  return res.json();
}

/* ── XAI Saliency Map ────────────────────────────────── */

export async function fetchSaliencyMap(detectionIndex: number = 0) {
  const res = await fetch(`${BASE}/xai/saliency?detection_index=${detectionIndex}`);
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to generate saliency map");
    throw new Error(msg);
  }
  return res.json();
}

/* ── Audit Trail ─────────────────────────────────────── */

export async function fetchAuditTrail(limit: number = 50) {
  const res = await fetch(`${BASE}/audit?limit=${limit}`);
  if (!res.ok) {
    const msg = await extractErrorMessage(res, "Failed to fetch audit trail");
    throw new Error(msg);
  }
  return res.json();
}
