/* ─────────────────────────────────────────────────────────
 * TypeScript types mirroring the Python backend dataclasses
 * (medsight/detection.py, medsight/analytics.py, medsight/pipeline.py,
 *  medsight/abcde.py, medsight/risk.py, medsight/reporting.py)
 * ─────────────────────────────────────────────────────── */

/* ── ABCDE Morphological Analysis ────────────────────── */

export interface ABCDEResult {
  asymmetry_score: number;   // 0–2
  border_score: number;      // 0–2
  color_score: number;       // 0–3
  color_count: number;       // distinct colors found
  diameter_mm: number;       // estimated mm
  diameter_score: number;    // 0–2
  evolution_score: number;   // 0 (placeholder)
  total_score: number;       // sum of all scores
}

export interface RiskAssessment {
  level: "Low" | "Moderate" | "High" | "Refer";
  total_score: number;
  summary: string;
}

/* ── Detection ───────────────────────────────────────── */

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  track_id: number | null;
  confirmed: boolean;
  duration_frames: number;
  duration_seconds: number;
  abcde?: ABCDEResult;
  risk?: RiskAssessment;
}

/* ── Analytics ───────────────────────────────────────── */

export interface FrameAnalytics {
  frame_index: number;
  total_frames: number;
  raw_detections: number;
  confirmed_detections: number;
  total_confirmed_lesions: number;
  active_lesions: number;
  average_confidence: number;
  detection_frequency: number;
  inference_ms: number;
  pipeline_ms: number;
  fps: number;
}

/* ── Pipeline Result ─────────────────────────────────── */

export interface PipelineResult {
  frame_index: number;
  total_frames: number;
  frame_width?: number;
  frame_height?: number;
  raw_detections: Detection[];
  confirmed_detections: Detection[];
  analytics: FrameAnalytics;
  annotated_frame_b64: string; // base64-encoded rendered frame
  logs: Array<[string, string]>;
}

/* ── System Status ───────────────────────────────────── */

export interface SystemStatus {
  model_name: string;
  model_path: string;
  device: string;
  status: "active" | "idle" | "error";
  fp16_enabled: boolean;
}

export type StreamState = "idle" | "streaming" | "paused" | "processing";

/* ── Clinical Report ─────────────────────────────────── */

export interface ReportFinding {
  finding_number: number;
  class_name: string;
  confidence: number;
  bounding_box: number[];
  track_id: number | null;
  confirmed: boolean;
  abcde?: {
    asymmetry: number;
    border: number;
    color: number;
    color_count: number;
    diameter_mm: number;
    diameter_score: number;
    evolution: number;
    total_score: number;
  };
  risk?: {
    level: string;
    total_score: number;
    summary: string;
  };
}

export interface ClinicalReport {
  report_id: string;
  timestamp: string;
  session_id: string;
  patient_metadata: Record<string, unknown>;
  image_hash: string;
  image_dimensions: [number, number];
  model_info: Record<string, string>;
  parameters: Record<string, unknown>;
  total_detections: number;
  confirmed_detections: number;
  findings: ReportFinding[];
  summary: string;
}

/* ── Saliency Result ─────────────────────────────────── */

export interface SaliencyResult {
  detection_index: number;
  saliency_frame_b64: string;
  confidence: number;
  bbox: number[];
}

/* ── Audit Entry ─────────────────────────────────────── */

export interface AuditEntry {
  timestamp: string;
  session_id: string;
  input_hash: string;
  model_path: string;
  confidence_threshold: number;
  detections_count: number;
  confirmed_count: number;
  high_risk_count: number;
}

export interface AuditTrail {
  total_entries: number;
  entries: AuditEntry[];
}
