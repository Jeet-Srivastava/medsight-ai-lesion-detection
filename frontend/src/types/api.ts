/* ─────────────────────────────────────────────────────────
 * TypeScript types mirroring the Python backend dataclasses
 * (medsight/detection.py, medsight/analytics.py, medsight/pipeline.py)
 * ─────────────────────────────────────────────────────── */

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  track_id: number | null;
  confirmed: boolean;
  duration_frames: number;
  duration_seconds: number;
}

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

export interface SystemStatus {
  model_name: string;
  model_path: string;
  device: string;
  status: "active" | "idle" | "error";
  fp16_enabled: boolean;
}

export type StreamState = "idle" | "streaming" | "paused" | "processing";
