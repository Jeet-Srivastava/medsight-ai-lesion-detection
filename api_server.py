"""
FastAPI backend for the MedSight React dashboard.

Exposes REST endpoints that the frontend consumes via /api proxy.
Bridges the existing medsight pipeline (detection, tracking, analytics)
to JSON responses the React UI expects.

Run: uvicorn api_server:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from medsight.config import DEFAULT_MODEL_PATH, DEFAULT_CONFIDENCE, DEFAULT_TEMPORAL_WINDOW
from medsight.detection import LesionDetector
from medsight.pipeline import MedSightPipeline

# ── App ──────────────────────────────────────────────────

app = FastAPI(title="MedSight API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ─────────────────────────────────────────

detector: LesionDetector | None = None
pipeline: MedSightPipeline | None = None
current_confidence: float = DEFAULT_CONFIDENCE
stream_active: bool = False

# Video stream state
video_cap: cv2.VideoCapture | None = None
video_frame_index: int = 0
video_total_frames: int = 0


def get_detector() -> LesionDetector:
    global detector
    if detector is None:
        detector = LesionDetector(DEFAULT_MODEL_PATH)
    return detector


def get_pipeline() -> MedSightPipeline:
    global pipeline
    if pipeline is None:
        det = get_detector()
        pipeline = MedSightPipeline(
            detector=det,
            temporal_window=DEFAULT_TEMPORAL_WINDOW,
            tracker_name="bytetrack.yaml",
            enable_fp16=det.fp16_supported,
        )
    return pipeline


# ── Schemas ──────────────────────────────────────────────

class ConfidenceUpdate(BaseModel):
    confidence: float


class StreamStartRequest(BaseModel):
    confidence: float = DEFAULT_CONFIDENCE


# ── Helpers ──────────────────────────────────────────────

def encode_frame(frame_rgb: np.ndarray) -> str:
    """Encode an RGB numpy frame to base64 JPEG."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def detection_to_dict(det) -> dict:
    return {
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": det.confidence,
        "bbox": list(det.bbox),
        "track_id": det.track_id,
        "confirmed": det.confirmed,
        "duration_frames": det.duration_frames,
        "duration_seconds": det.duration_seconds,
    }


def analytics_to_dict(a) -> dict:
    return {
        "frame_index": a.frame_index,
        "total_frames": a.total_frames,
        "raw_detections": a.raw_detections,
        "confirmed_detections": a.confirmed_detections,
        "total_confirmed_lesions": a.total_confirmed_lesions,
        "active_lesions": a.active_lesions,
        "average_confidence": a.average_confidence,
        "detection_frequency": a.detection_frequency,
        "inference_ms": a.inference_ms,
        "pipeline_ms": a.pipeline_ms,
        "fps": a.fps,
    }


def pipeline_result_to_dict(result) -> dict:
    height, width = result.raw_frame.shape[:2]
    return {
        "frame_index": result.frame_index,
        "total_frames": result.total_frames,
        "frame_width": width,
        "frame_height": height,
        "raw_detections": [detection_to_dict(d) for d in result.raw_detections],
        "confirmed_detections": [detection_to_dict(d) for d in result.confirmed_detections],
        "analytics": analytics_to_dict(result.analytics),
        "annotated_frame_b64": encode_frame(result.rendered_frame),
        "logs": result.logs,
    }


# ── Routes ───────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    det = get_detector()
    return {
        "model_name": "YOLO11",
        "model_path": det.model_path,
        "device": det.device.upper(),
        "status": "active" if stream_active else "idle",
        "fp16_enabled": det.fp16_supported,
    }


@app.post("/api/inference/image")
async def infer_image(file: UploadFile = File(...), confidence: float = DEFAULT_CONFIDENCE):
    """Run inference on a single uploaded image."""
    global current_confidence
    current_confidence = confidence

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pipe = get_pipeline()
    pipe.reset()
    result = pipe.process_image(frame_rgb, confidence=confidence)
    return pipeline_result_to_dict(result)


@app.post("/api/inference/video")
async def upload_video(file: UploadFile = File(...), confidence: float = DEFAULT_CONFIDENCE):
    """Upload a video file for frame-by-frame processing."""
    global video_cap, video_frame_index, video_total_frames, stream_active, current_confidence

    current_confidence = confidence

    # Save uploaded video to temp file
    suffix = Path(file.filename or "video.mp4").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()

    # Open video capture
    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        os.unlink(tmp.name)
        raise HTTPException(status_code=400, detail="Cannot open video file")

    video_cap = cap
    video_frame_index = 0
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stream_active = True

    pipe = get_pipeline()
    pipe.reset()

    return {"status": "ok", "total_frames": video_total_frames}


@app.post("/api/stream/start")
def start_stream(req: StreamStartRequest):
    """Start webcam stream (if available)."""
    global video_cap, video_frame_index, video_total_frames, stream_active, current_confidence

    current_confidence = req.confidence
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=503, detail="No webcam available")

    video_cap = cap
    video_frame_index = 0
    video_total_frames = 0  # unknown for live stream
    stream_active = True

    pipe = get_pipeline()
    pipe.reset()

    return {"status": "streaming"}


@app.post("/api/stream/stop")
def stop_stream():
    """Stop the active video/webcam stream."""
    global video_cap, stream_active

    if video_cap is not None:
        video_cap.release()
        video_cap = None
    stream_active = False
    return {"status": "stopped"}


@app.get("/api/stream/frame")
def get_stream_frame():
    """Fetch the next processed frame from the active stream."""
    global video_cap, video_frame_index, video_total_frames, stream_active

    if video_cap is None or not video_cap.isOpened():
        return {"status": "stopped"}

    ret, frame_bgr = video_cap.read()
    if not ret:
        # End of video
        video_cap.release()
        video_cap = None
        stream_active = False
        raise HTTPException(status_code=410, detail="Stream ended")

    video_frame_index += 1
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    pipe = get_pipeline()
    result = pipe.process_video_frame(
        frame_rgb,
        frame_index=video_frame_index,
        total_frames=video_total_frames,
        confidence=current_confidence,
    )
    return pipeline_result_to_dict(result)


@app.patch("/api/config/confidence")
def update_confidence(body: ConfidenceUpdate):
    """Update the confidence threshold for future inferences."""
    global current_confidence
    current_confidence = body.confidence
    return {"confidence": current_confidence}
