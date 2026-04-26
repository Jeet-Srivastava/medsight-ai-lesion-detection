from __future__ import annotations

APP_TITLE = "MedSight AI"
APP_SUBTITLE = "Real-Time AI for Lesion Detection"
APP_DESCRIPTION = (
    "Clinical-style computer vision workspace for lesion candidate detection, "
    "spatiotemporal tracking, analytics, and deployment benchmarking."
)

DEFAULT_MODEL_PATH = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.35
DEFAULT_TEMPORAL_WINDOW = 3

PAGE_HOME = "home"
PAGE_DASHBOARD = "dashboard"

FRAGMENT_INTERVAL = "120ms"
LOG_LIMIT = 160
MAX_SNAPSHOTS = 8
SNAPSHOT_INTERVAL_FRAMES = 12

UPLOAD_DIR = "uploads"
IMAGE_UPLOAD_DIR = "uploads/images"
VIDEO_UPLOAD_DIR = "uploads/videos"
MODEL_EXPORT_DIR = "models"

SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "webp"]
SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
