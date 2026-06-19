"""MedSight configuration — all tunable parameters in one place.

YOLO object detection does NOT use temperature / top-k / top-p
(those are for text-generation LLMs).  The knobs that actually
control detection precision and accuracy are listed below.
"""

from __future__ import annotations

APP_TITLE = "MedSight AI"
APP_SUBTITLE = "Real-Time AI for Lesion Detection"
APP_DESCRIPTION = (
    "Clinical-style computer vision workspace for lesion candidate detection, "
    "spatiotemporal tracking, analytics, and deployment benchmarking."
)

# ── Model ────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "yolo11n.pt"

# ── Inference Tuning ─────────────────────────────────────
#
#  These are the parameters that directly affect detection
#  precision (fewer false positives) and recall (fewer misses).
#

# Confidence threshold — only keep detections above this.
#   Higher = more precise, fewer false positives.
#   Lower  = higher recall, but noisier.
#   Medical use cases benefit from moderate values to avoid
#   missing real lesions while not flooding with noise.
DEFAULT_CONFIDENCE = 0.40

# Input image size fed to the model (pixels, square).
#   Larger = more spatial detail = better accuracy, but slower.
#   640 → 1024 gives ~15-20% mAP improvement on small objects.
#   Use 640 for real-time video, 1024 for single-image analysis.
DEFAULT_IMGSZ_IMAGE = 1024    # single image analysis — prioritize accuracy
DEFAULT_IMGSZ_VIDEO = 640     # video streaming — balance speed vs. accuracy

# IoU (Intersection over Union) threshold for NMS.
#   Lower = stricter overlap suppression = fewer duplicate boxes.
#   0.45 is standard; 0.35 is more aggressive for medical use
#   where overlapping detections are rarely valid.
DEFAULT_IOU_THRESHOLD = 0.35

# Maximum number of detections per frame.
#   Cap to prevent noise floods on complex images.
DEFAULT_MAX_DETECTIONS = 50

# Test-Time Augmentation (TTA) — runs inference on flipped/
#   scaled copies of the image and merges results.
#   Boosts accuracy ~2-5% at 2-3× speed cost.
#   Enable for single images, disable for video streams.
DEFAULT_AUGMENT_IMAGE = True
DEFAULT_AUGMENT_VIDEO = False

# Temporal confirmation — how many consecutive frames a
#   detection must persist before being marked "confirmed".
DEFAULT_TEMPORAL_WINDOW = 3
# ── Preprocessing (CLAHE) ────────────────────────────────

#   CLAHE enhances local contrast in the image, making subtle
#   lesion boundaries more visible to the model.

CLAHE_CLIP_LIMIT = 2.5        # contrast amplification limit (was 2.0)
CLAHE_TILE_GRID = (8, 8)      # grid size for local histogram equalization

# ── Segmentation (MobileSAM) ─────────────────────────────

SAM_MIN_AREA_PX = 50          # minimum mask area to be considered valid

# ── ABCDE Scoring ────────────────────────────────────────

ABCDE_PIXELS_PER_MM = 10.0    # approximate scale factor for diameter estimate
ABCDE_COLOR_MAX_K = 5         # max clusters for color analysis
ABCDE_COLOR_MIN_FRACTION = 0.08  # minimum cluster size to count as distinct

# Risk thresholds (ABCDE total score out of 9)
RISK_THRESHOLD_LOW = 2.0
RISK_THRESHOLD_MODERATE = 4.0
RISK_THRESHOLD_HIGH = 6.0     # 7+ = "Refer"

# ── XAI (Saliency) ──────────────────────────────────────

SALIENCY_GRID_SIZE = 8        # patches per axis inside the bbox
SALIENCY_MIN_CONF = 0.05      # low threshold for occlusion re-runs

# ── UI / Misc ────────────────────────────────────────────

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
