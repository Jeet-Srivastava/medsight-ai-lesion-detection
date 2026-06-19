"""Lesion segmentation using MobileSAM (YOLOSAMic).

Given a bounding box from YOLO, MobileSAM generates a pixel-perfect
mask that separates the lesion (foreground) from surrounding skin.
This replaces the previous GrabCut approach with a state-of-the-art
foundation model for dramatically improved boundary accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LesionMask:
    """The result of segmenting a single lesion."""

    mask: np.ndarray          # binary mask (0/255), same size as the input frame
    contour: np.ndarray       # biggest contour around the lesion
    area_pixels: int          # number of foreground pixels
    centroid: tuple[int, int] # (cx, cy) center of mass


# ── Lazy-loaded MobileSAM model ─────────────────────────

_sam_model = None


def _get_sam_model():
    """Load MobileSAM once and cache it globally.

    The `mobile_sam.pt` weights are auto-downloaded by ultralytics
    on first use (~10 MB).  Subsequent calls return the cached model.
    """
    global _sam_model
    if _sam_model is None:
        from ultralytics import SAM
        _sam_model = SAM("mobile_sam.pt")
    return _sam_model


def segment_lesion(
    frame_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    iterations: int | None = None,       # kept for API compatibility (ignored)
) -> LesionMask | None:
    """Extract a pixel-level lesion mask from a YOLO bounding box.

    Uses MobileSAM with the bbox as a prompt to generate
    a high-resolution segmentation mask, then finds the largest
    contour inside that region.

    Returns None if segmentation fails (e.g. bbox too small or no
    foreground found).
    """
    from medsight.config import SAM_MIN_AREA_PX

    x1, y1, x2, y2 = bbox
    height, width = frame_rgb.shape[:2]

    # Clamp bbox to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    box_w = x2 - x1
    box_h = y2 - y1

    # Skip if the box is too tiny to be meaningful
    if box_w < 10 or box_h < 10:
        return None

    # ── Run MobileSAM with the YOLO bbox as prompt ───────
    try:
        sam = _get_sam_model()
        results = sam(frame_rgb, bboxes=[[x1, y1, x2, y2]], verbose=False)
    except Exception:
        return None

    # Extract the mask from SAM results
    if not results or results[0].masks is None or len(results[0].masks) == 0:
        return None

    # SAM returns masks as tensors — convert to numpy binary mask
    mask_tensor = results[0].masks.data[0]  # first mask for first bbox
    sam_mask = mask_tensor.cpu().numpy()

    # Resize to original frame dimensions if needed
    if sam_mask.shape[:2] != (height, width):
        sam_mask = cv2.resize(
            sam_mask.astype(np.float32), (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    # Binarize: threshold at 0.5 → 0/255
    binary_mask = (sam_mask > 0.5).astype(np.uint8) * 255

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Find contours and pick the largest one
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(biggest))

    if area < SAM_MIN_AREA_PX:
        return None

    # Compute the center of mass
    moments = cv2.moments(biggest)
    if moments["m00"] == 0:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    else:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

    # Rebuild a clean mask from just the largest contour
    clean_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(clean_mask, [biggest], -1, 255, thickness=cv2.FILLED)

    return LesionMask(
        mask=clean_mask,
        contour=biggest,
        area_pixels=area,
        centroid=(cx, cy),
    )
