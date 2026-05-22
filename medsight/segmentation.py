"""Lesion segmentation using OpenCV GrabCut.

Given a bounding box from YOLO, GrabCut refines it into a pixel-level
mask that separates the lesion (foreground) from surrounding skin
(background).  No extra ML model needed — just OpenCV.
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


def segment_lesion(
    frame_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    iterations: int | None = None,
) -> LesionMask | None:
    """Extract a pixel-level lesion mask from a YOLO bounding box.

    Uses GrabCut with the bbox as the initial rectangle, then finds
    the largest contour inside that region.

    Returns None if segmentation fails (e.g. bbox too small or no
    foreground found).
    """
    from medsight.config import GRABCUT_ITERATIONS, GRABCUT_MARGIN_PX, GRABCUT_MIN_AREA_PX

    if iterations is None:
        iterations = GRABCUT_ITERATIONS
    x1, y1, x2, y2 = bbox
    height, width = frame_rgb.shape[:2]

    # Clamp bbox to image bounds and add a small margin
    margin = GRABCUT_MARGIN_PX
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width, x2 + margin)
    y2 = min(height, y2 + margin)

    box_w = x2 - x1
    box_h = y2 - y1

    # Skip if the box is too tiny to be meaningful
    if box_w < 10 or box_h < 10:
        return None

    # GrabCut works on BGR images
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Prepare the mask and model arrays GrabCut needs
    gc_mask = np.zeros((height, width), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    rect = (x1, y1, box_w, box_h)

    try:
        cv2.grabCut(
            frame_bgr, gc_mask, rect,
            bg_model, fg_model,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        # GrabCut can fail on very uniform images — that's ok
        return None

    # GrabCut labels: 0=bg, 1=fg, 2=probable bg, 3=probable fg
    # We treat both definite and probable foreground as our lesion
    binary_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255, 0,
    ).astype(np.uint8)

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

    if area < GRABCUT_MIN_AREA_PX:
        # Too small to be a real lesion
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
