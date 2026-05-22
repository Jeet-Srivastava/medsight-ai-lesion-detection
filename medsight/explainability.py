"""Explainability module — saliency heatmaps for lesion detections.

Generates a visual heatmap showing which parts of a detected lesion
most influenced the model's decision.  Uses a simple occlusion-based
approach: we systematically block out small patches inside the bounding
box and measure how much the detection confidence drops.

Regions where occlusion causes the biggest confidence drop are the
most "important" to the model → they glow brightest on the heatmap.
"""

from __future__ import annotations

import cv2
import numpy as np

from medsight.detection import LesionDetector


def generate_saliency_map(
    detector: LesionDetector,
    frame_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    confidence: float,
    grid_size: int = 8,
) -> np.ndarray:
    """Create a saliency heatmap for a single detection.

    Args:
        detector:   The loaded YOLO detector.
        frame_rgb:  The original RGB image.
        bbox:       The bounding box (x1, y1, x2, y2) of the detection.
        confidence: The original detection confidence (baseline).
        grid_size:  How many patches to divide the bbox into per axis.
                    Higher = more detailed but slower.  Default 8×8.

    Returns:
        A heatmap array (same height/width as frame_rgb) with values
        0–255 showing per-pixel saliency.  Hot spots = important regions.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame_rgb.shape[:2]

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w < 8 or box_h < 8:
        # Too small for meaningful occlusion analysis
        return np.zeros((h, w), dtype=np.uint8)

    # Divide the bounding box into a grid of patches
    patch_w = max(1, box_w // grid_size)
    patch_h = max(1, box_h // grid_size)

    # The mean color of the bbox region — used to fill occluded patches
    roi = frame_rgb[y1:y2, x1:x2]
    fill_color = roi.mean(axis=(0, 1)).astype(np.uint8)

    # For each patch, occlude it and measure the confidence drop
    importance = np.zeros((grid_size, grid_size), dtype=np.float32)

    for row in range(grid_size):
        for col in range(grid_size):
            # Patch coordinates
            px1 = x1 + col * patch_w
            py1 = y1 + row * patch_h
            px2 = min(x2, px1 + patch_w)
            py2 = min(y2, py1 + patch_h)

            # Create a copy with this patch occluded
            occluded = frame_rgb.copy()
            occluded[py1:py2, px1:px2] = fill_color

            # Run a quick forward pass
            new_conf = _get_max_confidence_in_box(
                detector, occluded, (x1, y1, x2, y2),
            )

            # The confidence drop = how important this patch was
            drop = max(0.0, confidence - new_conf)
            importance[row, col] = drop

    # Normalize importance to 0–255
    max_val = importance.max()
    if max_val > 0:
        importance = (importance / max_val * 255).astype(np.uint8)
    else:
        importance = importance.astype(np.uint8)

    # Upscale the grid to the bounding box size, then place into full frame
    heatmap_roi = cv2.resize(
        importance, (box_w, box_h), interpolation=cv2.INTER_CUBIC,
    )

    heatmap = np.zeros((h, w), dtype=np.uint8)
    heatmap[y1:y2, x1:x2] = heatmap_roi

    # Apply Gaussian blur for smoother visualization
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    return heatmap


def overlay_heatmap(
    frame_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a saliency heatmap onto the original frame.

    Args:
        frame_rgb: The original image (RGB).
        heatmap:   Grayscale saliency map (same size as frame_rgb).
        alpha:     Blending factor (0 = original only, 1 = heatmap only).

    Returns:
        The blended RGB image with a colorful heatmap overlay.
    """
    # Apply a colormap (jet goes from blue=cold to red=hot)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Only blend where the heatmap is non-zero
    mask = heatmap > 10
    result = frame_rgb.copy()
    result[mask] = cv2.addWeighted(
        frame_rgb[mask], 1.0 - alpha,
        colored_rgb[mask], alpha,
        0,
    )

    return result


def _get_max_confidence_in_box(
    detector: LesionDetector,
    frame_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """Run inference and return the highest confidence detection
    that overlaps with the given bounding box.
    """
    results = detector.model.predict(
        source=frame_rgb,
        conf=0.05,  # Low threshold to catch weakened detections
        imgsz=640,
        verbose=False,
        device=detector.device,
    )
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return 0.0

    x1, y1, x2, y2 = bbox
    best_conf = 0.0

    for box in result.boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
        # Check if this detection overlaps with our target box
        overlap_x = max(0, min(x2, bx2) - max(x1, bx1))
        overlap_y = max(0, min(y2, by2) - max(y1, by1))
        if overlap_x > 0 and overlap_y > 0:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf

    return best_conf
