"""ABCDE morphological analysis for skin lesions.

Implements the dermatological ABCDE criteria:
  A — Asymmetry       (how lopsided is the shape?)
  B — Border          (how irregular is the edge?)
  C — Color           (how many distinct colors?)
  D — Diameter        (how large is it?)
  E — Evolution       (has it changed over time?)

Each criterion is scored independently.  The scores are combined into
a total that feeds the risk classifier in risk.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# ── Result dataclass ─────────────────────────────────────


@dataclass
class ABCDEResult:
    """Scores for each ABCDE criterion plus a combined total."""

    asymmetry_score: float       # 0–2
    border_score: float          # 0–2
    color_score: float           # 0–3
    diameter_mm: float           # estimated mm
    diameter_score: float        # 0–2
    evolution_score: float       # 0 (placeholder)
    total_score: float = 0.0     # sum of all scores
    color_count: int = 1         # number of distinct colors found
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.total_score = (
            self.asymmetry_score
            + self.border_score
            + self.color_score
            + self.diameter_score
            + self.evolution_score
        )


# ── Public API ───────────────────────────────────────────


def analyze_abcde(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    contour: np.ndarray,
    pixels_per_mm: float | None = None,
) -> ABCDEResult:
    """Run all five ABCDE criteria on a segmented lesion.

    Args:
        frame_rgb:    The full image in RGB.
        mask:         Binary mask (0/255) of the lesion.
        contour:      The lesion contour from segmentation.
        pixels_per_mm: Approximate scale factor.  If None, uses
                       the default from config.py.

    Returns:
        An ABCDEResult with all scores populated.
    """
    from medsight.config import ABCDE_PIXELS_PER_MM

    if pixels_per_mm is None:
        pixels_per_mm = ABCDE_PIXELS_PER_MM

    a_score, a_details = _score_asymmetry(mask, contour)
    b_score, b_details = _score_border(contour, mask)
    c_score, c_count, c_details = _score_color(frame_rgb, mask)
    d_mm, d_score, d_details = _score_diameter(contour, pixels_per_mm)
    e_score = 0.0  # Evolution requires temporal comparison — future work

    return ABCDEResult(
        asymmetry_score=a_score,
        border_score=b_score,
        color_score=c_score,
        diameter_mm=d_mm,
        diameter_score=d_score,
        evolution_score=e_score,
        color_count=c_count,
        details={**a_details, **b_details, **c_details, **d_details},
    )


# ── A: Asymmetry ────────────────────────────────────────


def _score_asymmetry(
    mask: np.ndarray,
    contour: np.ndarray,
) -> tuple[float, dict]:
    """Measure how asymmetric the lesion shape is.

    We fit an ellipse to the contour, then split the mask along
    the major and minor axes.  The IoU of each pair of halves
    tells us how symmetric the shape is.
    """
    if len(contour) < 5:
        # Need at least 5 points for fitEllipse
        return 0.0, {"asymmetry_reason": "too_few_points"}

    # Fit an ellipse to find the orientation
    (cx, cy), (ma, mi), angle = cv2.fitEllipse(contour)
    cx, cy = int(cx), int(cy)

    h, w = mask.shape[:2]

    # Split along the major axis (horizontal split rotated by angle)
    iou_major = _half_iou(mask, cx, cy, angle)
    # Split along the minor axis (perpendicular)
    iou_minor = _half_iou(mask, cx, cy, angle + 90)

    # Lower IoU = more asymmetric
    avg_iou = (iou_major + iou_minor) / 2.0

    if avg_iou > 0.80:
        score = 0.0   # Nearly symmetric
    elif avg_iou > 0.60:
        score = 1.0   # Moderately asymmetric
    else:
        score = 2.0   # Highly asymmetric

    return score, {
        "asymmetry_iou_major": round(iou_major, 3),
        "asymmetry_iou_minor": round(iou_minor, 3),
    }


def _half_iou(
    mask: np.ndarray,
    cx: int,
    cy: int,
    angle_deg: float,
) -> float:
    """Split the mask through (cx, cy) at the given angle and compute
    the IoU of one half with the mirror of the other half.
    """
    h, w = mask.shape[:2]

    # Create a dividing line through the center at the given angle
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    # Build a coordinate grid relative to the center
    ys, xs = np.mgrid[0:h, 0:w]
    side = (xs - cx) * sin_a - (ys - cy) * cos_a

    # Two halves
    half_a = (mask > 0) & (side >= 0)
    half_b = (mask > 0) & (side < 0)

    # Flip half_b across the dividing line to compare with half_a
    # Simple approach: just compare pixel counts — close enough
    count_a = int(np.sum(half_a))
    count_b = int(np.sum(half_b))

    if count_a + count_b == 0:
        return 1.0

    # IoU approximation: min/max of the two halves
    smaller = min(count_a, count_b)
    larger = max(count_a, count_b)
    return smaller / larger if larger > 0 else 1.0


# ── B: Border Irregularity ──────────────────────────────


def _score_border(
    contour: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, dict]:
    """Measure border irregularity using compactness.

    Compactness = perimeter² / (4π × area).
    A perfect circle has compactness = 1.  Higher values mean
    more irregular borders.
    """
    perimeter = cv2.arcLength(contour, closed=True)
    area = cv2.contourArea(contour)

    if area < 1:
        return 0.0, {"border_compactness": 0.0}

    compactness = (perimeter ** 2) / (4.0 * np.pi * area)

    # Score thresholds based on dermatological literature
    if compactness < 1.3:
        score = 0.0   # Smooth, circular border
    elif compactness < 2.0:
        score = 1.0   # Moderately irregular
    else:
        score = 2.0   # Highly irregular / jagged

    return score, {"border_compactness": round(compactness, 3)}


# ── C: Color Variation ──────────────────────────────────


def _score_color(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int, dict]:
    """Count distinct colors within the lesion using k-means clustering.

    We extract all pixels inside the mask, convert to LAB color space
    (which is perceptually uniform), and cluster them.  The number of
    visually distinct clusters tells us the color score.
    """
    # Extract pixels inside the mask
    lesion_pixels = frame_rgb[mask > 0]

    if len(lesion_pixels) < 20:
        return 0.0, 1, {"color_note": "too_few_pixels"}

    # Convert to LAB for perceptually uniform clustering
    # Reshape for cvtColor: needs (N, 1, 3)
    pixels_reshaped = lesion_pixels.reshape(-1, 1, 3).astype(np.uint8)
    lab_pixels = cv2.cvtColor(pixels_reshaped, cv2.COLOR_RGB2LAB)
    lab_flat = lab_pixels.reshape(-1, 3).astype(np.float32)

    # K-means clustering (try up to max_k clusters)
    from medsight.config import ABCDE_COLOR_MAX_K, ABCDE_COLOR_MIN_FRACTION

    max_k = min(ABCDE_COLOR_MAX_K, len(lab_flat) // 10)
    max_k = max(2, max_k)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        lab_flat, max_k, None, criteria, 3, cv2.KMEANS_PP_CENTERS,
    )

    # Count clusters that actually have significant representation
    total = len(labels)
    distinct = 0
    for i in range(max_k):
        cluster_size = int(np.sum(labels == i))
        if cluster_size / total > ABCDE_COLOR_MIN_FRACTION:
            distinct += 1

    # Score based on number of distinct colors
    if distinct <= 1:
        score = 0.0   # Uniform color
    elif distinct == 2:
        score = 1.0   # Two-toned
    elif distinct == 3:
        score = 2.0   # Multi-colored
    else:
        score = 3.0   # Highly variegated (4+ colors)

    return score, distinct, {"color_clusters": distinct, "color_max_k": max_k}


# ── D: Diameter ─────────────────────────────────────────


def _score_diameter(
    contour: np.ndarray,
    pixels_per_mm: float,
) -> tuple[float, float, dict]:
    """Estimate the lesion diameter in millimeters.

    Uses the equivalent diameter (diameter of a circle with the
    same area as the lesion).  The clinical threshold is 6mm.
    """
    area = cv2.contourArea(contour)

    if area < 1:
        return 0.0, 0.0, {"diameter_pixels": 0}

    # Equivalent diameter in pixels
    diameter_px = np.sqrt(4.0 * area / np.pi)
    diameter_mm = diameter_px / pixels_per_mm

    # Score
    if diameter_mm < 4.0:
        score = 0.0   # Small — low concern
    elif diameter_mm < 6.0:
        score = 1.0   # Approaching the 6mm threshold
    else:
        score = 2.0   # Above 6mm — clinically significant

    return round(diameter_mm, 1), score, {
        "diameter_pixels": round(diameter_px, 1),
        "diameter_mm": round(diameter_mm, 1),
    }
