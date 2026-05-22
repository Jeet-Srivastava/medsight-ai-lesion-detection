"""Clinical report generator.

Produces a structured JSON report from a pipeline result.  The report
contains everything a clinician would need: patient metadata, detection
details, ABCDE scores, risk assessments, model info, and parameters.

Reports can be serialized to JSON for download or display in the frontend.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from medsight.detection import Detection


@dataclass
class ClinicalReport:
    """A structured clinical analysis report."""

    report_id: str
    timestamp: str
    session_id: str
    patient_metadata: dict[str, Any]
    image_hash: str
    image_dimensions: tuple[int, int]
    model_info: dict[str, str]
    parameters: dict[str, Any]
    total_detections: int
    confirmed_detections: int
    findings: list[dict[str, Any]]
    summary: str


def generate_report(
    detections: list[Detection],
    frame_rgb: np.ndarray,
    session_id: str,
    model_name: str = "YOLO11",
    model_path: str = "yolo11n.pt",
    confidence_threshold: float = 0.35,
    patient_metadata: dict[str, Any] | None = None,
) -> ClinicalReport:
    """Create a clinical report from detection results.

    Args:
        detections:           List of confirmed detections (with ABCDE if available).
        frame_rgb:            The analyzed image (used for dimensions and hashing).
        session_id:           Current session identifier.
        model_name:           Name of the model used.
        model_path:           Path to the model weights.
        confidence_threshold: The confidence threshold that was applied.
        patient_metadata:     Optional dict with patient info (age, sex, location, etc.).

    Returns:
        A ClinicalReport ready for serialization.
    """
    now = datetime.now(timezone.utc)
    h, w = frame_rgb.shape[:2]

    # Hash a small sample of the image for identity tracking
    sample = frame_rgb[::10, ::10].tobytes()
    image_hash = hashlib.sha256(sample).hexdigest()[:16]

    # Build findings from each detection
    findings = []
    for i, det in enumerate(detections):
        finding = _build_finding(det, index=i + 1)
        findings.append(finding)

    # Overall summary
    high_risk_count = sum(
        1 for det in detections
        if det.risk is not None and det.risk.level in ("High", "Refer")
    )
    summary = _build_summary(len(detections), high_risk_count)

    return ClinicalReport(
        report_id=str(uuid.uuid4())[:12],
        timestamp=now.isoformat(),
        session_id=session_id,
        patient_metadata=patient_metadata or {},
        image_hash=image_hash,
        image_dimensions=(w, h),
        model_info={
            "name": model_name,
            "weights": model_path,
            "type": "object_detection",
        },
        parameters={
            "confidence_threshold": confidence_threshold,
            "analysis_type": "ABCDE morphological",
        },
        total_detections=len(detections),
        confirmed_detections=sum(1 for d in detections if d.confirmed),
        findings=findings,
        summary=summary,
    )


def report_to_dict(report: ClinicalReport) -> dict[str, Any]:
    """Convert a ClinicalReport to a plain dict for JSON serialization."""
    return {
        "report_id": report.report_id,
        "timestamp": report.timestamp,
        "session_id": report.session_id,
        "patient_metadata": report.patient_metadata,
        "image_hash": report.image_hash,
        "image_dimensions": list(report.image_dimensions),
        "model_info": report.model_info,
        "parameters": report.parameters,
        "total_detections": report.total_detections,
        "confirmed_detections": report.confirmed_detections,
        "findings": report.findings,
        "summary": report.summary,
    }


# ── Helpers ──────────────────────────────────────────────


def _build_finding(det: Detection, index: int) -> dict[str, Any]:
    """Build a single finding entry for the report."""
    finding: dict[str, Any] = {
        "finding_number": index,
        "class_name": det.class_name,
        "confidence": round(det.confidence, 3),
        "bounding_box": list(det.bbox),
        "track_id": det.track_id,
        "confirmed": det.confirmed,
    }

    # Add ABCDE data if available
    if det.abcde is not None:
        finding["abcde"] = {
            "asymmetry": det.abcde.asymmetry_score,
            "border": det.abcde.border_score,
            "color": det.abcde.color_score,
            "color_count": det.abcde.color_count,
            "diameter_mm": det.abcde.diameter_mm,
            "diameter_score": det.abcde.diameter_score,
            "evolution": det.abcde.evolution_score,
            "total_score": round(det.abcde.total_score, 1),
        }

    # Add risk data if available
    if det.risk is not None:
        finding["risk"] = {
            "level": det.risk.level,
            "total_score": det.risk.total_score,
            "summary": det.risk.summary,
        }

    return finding


def _build_summary(total: int, high_risk: int) -> str:
    """Build the overall report summary text."""
    if total == 0:
        return "No lesion candidates detected in this image."

    parts = [f"Detected {total} lesion candidate(s)."]

    if high_risk > 0:
        parts.append(
            f"{high_risk} finding(s) flagged as high-risk or requiring referral."
        )
    else:
        parts.append("No high-risk findings detected.")

    parts.append(
        "This is an AI-assisted screening result and does not constitute "
        "a medical diagnosis. Clinical correlation is recommended."
    )

    return " ".join(parts)
