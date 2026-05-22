"""Audit trail for clinical compliance.

Every inference gets recorded in an append-only JSONL log file.
This is essential for healthcare systems — you need to be able to
trace back exactly what was analyzed, when, and with what parameters.

The log is file-based (no database needed), human-readable, and
easy to ship to a compliance system later.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# Default location for the audit log
AUDIT_DIR = Path("audit")
AUDIT_FILE = AUDIT_DIR / "audit_trail.jsonl"


class AuditLogger:
    """Append-only audit trail writer and reader."""

    def __init__(self, audit_file: Path | None = None) -> None:
        self.audit_file = audit_file or AUDIT_FILE
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)

    def log_inference(
        self,
        session_id: str,
        frame_rgb: np.ndarray | None,
        model_path: str,
        confidence_threshold: float,
        detections_count: int,
        confirmed_count: int,
        high_risk_count: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a single inference event to the audit trail.

        Args:
            session_id:           The session this inference belongs to.
            frame_rgb:            The input image (hashed, not stored).
            model_path:           Which model weights were used.
            confidence_threshold: The confidence threshold applied.
            detections_count:     Total number of raw detections.
            confirmed_count:      Number of confirmed detections.
            high_risk_count:      Number of high-risk findings.
            extra:                Any additional metadata to log.

        Returns:
            The audit entry dict that was written.
        """
        now = datetime.now(timezone.utc)

        # Hash the image for identity — we don't store the actual pixels
        if frame_rgb is not None:
            sample = frame_rgb[::10, ::10].tobytes()
            input_hash = hashlib.sha256(sample).hexdigest()[:16]
        else:
            input_hash = "unknown"

        entry = {
            "timestamp": now.isoformat(),
            "session_id": session_id,
            "input_hash": input_hash,
            "model_path": model_path,
            "confidence_threshold": confidence_threshold,
            "detections_count": detections_count,
            "confirmed_count": confirmed_count,
            "high_risk_count": high_risk_count,
        }

        if extra:
            entry["extra"] = extra

        # Append to the JSONL file (one JSON object per line)
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def get_recent_entries(self, n: int = 50) -> list[dict[str, Any]]:
        """Read the last N entries from the audit trail.

        Returns entries in reverse chronological order (newest first).
        """
        if not self.audit_file.exists():
            return []

        entries: list[dict[str, Any]] = []
        with open(self.audit_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # Skip corrupted lines

        # Return the last N, newest first
        return list(reversed(entries[-n:]))

    def get_entry_count(self) -> int:
        """Count the total number of audit entries."""
        if not self.audit_file.exists():
            return 0

        count = 0
        with open(self.audit_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
