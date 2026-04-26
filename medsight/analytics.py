from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

from medsight.detection import Detection
from medsight.tracking import SpatiotemporalTracker, TrackSummary


@dataclass
class FrameAnalytics:
    frame_index: int
    total_frames: int
    raw_detections: int
    confirmed_detections: int
    total_confirmed_lesions: int
    active_lesions: int
    average_confidence: float
    detection_frequency: float
    inference_ms: float
    pipeline_ms: float
    fps: float


class LesionAnalytics:
    def __init__(self, max_history: int = 180) -> None:
        self.history: deque[FrameAnalytics] = deque(maxlen=max_history)

    def reset(self) -> None:
        self.history.clear()

    def record(
        self,
        frame_index: int,
        total_frames: int,
        detections: list[Detection],
        tracker: SpatiotemporalTracker,
        inference_ms: float,
        pipeline_ms: float,
        fps: float,
    ) -> FrameAnalytics:
        confirmed = [detection for detection in detections if detection.confirmed]
        confidence_values = [detection.confidence for detection in confirmed] or [0.0]
        elapsed_seconds = frame_index / fps if fps > 0 else 0.0
        detection_frequency = (
            tracker.total_confirmed_tracks / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        )
        analytics = FrameAnalytics(
            frame_index=frame_index,
            total_frames=total_frames,
            raw_detections=len(detections),
            confirmed_detections=len(confirmed),
            total_confirmed_lesions=tracker.total_confirmed_tracks,
            active_lesions=tracker.active_confirmed_tracks,
            average_confidence=sum(confidence_values) / len(confidence_values),
            detection_frequency=detection_frequency,
            inference_ms=inference_ms,
            pipeline_ms=pipeline_ms,
            fps=fps,
        )
        self.history.append(analytics)
        return analytics


def build_track_dataframe(track_summaries: list[TrackSummary]) -> pd.DataFrame:
    if not track_summaries:
        return pd.DataFrame()
    rows = [
        {
            "Track ID": summary.track_id,
            "Frames": summary.frames_visible,
            "Duration": summary.duration_frames,
            "Avg confidence": round(summary.average_confidence, 3),
            "Confirmed": "Yes" if summary.confirmed else "Pending",
            "Active": "Yes" if summary.active else "No",
        }
        for summary in track_summaries
    ]
    return pd.DataFrame(rows)
