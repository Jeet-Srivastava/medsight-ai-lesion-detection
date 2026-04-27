from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from medsight.analytics import FrameAnalytics, LesionAnalytics
from medsight.detection import Detection, LesionDetector
from medsight.tracking import SpatiotemporalTracker


@dataclass
class PipelineFrameResult:
    frame_index: int
    total_frames: int
    raw_frame: np.ndarray
    preprocessed_frame: np.ndarray
    rendered_frame: np.ndarray
    raw_detections: list[Detection]
    confirmed_detections: list[Detection]
    analytics: FrameAnalytics
    logs: list[tuple[str, str]]
    snapshot_eligible: bool


class MedSightPipeline:
    def __init__(
        self,
        detector: LesionDetector,
        temporal_window: int,
        tracker_name: str,
        enable_fp16: bool,
    ) -> None:
        self.detector = detector
        self.tracker_name = tracker_name
        self.enable_fp16 = enable_fp16
        self.tracker = SpatiotemporalTracker(min_persist_frames=temporal_window)
        self.analytics = LesionAnalytics()

    def reset(self) -> None:
        self.tracker.reset()
        self.analytics.reset()

    def process_image(self, frame_rgb: np.ndarray, confidence: float) -> PipelineFrameResult:
        preprocessed = self._preprocess(frame_rgb)
        inference = self.detector.predict_image(
            frame_rgb=preprocessed,
            confidence=confidence,
            use_fp16=self.enable_fp16,
        )
        confirmed = []
        for detection in inference.detections:
            detection.confirmed = True
            confirmed.append(detection)
        rendered = self.detector.render(preprocessed, confirmed)
        fps = 1000.0 / max(inference.pipeline_ms, 1e-6)
        analytics = FrameAnalytics(
            frame_index=1,
            total_frames=1,
            raw_detections=len(inference.detections),
            confirmed_detections=len(confirmed),
            total_confirmed_lesions=len(confirmed),
            active_lesions=len(confirmed),
            average_confidence=self._average_confidence(confirmed),
            detection_frequency=float(len(confirmed)),
            inference_ms=inference.inference_ms,
            pipeline_ms=inference.pipeline_ms,
            fps=fps,
        )
        self.analytics.history.append(analytics)
        return PipelineFrameResult(
            frame_index=1,
            total_frames=1,
            raw_frame=frame_rgb,
            preprocessed_frame=preprocessed,
            rendered_frame=rendered,
            raw_detections=inference.detections,
            confirmed_detections=confirmed,
            analytics=analytics,
            logs=self._build_logs(frame_index=1, detections=confirmed),
            snapshot_eligible=bool(confirmed),
        )

    def process_video_frame(
        self,
        frame_rgb: np.ndarray,
        frame_index: int,
        total_frames: int,
        confidence: float,
    ) -> PipelineFrameResult:
        start = time.perf_counter()
        preprocessed = self._preprocess(frame_rgb)
        inference = self.detector.track_frame(
            frame_rgb=preprocessed,
            confidence=confidence,
            tracker=self.tracker_name,
            persist=True,
            use_fp16=self.enable_fp16,
        )
        instant_fps = 1000.0 / max(inference.pipeline_ms, 1e-6)
        tracked = self.tracker.update(
            detections=inference.detections,
            frame_index=frame_index,
            fps=instant_fps,
        )
        confirmed = [detection for detection in tracked if detection.confirmed]
        rendered = self.detector.render(preprocessed, confirmed)
        pipeline_ms = (time.perf_counter() - start) * 1000.0
        fps = 1000.0 / max(pipeline_ms, 1e-6)
        analytics = self.analytics.record(
            frame_index=frame_index,
            total_frames=total_frames,
            detections=tracked,
            tracker=self.tracker,
            inference_ms=inference.inference_ms,
            pipeline_ms=pipeline_ms,
            fps=fps,
        )
        return PipelineFrameResult(
            frame_index=frame_index,
            total_frames=total_frames,
            raw_frame=frame_rgb,
            preprocessed_frame=preprocessed,
            rendered_frame=rendered,
            raw_detections=tracked,
            confirmed_detections=confirmed,
            analytics=analytics,
            logs=self._build_logs(frame_index=frame_index, detections=tracked),
            snapshot_eligible=bool(confirmed),
        )

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        import cv2

        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    def _build_logs(self, frame_index: int, detections: list[Detection]) -> list[tuple[str, str]]:
        logs: list[tuple[str, str]] = [("info", f"Frame {frame_index} processed")]
        for detection in detections:
            if detection.confirmed:
                logs.append(("detect", f"Lesion detected at frame {frame_index} with {detection.confidence:.2f} confidence"))
            if detection.track_id is not None:
                logs.append(("track", f"ID {detection.track_id} updated ({detection.duration_frames} frames)"))
        return logs

    @staticmethod
    def _average_confidence(detections: list[Detection]) -> float:
        if not detections:
            return 0.0
        return sum(detection.confidence for detection in detections) / len(detections)
