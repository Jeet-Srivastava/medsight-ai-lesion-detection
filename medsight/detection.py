from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

_ULTRALYTICS_CONFIG_DIR = Path(".ultralytics").resolve()
_ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ULTRALYTICS_CONFIG_DIR))

from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    track_id: int | None = None
    confirmed: bool = False
    duration_frames: int = 1
    duration_seconds: float = 0.0


@dataclass
class InferenceResult:
    detections: list[Detection]
    inference_ms: float
    pipeline_ms: float
    annotated_frame: np.ndarray


class LesionDetector:
    def __init__(self, model_path: str) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model_path = model_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    @property
    def fp16_supported(self) -> bool:
        return self.device.startswith("cuda")

    def predict_image(
        self,
        frame_rgb: np.ndarray,
        confidence: float,
        use_fp16: bool = False,
    ) -> InferenceResult:
        return self._run_predict(
            mode="predict",
            frame_rgb=frame_rgb,
            confidence=confidence,
            tracker=None,
            persist=False,
            use_fp16=use_fp16,
        )

    def track_frame(
        self,
        frame_rgb: np.ndarray,
        confidence: float,
        tracker: str,
        persist: bool = True,
        use_fp16: bool = False,
    ) -> InferenceResult:
        return self._run_predict(
            mode="track",
            frame_rgb=frame_rgb,
            confidence=confidence,
            tracker=tracker,
            persist=persist,
            use_fp16=use_fp16,
        )

    def render(self, frame_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
        canvas = frame_rgb.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            confirmed = detection.confirmed
            color = (76, 215, 178) if confirmed else (108, 183, 255)
            title = "Lesion"
            if detection.track_id is not None:
                title += f" ID {detection.track_id}"
            suffix = "confirmed" if confirmed else "pending"
            label = f"{title} {detection.confidence:.2f} {suffix}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            box_top = max(0, y1 - text_h - 14)
            box_right = min(canvas.shape[1] - 1, x1 + text_w + 14)
            cv2.rectangle(canvas, (x1, box_top), (box_right, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 7, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (4, 17, 29),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def benchmark(self, frame_rgb: np.ndarray, repeats: int = 3) -> float:
        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            self.model.predict(
                source=frame_rgb,
                conf=0.25,
                imgsz=640,
                verbose=False,
                device=self.device,
                half=self.fp16_supported,
            )
            timings.append((time.perf_counter() - start) * 1000.0)
        return sum(timings) / len(timings)

    def _run_predict(
        self,
        mode: str,
        frame_rgb: np.ndarray,
        confidence: float,
        tracker: str | None,
        persist: bool,
        use_fp16: bool,
    ) -> InferenceResult:
        start = time.perf_counter()
        kwargs = {
            "source": frame_rgb,
            "conf": confidence,
            "imgsz": 640,
            "verbose": False,
            "device": self.device,
            "half": bool(use_fp16 and self.fp16_supported),
        }
        if mode == "track":
            results = self.model.track(tracker=tracker, persist=persist, **kwargs)
        else:
            results = self.model.predict(**kwargs)
        pipeline_ms = (time.perf_counter() - start) * 1000.0
        result = results[0]
        detections = self._parse_boxes(result.boxes)
        annotated = self.render(frame_rgb, detections)
        inference_ms = float(result.speed.get("inference", pipeline_ms))
        return InferenceResult(
            detections=detections,
            inference_ms=inference_ms,
            pipeline_ms=pipeline_ms,
            annotated_frame=annotated,
        )

    def _parse_boxes(self, boxes) -> list[Detection]:
        detections: list[Detection] = []
        if boxes is None:
            return detections

        track_ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()

        for index, box in enumerate(boxes):
            class_id = int(box.cls[0]) if box.cls is not None else 0
            class_name = self.class_names.get(class_id, str(class_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            track_id = track_ids[index] if track_ids is not None and index < len(track_ids) else None
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                )
            )
        return detections
