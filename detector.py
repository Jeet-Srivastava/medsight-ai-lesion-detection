from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [MedSight] - %(levelname)s - %(message)s",
)


@dataclass
class FrameAnalysis:
    inference_ms: float
    total_ms: float
    detections: int
    top_confidence: float


class LesionDetector:
    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        logging.info("Loading YOLO model from %s", model_path)
        self.model = YOLO(model_path)

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        conf_threshold: float = 0.5,
        draw_boxes: bool = True,
    ) -> tuple[np.ndarray, FrameAnalysis]:
        """
        Run YOLO inference on an RGB frame and return an annotated frame plus stats.
        """
        start_time = time.perf_counter()
        results = self.model.predict(frame_rgb, conf=conf_threshold, verbose=False)
        total_ms = (time.perf_counter() - start_time) * 1000.0

        result = results[0]
        detections = result.boxes
        annotated_frame = frame_rgb.copy()

        top_confidence = 0.0
        detection_count = 0

        for box in detections:
            confidence = float(box.conf[0])
            top_confidence = max(top_confidence, confidence)
            detection_count += 1

            if not draw_boxes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            self._draw_detection(annotated_frame, x1, y1, x2, y2, confidence)

        if detection_count > 0:
            logging.info(
                "Detected %s potential lesion(s) above %.2f threshold",
                detection_count,
                conf_threshold,
            )

        model_inference_ms = float(result.speed.get("inference", total_ms))
        analysis = FrameAnalysis(
            inference_ms=model_inference_ms,
            total_ms=total_ms,
            detections=detection_count,
            top_confidence=top_confidence,
        )
        return annotated_frame, analysis

    @staticmethod
    def _draw_detection(
        frame_rgb: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        confidence: float,
    ) -> None:
        box_color = (31, 111, 235)
        accent_color = (219, 234, 254)
        label = f"Potential Lesion (Simulated) {confidence:.2f}"

        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), box_color, 2)
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        label_top = max(y1 - 28, 0)
        label_bottom = min(y1, frame_rgb.shape[0])
        label_right = min(x1 + text_width + 12, frame_rgb.shape[1] - 1)
        cv2.rectangle(
            frame_rgb,
            (x1, label_top),
            (label_right, label_bottom),
            accent_color,
            -1,
        )
        cv2.putText(
            frame_rgb,
            label,
            (x1 + 6, max(label_bottom - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (15, 23, 42),
            1,
            cv2.LINE_AA,
        )
