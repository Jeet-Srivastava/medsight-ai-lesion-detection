from __future__ import annotations

import numpy as np

from medsight.detection import LesionDetector
from medsight.pipeline import MedSightPipeline


def main() -> None:
    detector = LesionDetector("yolov8n.pt")
    pipeline = MedSightPipeline(
        detector=detector,
        temporal_window=2,
        tracker_name="bytetrack.yaml",
        enable_fp16=True,
    )
    sample = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.process_image(sample, confidence=0.25)
    print("Pipeline smoke test complete")
    print(f"Confirmed detections: {len(result.confirmed_detections)}")
    print(f"Inference time: {result.analytics.inference_ms:.2f} ms")


if __name__ == "__main__":
    main()
