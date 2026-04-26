from __future__ import annotations

import unittest

import numpy as np

from medsight.detection import LesionDetector
from medsight.pipeline import MedSightPipeline


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = LesionDetector("yolov8n.pt")

    def test_image_pipeline_returns_metrics(self) -> None:
        pipeline = MedSightPipeline(
            detector=self.detector,
            temporal_window=2,
            tracker_name="bytetrack.yaml",
            enable_fp16=False,
        )
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        result = pipeline.process_image(frame, confidence=0.25)
        self.assertEqual(result.frame_index, 1)
        self.assertIsNotNone(result.rendered_frame)
        self.assertGreaterEqual(result.analytics.inference_ms, 0.0)
        self.assertGreaterEqual(result.analytics.pipeline_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
