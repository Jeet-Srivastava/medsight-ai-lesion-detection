from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from medsight.optimization import ModelOptimizer


class OptimizationTests(unittest.TestCase):
    @patch("medsight.optimization.YOLO")
    def test_optimizer_reports_latency(self, mock_yolo) -> None:
        mock_detector = MagicMock()
        mock_detector.fp16_supported = False
        mock_detector.model.export.return_value = "models/yolo11n.onnx"
        mock_detector.benchmark.return_value = 12.0

        mock_onnx_model = MagicMock()
        mock_yolo.return_value = mock_onnx_model
        mock_onnx_model.predict.return_value = []

        optimizer = ModelOptimizer("yolo11n.pt")
        result = optimizer.optimize(mock_detector, sample_frame=np.zeros((16, 16, 3), dtype=np.uint8))

        self.assertIsNone(result.error)
        self.assertGreater(result.pytorch_latency_ms, 0.0)
        self.assertGreaterEqual(result.onnx_latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
