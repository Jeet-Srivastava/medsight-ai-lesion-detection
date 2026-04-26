from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from shutil import move

import numpy as np
from ultralytics import YOLO

from medsight.config import MODEL_EXPORT_DIR
from medsight.detection import LesionDetector


@dataclass
class OptimizationResult:
    onnx_path: Path
    fp16_enabled: bool
    pytorch_latency_ms: float
    onnx_latency_ms: float
    speedup: float
    provider: str
    error: str | None = None


class ModelOptimizer:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.export_dir = Path(MODEL_EXPORT_DIR)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def optimize(
        self,
        detector: LesionDetector,
        sample_frame: np.ndarray | None = None,
    ) -> OptimizationResult:
        sample = sample_frame if sample_frame is not None else np.zeros((640, 640, 3), dtype=np.uint8)
        fp16_enabled = detector.fp16_supported
        onnx_path = self.export_dir / f"{self.model_path.stem}.onnx"
        try:
            exported = detector.model.export(
                format="onnx",
                imgsz=640,
                half=fp16_enabled,
                dynamic=True,
                simplify=False,
                opset=19,
                project=str(self.export_dir),
                name=self.model_path.stem,
                exist_ok=True,
            )
            onnx_path = Path(exported)
            if onnx_path.parent != self.export_dir:
                destination = self.export_dir / onnx_path.name
                move(str(onnx_path), destination)
                onnx_path = destination
            pytorch_latency = detector.benchmark(sample)
            onnx_model = YOLO(str(onnx_path), task="detect")
            onnx_latency = self._benchmark_onnx(onnx_model, sample)
            speedup = pytorch_latency / onnx_latency if onnx_latency > 0 else 0.0
            provider = "onnxruntime"
            return OptimizationResult(
                onnx_path=onnx_path,
                fp16_enabled=fp16_enabled,
                pytorch_latency_ms=pytorch_latency,
                onnx_latency_ms=onnx_latency,
                speedup=speedup,
                provider=provider,
            )
        except Exception as exc:
            return OptimizationResult(
                onnx_path=onnx_path,
                fp16_enabled=fp16_enabled,
                pytorch_latency_ms=0.0,
                onnx_latency_ms=0.0,
                speedup=0.0,
                provider="unavailable",
                error=str(exc),
            )

    def _benchmark_onnx(self, model: YOLO, sample_frame: np.ndarray, repeats: int = 3) -> float:
        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.predict(source=sample_frame, conf=0.25, imgsz=640, verbose=False)
            timings.append((time.perf_counter() - start) * 1000.0)
        return sum(timings) / len(timings)
