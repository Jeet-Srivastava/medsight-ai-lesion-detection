from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class VideoStream:
    def __init__(self, source: int | str, loop_video: bool = False) -> None:
        import cv2

        self.source = source
        self.loop_video = loop_video
        self.capture = cv2.VideoCapture(source)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) if self.capture.isOpened() else 0
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) if self.capture.isOpened() else 0.0

    def is_opened(self) -> bool:
        return bool(self.capture and self.capture.isOpened())

    def read_frame(self) -> np.ndarray | None:
        import cv2

        if not self.is_opened():
            return None
        ok, frame_bgr = self.capture.read()
        if not ok and self.loop_video and isinstance(self.source, str):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = self.capture.read()
        if not ok:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()


def save_uploaded_file(uploaded_file, upload_dir: str) -> str:
    target_dir = Path(upload_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = uploaded_file.getbuffer()
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()[:14]
    extension = Path(uploaded_file.name).suffix.lower()
    output_path = target_dir / f"{digest}{extension}"
    if not output_path.exists():
        output_path.write_bytes(payload)
    return str(output_path)


def decode_uploaded_image(image_path: str) -> np.ndarray:
    import cv2

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Unable to decode image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
