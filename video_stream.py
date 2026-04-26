from __future__ import annotations

import hashlib
from pathlib import Path

import cv2


class VideoStream:
    def __init__(self, source: int | str = 0, loop_video: bool = False) -> None:
        self.source = source
        self.loop_video = loop_video
        self.cap = cv2.VideoCapture(self.source)

    def is_opened(self) -> bool:
        return bool(self.cap and self.cap.isOpened())

    def read_frame(self):
        if not self.is_opened():
            return None

        success, frame_bgr = self.cap.read()
        if not success and self.loop_video and isinstance(self.source, str):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame_bgr = self.cap.read()

        if not success:
            return None

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self.cap:
            self.cap.release()


def save_uploaded_video(uploaded_file, upload_dir: str = "uploads") -> str:
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    file_bytes = uploaded_file.getbuffer()
    file_hash = hashlib.md5(file_bytes, usedforsecurity=False).hexdigest()[:12]
    extension = Path(uploaded_file.name).suffix or ".mp4"
    saved_path = upload_path / f"{file_hash}{extension}"

    if not saved_path.exists():
        saved_path.write_bytes(file_bytes)

    return str(saved_path)
