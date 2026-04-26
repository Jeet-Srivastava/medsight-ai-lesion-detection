from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from medsight.video import VideoStream


class VideoTests(unittest.TestCase):
    def test_video_stream_reads_generated_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (64, 64),
            )
            for _ in range(3):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()

            stream = VideoStream(str(video_path))
            self.assertTrue(stream.is_opened())
            frame = stream.read_frame()
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (64, 64, 3))
            stream.release()


if __name__ == "__main__":
    unittest.main()
