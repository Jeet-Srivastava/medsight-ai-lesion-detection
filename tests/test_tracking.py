from __future__ import annotations

import unittest

from medsight.detection import Detection
from medsight.tracking import SpatiotemporalTracker


class TrackingTests(unittest.TestCase):
    def test_temporal_confirmation_requires_persistence(self) -> None:
        tracker = SpatiotemporalTracker(min_persist_frames=3)
        detection = Detection(class_id=0, class_name="obj", confidence=0.8, bbox=(0, 0, 10, 10), track_id=7)

        tracker.update([detection], frame_index=1, fps=20.0)
        self.assertFalse(detection.confirmed)

        detection = Detection(class_id=0, class_name="obj", confidence=0.8, bbox=(0, 0, 10, 10), track_id=7)
        tracker.update([detection], frame_index=2, fps=20.0)
        self.assertFalse(detection.confirmed)

        detection = Detection(class_id=0, class_name="obj", confidence=0.8, bbox=(0, 0, 10, 10), track_id=7)
        tracker.update([detection], frame_index=3, fps=20.0)
        self.assertTrue(detection.confirmed)
        self.assertEqual(tracker.total_confirmed_tracks, 1)


if __name__ == "__main__":
    unittest.main()
