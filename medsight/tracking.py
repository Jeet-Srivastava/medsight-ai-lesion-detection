from __future__ import annotations

from dataclasses import dataclass

from medsight.detection import Detection


@dataclass
class TrackSummary:
    track_id: int
    frames_visible: int
    duration_frames: int
    duration_seconds: float
    average_confidence: float
    last_confidence: float
    confirmed: bool
    active: bool


@dataclass
class _TrackState:
    track_id: int
    first_frame: int
    last_frame: int
    frames_visible: int
    consecutive_frames: int
    total_confidence: float
    last_confidence: float
    confirmed: bool = False
    active: bool = False

    @property
    def average_confidence(self) -> float:
        return self.total_confidence / max(self.frames_visible, 1)


class SpatiotemporalTracker:
    def __init__(self, min_persist_frames: int) -> None:
        self.min_persist_frames = min_persist_frames
        self.tracks: dict[int, _TrackState] = {}
        self.active_ids: set[int] = set()

    def reset(self) -> None:
        self.tracks.clear()
        self.active_ids.clear()

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        fps: float,
    ) -> list[Detection]:
        self.active_ids = set()
        for state in self.tracks.values():
            state.active = False

        for detection in detections:
            if detection.track_id is None:
                continue

            state = self.tracks.get(detection.track_id)
            if state is None:
                state = _TrackState(
                    track_id=detection.track_id,
                    first_frame=frame_index,
                    last_frame=frame_index,
                    frames_visible=1,
                    consecutive_frames=1,
                    total_confidence=detection.confidence,
                    last_confidence=detection.confidence,
                )
                self.tracks[detection.track_id] = state
            else:
                gap = frame_index - state.last_frame
                state.consecutive_frames = state.consecutive_frames + 1 if gap == 1 else 1
                state.frames_visible += 1
                state.total_confidence += detection.confidence
                state.last_frame = frame_index
                state.last_confidence = detection.confidence

            state.last_frame = frame_index
            state.active = True
            if state.consecutive_frames >= self.min_persist_frames:
                state.confirmed = True

            duration_frames = state.last_frame - state.first_frame + 1
            detection.confirmed = state.confirmed
            detection.duration_frames = duration_frames
            detection.duration_seconds = duration_frames / fps if fps > 0 else 0.0
            self.active_ids.add(detection.track_id)

        return detections

    @property
    def total_confirmed_tracks(self) -> int:
        return sum(1 for state in self.tracks.values() if state.confirmed)

    @property
    def active_confirmed_tracks(self) -> int:
        return sum(1 for track_id in self.active_ids if self.tracks[track_id].confirmed)

    def snapshot(self) -> list[TrackSummary]:
        summaries: list[TrackSummary] = []
        for state in sorted(self.tracks.values(), key=lambda item: item.track_id):
            duration_frames = state.last_frame - state.first_frame + 1
            summaries.append(
                TrackSummary(
                    track_id=state.track_id,
                    frames_visible=state.frames_visible,
                    duration_frames=duration_frames,
                    duration_seconds=0.0,
                    average_confidence=state.average_confidence,
                    last_confidence=state.last_confidence,
                    confirmed=state.confirmed,
                    active=state.active,
                )
            )
        return summaries
