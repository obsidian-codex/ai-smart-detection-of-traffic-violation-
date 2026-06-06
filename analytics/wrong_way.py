"""Wrong-way vehicle detection via trajectory direction analysis."""
from __future__ import annotations

import math

from config.settings import (
    EXPECTED_FLOW_DIRECTION,
    WRONG_WAY_MIN_DISPLACEMENT,
    WRONG_WAY_MIN_FRAMES,
)


class WrongWayDetector:
    def __init__(
        self,
        expected_direction: tuple[float, float] = EXPECTED_FLOW_DIRECTION,
        min_displacement: float = WRONG_WAY_MIN_DISPLACEMENT,
        min_frames: int = WRONG_WAY_MIN_FRAMES,
    ):
        self.expected = self._normalize(expected_direction)
        self.min_displacement = min_displacement
        self.min_frames = min_frames
        self.flagged: set[int] = set()

    @staticmethod
    def _normalize(v: tuple[float, float]) -> tuple[float, float]:
        mag = math.hypot(v[0], v[1])
        if mag < 1e-6:
            return (0.0, -1.0)
        return (v[0] / mag, v[1] / mag)

    def check(self, track_id: int, trajectory: list[tuple[int, int]]) -> bool:
        if len(trajectory) < 2:
            return track_id in self.flagged

        previous_center = trajectory[-2]
        current_center = trajectory[-1]
        dx = current_center[0] - previous_center[0]
        dy = current_center[1] - previous_center[1]
        displacement = math.hypot(dx, dy)

        if len(trajectory) >= self.min_frames:
            start = trajectory[-self.min_frames]
            end = trajectory[-1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            displacement = math.hypot(dx, dy)

        if displacement < self.min_displacement:
            return track_id in self.flagged

        move_dir = self._normalize((dx, dy))
        dot = move_dir[0] * self.expected[0] + move_dir[1] * self.expected[1]

        if dot < -0.3:
            self.flagged.add(track_id)
            return True
        return track_id in self.flagged
