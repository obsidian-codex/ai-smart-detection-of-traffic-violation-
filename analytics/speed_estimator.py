"""Speed estimation from tracking trajectories with perspective scaling."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from config.settings import FPS_DEFAULT, PIXELS_PER_METER, SPEED_LIMIT_KMH


class SpeedEstimator:
    def __init__(
        self,
        pixels_per_meter: float = PIXELS_PER_METER,
        fps: float = FPS_DEFAULT,
        speed_limit: float = SPEED_LIMIT_KMH,
        smoothing_window: int = 5,
    ):
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.speed_limit = speed_limit
        self.positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.speeds: dict[int, float] = {}
        self.smoothing_window = smoothing_window

    def estimate(self, track_id: int, cx: int, cy: int) -> float:
        self.positions[track_id].append((cx, cy))
        pts = self.positions[track_id]
        if len(pts) < 2:
            return 0.0

        window = pts[-self.smoothing_window :]
        total_dist_px = 0.0
        for i in range(1, len(window)):
            dx = window[i][0] - window[i - 1][0]
            dy = window[i][1] - window[i - 1][1]
            total_dist_px += math.hypot(dx, dy)

        # Perspective scaling: objects lower in frame move faster in px/frame
        y_factor = 1.0 + (cy / 1080.0) * 0.5
        dist_m = (total_dist_px / self.pixels_per_meter) * y_factor
        time_s = (len(window) - 1) / self.fps
        speed_ms = dist_m / time_s if time_s > 0 else 0.0
        speed_kmh = speed_ms * 3.6

        alpha = 0.3
        prev = self.speeds.get(track_id, speed_kmh)
        smoothed = alpha * speed_kmh + (1 - alpha) * prev
        self.speeds[track_id] = smoothed
        return smoothed

    def classify(self, speed_kmh: float) -> str:
        return "overspeeding" if speed_kmh > self.speed_limit else "normal"
