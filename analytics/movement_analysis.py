"""Vehicle movement pattern and direction analytics."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


class MovementAnalyzer:
    DIRECTIONS = ["North", "South", "East", "West", "NE", "NW", "SE", "SW"]

    def __init__(self):
        self.trajectories: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.classes: dict[int, str] = {}

    def record(self, track_id: int, cx: int, cy: int, class_name: str):
        self.trajectories[track_id].append((cx, cy))
        self.classes[track_id] = class_name
        if len(self.trajectories[track_id]) > 200:
            self.trajectories[track_id] = self.trajectories[track_id][-200:]

    def _angle_to_direction(self, dx: float, dy: float) -> str:
        angle = math.degrees(math.atan2(-dy, dx))
        if angle < 0:
            angle += 360
        sectors = [
            (337.5, 360, "East"),
            (0, 22.5, "East"),
            (22.5, 67.5, "SE"),
            (67.5, 112.5, "South"),
            (112.5, 157.5, "SW"),
            (157.5, 202.5, "West"),
            (202.5, 247.5, "NW"),
            (247.5, 292.5, "North"),
            (292.5, 337.5, "NE"),
        ]
        for lo, hi, name in sectors:
            if lo <= angle < hi:
                return name
        return "Unknown"

    def get_direction_stats(self) -> dict[str, int]:
        stats: dict[str, int] = defaultdict(int)
        for tid, pts in self.trajectories.items():
            if len(pts) < 5:
                continue
            dx = pts[-1][0] - pts[0][0]
            dy = pts[-1][1] - pts[0][1]
            direction = self._angle_to_direction(dx, dy)
            stats[direction] += 1
        return dict(stats)

    def get_flow_imbalance(self) -> dict:
        stats = self.get_direction_stats()
        if not stats:
            return {"imbalanced": False, "ratio": 1.0}
        values = list(stats.values())
        max_flow = max(values)
        min_flow = min(values) if min(values) > 0 else 1
        ratio = max_flow / min_flow
        return {
            "imbalanced": ratio > 2.5,
            "ratio": round(ratio, 2),
            "dominant": max(stats, key=stats.get),
            "stats": stats,
        }

    def get_trajectory_overlay_data(self) -> list[dict]:
        data = []
        for tid, pts in self.trajectories.items():
            if len(pts) < 2:
                continue
            data.append(
                {
                    "track_id": tid,
                    "class": self.classes.get(tid, "unknown"),
                    "points": pts,
                }
            )
        return data
