"""Tripling detection — motorcycles carrying more than 2 riders."""
from __future__ import annotations

import math
from collections import defaultdict

from config.settings import TRIPLING_BBOX_MARGIN, TRIPLING_MIN_RIDERS


class TriplingDetector:
    """Associate person detections with motorcycles and flag tripling violations."""

    def __init__(
        self,
        bbox_margin: float = TRIPLING_BBOX_MARGIN,
        min_riders_for_violation: int = TRIPLING_MIN_RIDERS,
    ):
        self.bbox_margin = bbox_margin
        self.min_riders = min_riders_for_violation
        self.flagged_ids: set[int] = set()

    @staticmethod
    def _expand_bbox(bbox: list[float], margin: float) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        return x1 - w * margin, y1 - h * margin, x2 + w * margin, y2 + h * margin

    @staticmethod
    def _point_in_bbox(px: float, py: float, bbox: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        return x1 <= px <= x2 and y1 <= py <= y2

    @staticmethod
    def _dist_to_bbox(px: float, py: float, bbox: list[float]) -> float:
        x1, y1, x2, y2 = bbox
        dx = max(x1 - px, 0, px - x2)
        dy = max(y1 - py, 0, py - y2)
        return math.hypot(dx, dy)

    @staticmethod
    def _bbox_center(bbox: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _assign_persons(self, motorcycles: list[dict], persons: list[dict]) -> dict[int, list[dict]]:
        """Assign each person to the nearest motorcycle (inside or very close to bbox)."""
        assignment: dict[int, list[dict]] = defaultdict(list)
        if not motorcycles or not persons:
            return assignment

        for person in persons:
            pcx, pcy = person["center"]
            best_id: int | None = None
            best_score = float("inf")

            for moto in motorcycles:
                expanded = self._expand_bbox(moto["bbox"], self.bbox_margin)
                inside = self._point_in_bbox(pcx, pcy, expanded)
                dist = self._dist_to_bbox(pcx, pcy, moto["bbox"])
                moto_cx, moto_cy = self._bbox_center(moto["bbox"])
                center_dist = math.hypot(pcx - moto_cx, pcy - moto_cy)
                moto_w = moto["bbox"][2] - moto["bbox"][0]
                proximity = max(20.0, moto_w * 0.25)

                if inside:
                    score = center_dist
                elif dist <= proximity:
                    score = dist + center_dist * 0.5
                else:
                    continue

                if score < best_score:
                    best_score = score
                    best_id = moto["track_id"]

            if best_id is not None:
                assignment[best_id].append(person)

        return assignment

    def detect(self, motorcycles: list[dict], persons: list[dict]) -> list[dict]:
        """
        Return frame-level tripling violations.

        1 person = rider only (legal)
        2 persons = legal
        3+ persons = tripling violation
        """
        assignment = self._assign_persons(motorcycles, persons)
        violations = []

        for moto in motorcycles:
            tid = moto["track_id"]
            rider_count = len(assignment.get(tid, []))
            if rider_count >= self.min_riders:
                self.flagged_ids.add(tid)
                violations.append(
                    {
                        "track_id": tid,
                        "rider_count": rider_count,
                        "bbox": moto["bbox"],
                        "center": moto["center"],
                    }
                )

        return violations
