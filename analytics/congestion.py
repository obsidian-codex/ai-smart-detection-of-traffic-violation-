"""Congestion classification — simple rule-based for demo."""
from __future__ import annotations


class CongestionAnalyzer:
    def classify(self, total_vehicles: int) -> tuple[str, float]:
        if total_vehicles < 20:
            level = "Low"
            score = round(total_vehicles / 20 * 33, 1)
        elif total_vehicles < 50:
            level = "Medium"
            score = round(34 + (total_vehicles - 20) / 30 * 33, 1)
        else:
            level = "High"
            score = round(min(67 + (total_vehicles - 50) / 50 * 33, 100), 1)
        return level, score
