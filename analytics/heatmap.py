"""Traffic density and violation heatmap generation."""
from __future__ import annotations

import cv2
import numpy as np


class HeatmapGenerator:
    def __init__(self, decay: float = 0.995):
        self.decay = decay
        self.density_map: np.ndarray | None = None
        self.violation_map: np.ndarray | None = None
        self.shape: tuple[int, int] | None = None

    def _init_maps(self, h: int, w: int):
        if self.density_map is None or self.shape != (h, w):
            self.density_map = np.zeros((h, w), dtype=np.float32)
            self.violation_map = np.zeros((h, w), dtype=np.float32)
            self.shape = (h, w)

    def add_point(self, x: int, y: int, weight: float = 1.0):
        if self.density_map is None:
            return
        h, w = self.shape
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(self.density_map, (x, y), 15, weight, -1)

    def add_violation(self, x: int, y: int, weight: float = 3.0):
        if self.violation_map is None:
            return
        h, w = self.shape
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(self.violation_map, (x, y), 20, weight, -1)

    def update_shape(self, h: int, w: int):
        self._init_maps(h, w)

    def tick(self):
        if self.density_map is not None:
            self.density_map *= self.decay
            self.violation_map *= self.decay

    def render(self, base_frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        h, w = base_frame.shape[:2]
        self._init_maps(h, w)
        combined = self.density_map + self.violation_map * 1.5
        if combined.max() > 0:
            norm = (combined / combined.max() * 255).astype(np.uint8)
        else:
            norm = combined.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(base_frame, 1 - alpha, heatmap_color, alpha, 0)

    def get_density_hotspots(self, top_n: int = 5) -> list[dict]:
        if self.density_map is None or self.density_map.max() == 0:
            return []
        flat = self.density_map.flatten()
        indices = np.argpartition(flat, -top_n)[-top_n:]
        h, w = self.shape
        hotspots = []
        for idx in indices:
            y, x = divmod(int(idx), w)
            hotspots.append({"x": x, "y": y, "intensity": float(flat[idx])})
        return sorted(hotspots, key=lambda h: h["intensity"], reverse=True)
