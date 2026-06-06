"""Pipeline diagnostics for debugging detection failures."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("traffic.diagnostics")


@dataclass
class FrameDiagnostics:
    frame_idx: int
    frame_shape: tuple[int, int, int] | None = None
    video_read_ok: bool = False
    model_loaded: bool = False
    model_path: str = ""
    raw_detections: int = 0
    person_detections: int = 0
    vehicle_detections: int = 0
    vehicles_with_track_id: int = 0
    vehicles_without_track_id: int = 0
    filtered_out: int = 0
    inference_ms: float = 0.0
    classes_detected: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Frame {self.frame_idx} | shape={self.frame_shape} | "
            f"read={self.video_read_ok} | raw={self.raw_detections} | "
            f"vehicles={self.vehicle_detections} (tracked={self.vehicles_with_track_id}) | "
            f"inference={self.inference_ms:.0f}ms | classes={self.classes_detected}"
        )


class DiagnosticsLog:
    def __init__(self, max_entries: int = 50):
        self.entries: list[FrameDiagnostics] = []
        self.max_entries = max_entries
        self.model_status = "not loaded"
        self.model_path = ""

    def add(self, entry: FrameDiagnostics):
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        logger.info(entry.summary())
        if entry.errors:
            for e in entry.errors:
                logger.error("[%s] %s", entry.frame_idx, e)

    def last(self) -> FrameDiagnostics | None:
        return self.entries[-1] if self.entries else None
