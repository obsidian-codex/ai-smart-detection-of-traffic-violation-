"""Helmet violation detection using fine-tuned YOLO11s."""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from config.settings import CONFIDENCE_THRESHOLD, HELMET_CLASSES, HELMET_MODEL_PATH, VEHICLE_MODEL


class HelmetDetector:
    def __init__(self, model_path: str | None = None, device: str = ""):
        path = model_path or str(HELMET_MODEL_PATH)
        if not __import__("pathlib").Path(path).exists():
            self.model = None
            self._fallback = True
        else:
            self.model = YOLO(path)
            self._fallback = False
        self.device = device

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def detect_on_crop(self, crop: np.ndarray) -> list[dict]:
        if crop.size == 0:
            return []
        if self._fallback:
            return self._heuristic_detect(crop)
        results = self.model.predict(
            crop, conf=CONFIDENCE_THRESHOLD, verbose=False, device=self.device or None
        )
        detections = []
        if not results or results[0].boxes is None:
            return detections
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(boxes.conf[i].item()),
                    "class_id": cls_id,
                    "class_name": HELMET_CLASSES[cls_id] if cls_id < len(HELMET_CLASSES) else "unknown",
                }
            )
        return detections

    def _heuristic_detect(self, crop: np.ndarray) -> list[dict]:
        """Demo fallback when helmet model not yet trained."""
        h, w = crop.shape[:2]
        return [
            {
                "bbox": [w * 0.2, 0, w * 0.8, h * 0.4],
                "confidence": 0.5,
                "class_id": 1,
                "class_name": "no_helmet",
            }
        ]
