"""Vehicle detection using pretrained YOLO11s."""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from config.settings import (
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    VEHICLE_CLASS_IDS,
    VEHICLE_MODEL,
)


class VehicleDetector:
    def __init__(self, model_path: str | None = None, device: str = ""):
        self.model = YOLO(model_path or VEHICLE_MODEL)
        self.allowed_ids = set(VEHICLE_CLASS_IDS.keys())
        self.device = device

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=list(self.allowed_ids),
            verbose=False,
            device=self.device or None,
        )
        detections = []
        if not results or results[0].boxes is None:
            return detections
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in self.allowed_ids:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].item())
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": VEHICLE_CLASS_IDS[cls_id],
                }
            )
        return detections
