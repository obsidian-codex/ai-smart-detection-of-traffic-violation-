"""End-to-end traffic analysis pipeline with ByteTrack."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

from analytics.congestion import CongestionAnalyzer
from analytics.heatmap import HeatmapGenerator
from analytics.movement_analysis import MovementAnalyzer
from analytics.risk_score import RiskScoreCalculator
from analytics.speed_estimator import SpeedEstimator
from analytics.tripling_detector import TriplingDetector
from analytics.wrong_way import WrongWayDetector
from config.settings import (
    CONFIDENCE_THRESHOLD,
    DETECTION_CLASS_IDS,
    FPS_DEFAULT,
    IOU_THRESHOLD,
    PERSON_CLASS_ID,
    TRACKER_CONFIG,
    VEHICLE_CLASS_IDS,
    VEHICLE_MODEL,
)
from tracking.diagnostics import DiagnosticsLog, FrameDiagnostics
from tracking.helmet_detector import HelmetDetector


@dataclass
class FrameAnalytics:
    frame_idx: int
    timestamp: str
    vehicle_counts: dict
    total_vehicles: int
    active_tracks: int
    congestion_level: str
    congestion_score: float
    avg_speed_kmh: float
    helmet_violations: int
    tripling_violations: int
    tripling_events: list
    tripling_vehicle_ids: list
    wrong_way_events: list
    overspeeding_vehicles: list
    risk_level: str
    risk_score: float
    risk_reasons: list = field(default_factory=list)
    cumulative_counts: dict = field(default_factory=dict)
    track_data: list = field(default_factory=list)
    annotated_frame: np.ndarray | None = None
    debug: dict = field(default_factory=dict)


class TrafficPipeline:
    COLORS = {
        "car": (0, 200, 255),
        "motorcycle": (255, 150, 0),
        "bus": (0, 255, 100),
        "truck": (200, 100, 255),
        "bicycle": (100, 255, 255),
    }

    def __init__(
        self,
        flow_direction: tuple[float, float] = (0, -1),
        pixels_per_meter: float = 8.0,
        fps: float = FPS_DEFAULT,
        speed_limit: float = 60.0,
        device: str = "",
    ):
        self.device = device
        self.fps = fps
        vehicle_weights = VEHICLE_MODEL
        self.yolo = YOLO(vehicle_weights)
        self.helmet_detector = HelmetDetector(device=device)
        self.speed_estimator = SpeedEstimator(pixels_per_meter, fps, speed_limit)
        self.wrong_way = WrongWayDetector(flow_direction)
        self.congestion = CongestionAnalyzer()
        self.heatmap = HeatmapGenerator()
        self.movement = MovementAnalyzer()
        self.risk_calc = RiskScoreCalculator()
        self.tripling = TriplingDetector()
        self.trajectories: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.track_classes: dict[int, str] = {}
        self.helmet_violation_ids: set[int] = set()
        self.wrong_way_log: list[dict] = []
        self.overspeed_log: list[dict] = []
        self.overspeed_ids: set[int] = set()
        self.total_helmet_violations = 0
        self.tripling_violation_ids: set[int] = set()
        self.tripling_log: list[dict] = []
        self.frame_history: list[FrameAnalytics] = []
        self.seen_tracks: dict[int, str] = {}
        self._class_label = {
            "car": "Car",
            "motorcycle": "Bike",
            "bicycle": "Bike",
            "bus": "Bus",
            "truck": "Truck",
        }
        self.diagnostics = DiagnosticsLog()
        ckpt = getattr(self.yolo, "ckpt_path", None) or VEHICLE_MODEL
        model_path = Path(ckpt)
        if not model_path.is_absolute():
            local = Path.cwd() / ckpt
            model_path = local if local.exists() else model_path
        self.diagnostics.model_path = str(model_path.resolve()) if model_path.exists() else str(ckpt)
        self.diagnostics.model_status = "loaded" if model_path.exists() else "loaded (auto-download)"

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameAnalytics:
        dbg = FrameDiagnostics(
            frame_idx=frame_idx,
            frame_shape=frame.shape if frame is not None else None,
            video_read_ok=frame is not None and frame.size > 0,
            model_loaded=True,
            model_path=self.diagnostics.model_path,
        )
        h, w = frame.shape[:2]
        self.heatmap.update_shape(h, w)
        self.heatmap.tick()

        t0 = time.perf_counter()
        # Track vehicles only; persons via separate predict (more reliable)
        vehicle_classes = list(VEHICLE_CLASS_IDS.keys())
        results = self.yolo.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            classes=vehicle_classes,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=640,
            verbose=False,
            device=self.device or None,
        )
        dbg.inference_ms = (time.perf_counter() - t0) * 1000

        # Person detections for tripling (no tracking needed)
        person_results = self.yolo.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            imgsz=640,
            verbose=False,
            device=self.device or None,
        )

        track_data = []
        motorcycles = []
        persons = []
        speeds = []
        frame_overspeed = []
        motorcycle_crops = []

        if person_results and person_results[0].boxes is not None:
            for i in range(len(person_results[0].boxes)):
                x1, y1, x2, y2 = person_results[0].boxes.xyxy[i].cpu().numpy()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                persons.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "center": (cx, cy)})
                dbg.person_detections += 1

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            dbg.raw_detections = len(boxes)
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                bbox = [float(x1), float(y1), float(x2), float(y2)]

                if cls_id not in VEHICLE_CLASS_IDS:
                    dbg.filtered_out += 1
                    continue

                dbg.vehicle_detections += 1
                cls_name = VEHICLE_CLASS_IDS[cls_id]
                if cls_name not in dbg.classes_detected:
                    dbg.classes_detected.append(cls_name)

                has_track_id = boxes.id is not None
                if has_track_id:
                    track_id = int(boxes.id[i].item())
                    dbg.vehicles_with_track_id += 1
                else:
                    track_id = 100000 + frame_idx * 1000 + i
                    dbg.vehicles_without_track_id += 1
                if has_track_id:
                    self.track_classes[track_id] = cls_name
                    self.seen_tracks[track_id] = cls_name
                    self.trajectories[track_id].append((cx, cy))
                    if len(self.trajectories[track_id]) > 120:
                        self.trajectories[track_id] = self.trajectories[track_id][-120:]
                    speed = self.speed_estimator.estimate(track_id, cx, cy)
                    speeds.append(speed)
                    self.movement.record(track_id, cx, cy, cls_name)
                else:
                    speed = 0.0

                self.heatmap.add_point(cx, cy)

                if has_track_id and cls_name == "motorcycle":
                    motorcycles.append(
                        {"track_id": track_id, "bbox": bbox, "center": (cx, cy)}
                    )
                    w_crop = x2 - x1
                    h_crop = y2 - y1
                    cx1 = max(0, int(x1 - w_crop * 0.1))
                    cy1 = max(0, int(y1 - h_crop * 0.1))
                    cx2 = min(frame.shape[1], int(x2 + w_crop * 0.1))
                    cy2 = min(frame.shape[0], int(y2 + h_crop * 0.1))
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        motorcycle_crops.append((track_id, crop, cx1, cy1))
                elif has_track_id and cls_name == "bicycle":
                    w_crop = x2 - x1
                    h_crop = y2 - y1
                    cx1 = max(0, int(x1 - w_crop * 0.1))
                    cy1 = max(0, int(y1 - h_crop * 0.1))
                    cx2 = min(frame.shape[1], int(x2 + w_crop * 0.1))
                    cy2 = min(frame.shape[0], int(y2 + h_crop * 0.1))
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        motorcycle_crops.append((track_id, crop, cx1, cy1))

                if has_track_id:
                    is_overspeed = speed > self.speed_estimator.speed_limit
                    if is_overspeed:
                        frame_overspeed.append({"track_id": track_id, "speed_kmh": round(speed, 1)})
                        if track_id not in self.overspeed_ids:
                            self.overspeed_ids.add(track_id)
                            self.overspeed_log.append(
                                {
                                    "track_id": track_id,
                                    "speed_kmh": round(speed, 1),
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )

                track_data.append(
                    {
                        "track_id": track_id,
                        "class": cls_name,
                        "bbox": bbox,
                        "speed_kmh": round(speed, 1),
                        "center": (cx, cy),
                        "has_track_id": has_track_id,
                    }
                )

        # Tripling detection (reuses same YOLO person + motorcycle boxes)
        tripling_events = self.tripling.detect(motorcycles, persons)
        for event in tripling_events:
            tid = event["track_id"]
            if tid not in self.tripling_violation_ids:
                self.tripling_violation_ids.add(tid)
                self.tripling_log.append(
                    {
                        "track_id": tid,
                        "rider_count": event["rider_count"],
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "frame_idx": frame_idx,
                    }
                )
            self.heatmap.add_violation(event["center"][0], event["center"][1])

        # Helmet detection on motorcycle/bicycle crops
        frame_helmet_violations = 0
        helmet_boxes = []
        for track_id, crop, mx1, my1 in motorcycle_crops:
            helmet_dets = self.helmet_detector.detect_on_crop(crop)
            for det in helmet_dets:
                if det["class_name"] == "no_helmet" and det["confidence"] > 0.20:
                    if track_id not in self.helmet_violation_ids:
                        self.helmet_violation_ids.add(track_id)
                        self.total_helmet_violations += 1
                    frame_helmet_violations += 1
                    self.heatmap.add_violation(crop.shape[1] // 2 + mx1, crop.shape[0] // 4 + my1)
                    hx1, hy1, hx2, hy2 = det["bbox"]
                    helmet_boxes.append((int(hx1 + mx1), int(hy1 + my1), int(hx2 + mx1), int(hy2 + my1)))

        # Wrong-way detection
        wrong_way_events = []
        for td in track_data:
            if not td.get("has_track_id", True):
                continue
            tid = td["track_id"]
            traj = self.trajectories.get(tid, [])
            if self.wrong_way.check(tid, traj):
                event = {
                    "track_id": tid,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "class": td["class"],
                }
                wrong_way_events.append(event)
                if not any(w["track_id"] == tid for w in self.wrong_way_log):
                    self.wrong_way_log.append(event)
                self.heatmap.add_violation(td["center"][0], td["center"][1])

        # Vehicle counts
        counts = defaultdict(int)
        for td in track_data:
            cls = td["class"]
            if cls == "car":
                counts["cars"] += 1
            elif cls in ("motorcycle", "bicycle"):
                counts["bikes"] += 1
            elif cls == "bus":
                counts["buses"] += 1
            elif cls == "truck":
                counts["trucks"] += 1
        cumulative = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0}
        for cls in self.seen_tracks.values():
            if cls == "car":
                cumulative["cars"] += 1
            elif cls in ("motorcycle", "bicycle"):
                cumulative["bikes"] += 1
            elif cls == "bus":
                cumulative["buses"] += 1
            elif cls == "truck":
                cumulative["trucks"] += 1

        counts_dict = {
            "cars": counts["cars"],
            "bikes": counts["bikes"],
            "buses": counts["buses"],
            "trucks": counts["trucks"],
        }
        total = sum(counts_dict.values())
        total_seen = sum(cumulative.values())

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        active_tracks = len(track_data)

        cong_level, cong_score = self.congestion.classify(active_tracks)
        risk_level, risk_score, risk_reasons = self.risk_calc.calculate(
            cong_level,
            cong_score,
            len(self.wrong_way_log),
            self.total_helmet_violations,
            len(self.overspeed_log),
            len(self.tripling_violation_ids),
        )

        annotated = self._annotate(
            frame, track_data, cumulative, cong_level, risk_level, risk_score, risk_reasons, helmet_boxes
        )
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        analytics = FrameAnalytics(
            frame_idx=frame_idx,
            timestamp=ts,
            vehicle_counts=counts_dict,
            total_vehicles=total,
            active_tracks=active_tracks,
            congestion_level=cong_level,
            congestion_score=cong_score,
            avg_speed_kmh=round(avg_speed, 1),
            helmet_violations=self.total_helmet_violations,
            tripling_violations=len(self.tripling_violation_ids),
            tripling_events=tripling_events,
            tripling_vehicle_ids=sorted(self.tripling_violation_ids),
            wrong_way_events=wrong_way_events,
            overspeeding_vehicles=frame_overspeed,
            risk_level=risk_level,
            risk_score=risk_score,
            risk_reasons=risk_reasons,
            cumulative_counts=cumulative,
            track_data=track_data,
            annotated_frame=annotated,
            debug={
                "raw_detections": dbg.raw_detections,
                "vehicle_detections": dbg.vehicle_detections,
                "person_detections": dbg.person_detections,
                "vehicles_with_track_id": dbg.vehicles_with_track_id,
                "vehicles_without_track_id": dbg.vehicles_without_track_id,
                "inference_ms": round(dbg.inference_ms, 1),
                "classes_detected": dbg.classes_detected,
                "model_path": dbg.model_path,
                "frame_shape": dbg.frame_shape,
            },
        )
        self.diagnostics.add(dbg)
        self.frame_history.append(analytics)
        return analytics

    def _annotate(
        self,
        frame: np.ndarray,
        track_data: list,
        cumulative: dict,
        cong_level: str,
        risk_level: str,
        risk_score: float,
        risk_reasons: list,
        helmet_boxes: list | None = None,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        for td in track_data:
            tid = td["track_id"]
            cls_name = td["class"]
            x1, y1, x2, y2 = [int(v) for v in td["bbox"]]
            color = self.COLORS.get(cls_name, (255, 255, 255))
            is_wrong_way = tid in self.wrong_way.flagged
            is_tripling = tid in self.tripling_violation_ids
            if tid in self.helmet_violation_ids or is_wrong_way or is_tripling:
                color = (0, 0, 255)
            if is_wrong_way:
                cv2.putText(
                    out, "WRONG WAY!", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                )
            if is_tripling and cls_name == "motorcycle":
                cv2.putText(
                    out, "TRIPLING VIOLATION", (x1, y1 - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                )
            if tid in self.helmet_violation_ids:
                y_pos = y1 - 40 if (is_tripling and cls_name == "motorcycle") else y1 - 24
                cv2.putText(
                    out, "NO HELMET", (x1, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                )
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if td.get("has_track_id", True):
                label = f"{self._class_label.get(cls_name, cls_name)} ID: {tid}"
            else:
                label = f"{self._class_label.get(cls_name, cls_name)}"
            cv2.putText(out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for tid, pts in self.trajectories.items():
            if len(pts) > 1:
                col = (0, 0, 255) if tid in self.wrong_way.flagged else (0, 255, 255)
                for j in range(1, len(pts)):
                    cv2.line(out, pts[j - 1], pts[j], col, 2)

        if helmet_boxes:
            for (hx1, hy1, hx2, hy2) in helmet_boxes:
                cv2.rectangle(out, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)

        # HUD panel
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 175), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        y = 22
        lines = [
            f"Cars: {cumulative['cars']}  Bikes: {cumulative['bikes']}  "
            f"Buses: {cumulative['buses']}  Trucks: {cumulative['trucks']}",
            f"Congestion: {cong_level}  |  Helmet: {self.total_helmet_violations}  |  "
            f"Tripling: {len(self.tripling_violation_ids)}",
            f"Wrong-Way: {len(self.wrong_way_log)}  |  Risk Score: {risk_score:.0f}/100 ({risk_level})",
        ]
        for line in lines:
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            y += 28

        y = 108
        cv2.putText(out, "Reasons:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        y += 22
        for reason in risk_reasons[:2]:
            cv2.putText(out, f"- {reason}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
            y += 20

        return out

    def process_video(
        self,
        source: str,
        max_frames: int | None = None,
        callback: Callable[[FrameAnalytics], None] | None = None,
    ) -> list[FrameAnalytics]:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
        self.speed_estimator.fps = self.fps

        results = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            analytics = self.process_frame(frame, frame_idx)
            results.append(analytics)
            if callback:
                callback(analytics)
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
        cap.release()
        return results

    def get_summary(self) -> dict:
        if not self.frame_history:
            return {}
        last = self.frame_history[-1]
        return {
            "total_frames": len(self.frame_history),
            "vehicle_counts": last.vehicle_counts,
            "total_vehicles": last.total_vehicles,
            "congestion_level": last.congestion_level,
            "congestion_score": last.congestion_score,
            "avg_speed_kmh": last.avg_speed_kmh,
            "helmet_violations": self.total_helmet_violations,
            "tripling_violations": len(self.tripling_violation_ids),
            "tripling_vehicle_ids": sorted(self.tripling_violation_ids),
            "tripling_log": self.tripling_log,
            "wrong_way_violations": self.wrong_way_log,
            "overspeeding_events": len(self.overspeed_log),
            "risk_level": last.risk_level,
            "risk_score": last.risk_score,
            "risk_reasons": last.risk_reasons,
            "cumulative_counts": last.cumulative_counts,
            "direction_stats": self.movement.get_direction_stats(),
        }
