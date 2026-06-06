"""CSV, JSON, and daily analytics report generation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import REPORTS_DIR


class ReportGenerator:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_frame_log_csv(self, frame_history: list, prefix: str = "traffic") -> Path:
        rows = []
        for fa in frame_history:
            if hasattr(fa, "__dict__"):
                d = fa.__dict__
            else:
                d = fa
            rows.append(
                {
                    "frame_idx": d.get("frame_idx"),
                    "timestamp": d.get("timestamp"),
                    "total_vehicles": d.get("total_vehicles"),
                    "cars": d.get("vehicle_counts", {}).get("cars", 0),
                    "bikes": d.get("vehicle_counts", {}).get("bikes", 0),
                    "buses": d.get("vehicle_counts", {}).get("buses", 0),
                    "trucks": d.get("vehicle_counts", {}).get("trucks", 0),
                    "congestion_level": d.get("congestion_level"),
                    "congestion_score": d.get("congestion_score"),
                    "avg_speed_kmh": d.get("avg_speed_kmh"),
                    "helmet_violations": d.get("helmet_violations"),
                    "tripling_violations": d.get("tripling_violations"),
                    "tripling_vehicle_ids": str(d.get("tripling_vehicle_ids", [])),
                    "risk_level": d.get("risk_level"),
                    "risk_score": d.get("risk_score"),
                }
            )
        df = pd.DataFrame(rows)
        path = self.output_dir / f"{prefix}_{self._timestamp()}.csv"
        df.to_csv(path, index=False)
        return path

    def export_summary_json(self, summary: dict, advisory: dict | None = None, prefix: str = "summary") -> Path:
        payload = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
        }
        if advisory:
            payload["advisory"] = advisory
        path = self.output_dir / f"{prefix}_{self._timestamp()}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    def export_daily_summary(self, summary: dict, advisory: dict | None = None) -> Path:
        daily = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "location": "Bengaluru Traffic Zone",
            "peak_congestion": summary.get("congestion_level"),
            "total_helmet_violations": summary.get("helmet_violations", 0),
            "tripling_violations": summary.get("tripling_violations", 0),
            "tripling_vehicle_ids": summary.get("tripling_vehicle_ids", []),
            "wrong_way_incidents": len(summary.get("wrong_way_violations", [])),
            "overspeeding_events": summary.get("overspeeding_events", 0),
            "avg_speed_kmh": summary.get("avg_speed_kmh"),
            "risk_level": summary.get("risk_level"),
            "direction_stats": summary.get("direction_stats", {}),
            "advisory_priority": advisory.get("priority") if advisory else "N/A",
            "top_recommendations": advisory.get("recommendations", [])[:3] if advisory else [],
        }
        path = self.output_dir / f"daily_analytics_{datetime.now().strftime('%Y%m%d')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(daily, f, indent=2)
        return path

    def export_violations_csv(
        self,
        wrong_way: list,
        overspeed: list,
        tripling: list | None = None,
        prefix: str = "violations",
    ) -> Path:
        rows = []
        for w in wrong_way:
            rows.append({"type": "wrong_way", **w})
        for o in overspeed:
            rows.append({"type": "overspeeding", **o})
        for t in tripling or []:
            rows.append({"type": "tripling", **t})
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["type"])
        path = self.output_dir / f"{prefix}_{self._timestamp()}.csv"
        df.to_csv(path, index=False)
        return path

    def export_tripling_json(self, tripling_log: list, prefix: str = "tripling") -> Path:
        payload = {
            "generated_at": datetime.now().isoformat(),
            "total_violations": len(tripling_log),
            "violations": tripling_log,
        }
        path = self.output_dir / f"{prefix}_{self._timestamp()}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return path
