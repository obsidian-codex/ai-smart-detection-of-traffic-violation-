"""AI Traffic Advisory Engine — rule-based decision support for authorities."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Advisory:
    status: str
    reasons: list[str]
    problems: list[str]
    recommendations: list[str]
    priority: str
    confidence: float
    alerts: list[str] = field(default_factory=list)


class TrafficAdvisoryEngine:
    """Converts raw analytics into actionable traffic management recommendations."""

    PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]

    def __init__(self):
        self.history: list[dict] = []
        self.baseline_speed: float | None = None

    def _higher_priority(self, a: str, b: str) -> str:
        order = self.PRIORITY_ORDER
        return a if order.index(a) >= order.index(b) else b

    def _update_baseline(self, avg_speed: float):
        if self.baseline_speed is None:
            self.baseline_speed = avg_speed
        else:
            self.baseline_speed = 0.95 * self.baseline_speed + 0.05 * avg_speed

    def _speed_drop_pct(self, avg_speed: float) -> float:
        if self.baseline_speed and self.baseline_speed > 0:
            return max(0, (1 - avg_speed / self.baseline_speed) * 100)
        return 0.0

    def _density_trend(self) -> float:
        if len(self.history) < 5:
            return 0.0
        recent = [h.get("density", 0) for h in self.history[-5:]]
        older = [h.get("density", 0) for h in self.history[-10:-5]] if len(self.history) >= 10 else recent
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older) if older else avg_recent
        if avg_older == 0:
            return 0.0
        return ((avg_recent - avg_older) / avg_older) * 100

    def generate(self, analytics: dict[str, Any], flow_stats: dict | None = None) -> Advisory:
        congestion = analytics.get("congestion_level", "Low")
        cong_score = analytics.get("congestion_score", 0)
        vehicle_count = analytics.get("total_vehicles", 0)
        avg_speed = analytics.get("avg_speed_kmh", 0)
        density = analytics.get("density", vehicle_count)
        wrong_way = analytics.get("wrong_way_violations", [])
        helmet_v = analytics.get("helmet_violations", 0)
        tripling_v = analytics.get("tripling_violations", 0)
        tripling_log = analytics.get("tripling_log", [])
        tripling_ids = analytics.get("tripling_vehicle_ids", [])
        overspeed = analytics.get("overspeeding_events", 0)
        risk_level = analytics.get("risk_level", "Low Risk")

        self._update_baseline(avg_speed)
        speed_drop = self._speed_drop_pct(avg_speed)
        density_trend = self._density_trend()

        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "density": density,
                "avg_speed": avg_speed,
                "congestion": congestion,
            }
        )
        if len(self.history) > 100:
            self.history = self.history[-100:]

        reasons: list[str] = []
        problems: list[str] = []
        recommendations: list[str] = []
        alerts: list[str] = []
        priority = "Low"
        confidence = 0.75

        # Congestion analysis
        if congestion == "High" or cong_score > 65:
            status = "HIGH CONGESTION"
            reasons.append(f"{vehicle_count} vehicles detected in frame")
            if avg_speed < 15:
                reasons.append(f"Average speed below {avg_speed:.0f} km/h")
            if density_trend > 20:
                reasons.append(f"Traffic density increased by {density_trend:.0f}%")
            problems.append("Severe gridlock conditions detected")
            green_extend = 20 if cong_score < 80 else 30
            recommendations.append(
                f"Extend green signal timing by {green_extend} seconds and redirect traffic through alternate corridor"
            )
            if density_trend > 30:
                recommendations.append(
                    "Traffic density rising rapidly — suggest diversion through alternate routes (ORR / Inner Ring Road)"
                )
            alerts.append("CONGESTION ALERT: Immediate signal optimization required")
            priority = "Critical" if cong_score > 80 else "High"
            confidence = 0.92
        elif congestion == "Medium":
            status = "MODERATE CONGESTION"
            reasons.append(f"{vehicle_count} vehicles with average speed {avg_speed:.0f} km/h")
            problems.append("Traffic flow below optimal levels")
            recommendations.append("Monitor intersection closely; prepare contingency signal plan")
            priority = "Medium"
            confidence = 0.82
        else:
            status = "NORMAL TRAFFIC"
            reasons.append(f"Traffic flowing normally with {vehicle_count} vehicles")
            recommendations.append("Continue standard signal timing; no intervention required")
            priority = "Low"
            confidence = 0.88

        # Speed drop
        if speed_drop > 25:
            reasons.append(f"Average speed dropped by {speed_drop:.0f}% compared to normal conditions")
            problems.append("Significant speed reduction indicating bottleneck")
            recommendations.append("Deploy traffic personnel to identify obstruction cause")
            priority = self._higher_priority(priority, "High")
            confidence = min(confidence + 0.05, 0.95)

        # Wrong-way
        if wrong_way:
            latest = wrong_way[-1] if isinstance(wrong_way, list) else wrong_way
            tid = latest.get("track_id", "?") if isinstance(latest, dict) else "?"
            problems.append(f"Wrong-way vehicle detected (ID: {tid})")
            recommendations.append(
                f"Wrong-way vehicle ID {tid} detected — immediate intervention recommended"
            )
            alerts.append(f"WRONG-WAY ALERT: Vehicle ID {tid} moving against traffic flow")
            priority = "Critical"
            confidence = 0.96

        # Helmet violations
        if helmet_v >= 3:
            problems.append(f"Helmet violation hotspot: {helmet_v} violations detected")
            recommendations.append(
                "Helmet violation hotspot identified — recommend targeted enforcement at this junction"
            )
            alerts.append("SAFETY ALERT: Elevated helmet non-compliance")
            if priority != "Critical":
                priority = "High"

        # Tripling violations
        if tripling_log or tripling_v:
            latest_t = tripling_log[-1] if tripling_log else {}
            tid = latest_t.get("track_id", tripling_ids[-1] if tripling_ids else "?")
            riders = latest_t.get("rider_count", 3)
            problems.append(f"Tripling violation detected on motorcycle ID {tid} ({riders} riders)")
            recommendations.append(
                f"Tripling violation detected on motorcycle ID {tid}. "
                "Recommend immediate traffic police intervention"
            )
            alerts.append(f"TRIPLING ALERT: Motorcycle ID {tid} carrying {riders} persons")
            priority = self._higher_priority(priority, "High")
            confidence = max(confidence, 0.94)

        if tripling_v >= 3 or (helmet_v >= 3 and tripling_v >= 1):
            problems.append("High concentration of motorcycle safety violations observed")
            recommendations.append(
                "Deploy motorcycle safety checkpoint — multiple tripling and helmet violations detected"
            )
            alerts.append("MOTORCYCLE SAFETY ALERT: High violation concentration in zone")
            priority = self._higher_priority(priority, "Critical")

        # Overspeeding
        if overspeed >= 5:
            problems.append(f"{overspeed} overspeeding events recorded")
            recommendations.append(
                "Overspeeding frequency increased in Zone A — recommend speed monitoring and camera enforcement"
            )
            alerts.append("SPEED ALERT: Multiple overspeeding vehicles detected")

        # Flow imbalance
        if flow_stats and flow_stats.get("imbalanced"):
            dominant = flow_stats.get("dominant", "unknown")
            ratio = flow_stats.get("ratio", 1)
            problems.append(f"Traffic flow imbalance detected (ratio {ratio}:1)")
            recommendations.append(
                f"Traffic flow imbalance between lanes — dominant direction: {dominant}. "
                "Consider asymmetric signal phasing"
            )

        # Risk-based override
        if "Critical" in risk_level:
            priority = "Critical"
            alerts.append(f"SYSTEM RISK: {risk_level} — score {analytics.get('risk_score', 0)}")

        if not problems:
            problems.append("No critical issues detected")

        return Advisory(
            status=status,
            reasons=reasons,
            problems=problems,
            recommendations=recommendations,
            priority=priority,
            confidence=round(confidence, 2),
            alerts=alerts,
        )
