"""Smart Traffic Risk Score calculator."""
from __future__ import annotations

from config.settings import RISK_WEIGHTS


class RiskScoreCalculator:
    def calculate(
        self,
        congestion_level: str,
        congestion_score: float,
        wrong_way_count: int,
        helmet_violations: int,
        overspeed_count: int,
        tripling_count: int = 0,
    ) -> tuple[str, float, list[str]]:
        reasons: list[str] = []
        ww_component = min(wrong_way_count * 15, 100)
        helmet_component = min(helmet_violations * 8, 100)
        overspeed_component = min(overspeed_count * 5, 100)
        tripling_component = min(tripling_count * 12, 100)

        if congestion_level == "High":
            reasons.append("High Congestion")
        elif congestion_level == "Medium":
            reasons.append("Moderate Congestion")
        if wrong_way_count:
            reasons.append(f"{wrong_way_count} Wrong-Way Vehicle{'s' if wrong_way_count > 1 else ''}")
        if helmet_violations:
            reasons.append(f"{helmet_violations} Helmet Violation{'s' if helmet_violations > 1 else ''}")
        if tripling_count:
            reasons.append(f"{tripling_count} Tripling Violation{'s' if tripling_count > 1 else ''}")
        if overspeed_count:
            reasons.append(f"{overspeed_count} Overspeeding Event{'s' if overspeed_count > 1 else ''}")
        if not reasons:
            reasons.append("Normal traffic conditions")

        score = (
            congestion_score * RISK_WEIGHTS["congestion"]
            + ww_component * RISK_WEIGHTS["wrong_way"]
            + helmet_component * RISK_WEIGHTS["helmet"]
            + overspeed_component * RISK_WEIGHTS["overspeed"]
            + tripling_component * RISK_WEIGHTS["tripling"]
        )
        score = min(100, round(score, 1))

        if score < 25:
            level = "Low Risk"
        elif score < 50:
            level = "Medium Risk"
        elif score < 75:
            level = "High Risk"
        else:
            level = "Critical Risk"

        return level, score, reasons
