"""One-command hackathon demo — full end-to-end pipeline."""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analytics.advisory_engine import TrafficAdvisoryEngine
from analytics.reports import ReportGenerator
from config.settings import OUTPUTS_DIR, SCREENSHOTS_DIR
from tracking.traffic_pipeline import TrafficPipeline
from tracking.video_source import resolve_video_source

# Best VisDrone sequences with visible traffic
DEMO_SEQUENCE = "uav0000071_03240_v"
MAX_FRAMES = 200


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BENGALURU SMART TRAFFIC INTELLIGENCE — LIVE DEMO")
    print("=" * 60)

    source = resolve_video_source(f"visdrone:{DEMO_SEQUENCE}")
    print(f"\n[1/6] Video source: {source}")

    pipeline = TrafficPipeline(flow_direction=(0, -1))
    advisory_engine = TrafficAdvisoryEngine()

    out_video = str(OUTPUTS_DIR / "hackathon_demo.mp4")
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t0 = time.time()
    frame_times = []

    print("[2/6] Running YOLO11s + ByteTrack inference...")

    def on_frame(a):
        if a.annotated_frame is not None:
            writer.write(a.annotated_frame)
        frame_times.append(time.time())
        if a.frame_idx % 25 == 0:
            fps_now = 25 / (frame_times[-1] - frame_times[-26]) if len(frame_times) > 26 else 0
            ids = [f"{td['class'][:3].upper()} ID:{td['track_id']}" for td in a.track_data[:5]]
            print(
                f"  Frame {a.frame_idx:4d} | FPS~{fps_now:.1f} | "
                f"Vehicles={a.total_vehicles} | Congestion={a.congestion_level} | "
                f"Risk={a.risk_score:.0f}/100 | IDs: {', '.join(ids)}"
            )

    results = pipeline.process_video(source, max_frames=MAX_FRAMES, callback=on_frame)
    writer.release()
    elapsed = time.time() - t0
    avg_fps = len(results) / elapsed if elapsed > 0 else 0

    print(f"\n[3/6] Processed {len(results)} frames in {elapsed:.1f}s (avg {avg_fps:.1f} FPS)")
    print(f"      Demo video saved: {out_video}")

    summary = pipeline.get_summary()
    summary["cumulative_counts"] = results[-1].cumulative_counts if results else {}
    summary["risk_reasons"] = results[-1].risk_reasons if results else []
    flow = pipeline.movement.get_flow_imbalance()
    advisory = advisory_engine.generate(summary, flow)

    print("\n[4/6] VEHICLE COUNTS (session unique IDs)")
    cc = summary.get("cumulative_counts", {})
    print(f"  Cars: {cc.get('cars', 0)}  Bikes: {cc.get('bikes', 0)}  "
          f"Buses: {cc.get('buses', 0)}  Trucks: {cc.get('trucks', 0)}")

    print("\n[5/6] RISK SCORE")
    print(f"  Risk Score: {summary.get('risk_score', 0):.0f}/100 ({summary.get('risk_level')})")
    print("  Reasons:")
    for r in summary.get("risk_reasons", []):
        print(f"    - {r}")
    print(f"  Helmet Violations: {summary.get('helmet_violations', 0)}")
    print(f"  Tripling Violations: {summary.get('tripling_violations', 0)}")
    print(f"  Tripling Vehicle IDs: {summary.get('tripling_vehicle_ids', [])}")
    print(f"  Wrong-Way Vehicles: {len(summary.get('wrong_way_violations', []))}")

    print("\n[6/6] AI ADVISORY")
    print(f"  Status: {advisory.status}")
    print(f"  Priority: {advisory.priority} (confidence: {advisory.confidence})")
    for rec in advisory.recommendations:
        print(f"  → {rec}")

    if results and results[-1].annotated_frame is not None:
        ss = SCREENSHOTS_DIR / f"demo_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(str(ss), results[-1].annotated_frame)
        print(f"\n  Screenshot: {ss}")

    reporter = ReportGenerator()
    reporter.export_frame_log_csv(results)
    reporter.export_summary_json(summary, advisory.__dict__)
    reporter.export_daily_summary(summary, advisory.__dict__)
    reporter.export_violations_csv(
        pipeline.wrong_way_log, pipeline.overspeed_log, pipeline.tripling_log
    )
    reporter.export_tripling_json(pipeline.tripling_log)
    print("\n  Reports exported to reports/")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE — Launch dashboard:")
    print("  streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
