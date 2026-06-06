"""CLI entry point for Smart Traffic Intelligence System."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analytics.advisory_engine import TrafficAdvisoryEngine
from analytics.reports import ReportGenerator
from config.settings import DEFAULT_DEMO_SEQUENCE, OUTPUTS_DIR
from tracking.traffic_pipeline import TrafficPipeline
from tracking.video_source import resolve_video_source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Traffic Intelligence System — Bengaluru Gridlock Challenge"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=f"visdrone:{DEFAULT_DEMO_SEQUENCE}",
        help="Video source: file path, visdrone:SEQ, detrac:SEQ, or 0 for webcam",
    )
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--flow-x", type=float, default=0.0)
    parser.add_argument("--flow-y", type=float, default=-1.0)
    parser.add_argument("--pixels-per-meter", type=float, default=8.0)
    parser.add_argument("--speed-limit", type=float, default=60.0)
    parser.add_argument("--output-video", type=str, default=None)
    parser.add_argument("--export", action="store_true", help="Export CSV/JSON reports")
    parser.add_argument("--train-helmet", action="store_true", help="Train helmet model")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict | None:
    if args.train_helmet:
        from training.train_helmet import train
        train()
        return None

    if args.dashboard:
        import subprocess
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
        return None

    source = resolve_video_source(args.source) if args.source != "0" else "0"
    pipeline = TrafficPipeline(
        flow_direction=(args.flow_x, args.flow_y),
        pixels_per_meter=args.pixels_per_meter,
        speed_limit=args.speed_limit,
    )
    advisory_engine = TrafficAdvisoryEngine()

    writer = None
    if args.output_video:
        import cv2
        cap = __import__("cv2").VideoCapture(int(source) if source == "0" else source)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_video, fourcc, fps, (w, h))

    def on_frame(analytics):
        if writer and analytics.annotated_frame is not None:
            writer.write(analytics.annotated_frame)
        if analytics.frame_idx % 30 == 0:
            ids = ", ".join(
                f"{td['class'].title()} ID:{td['track_id']}" for td in analytics.track_data[:4]
            )
            print(
                f"Frame {analytics.frame_idx}: "
                f"Vehicles={analytics.total_vehicles} "
                f"Congestion={analytics.congestion_level} "
                f"Risk={analytics.risk_score:.0f}/100 "
                f"IDs=[{ids}]"
            )

    print(f"Processing: {source}")
    results = pipeline.process_video(source, max_frames=args.max_frames, callback=on_frame)

    if writer:
        writer.release()
        print(f"Output video saved: {args.output_video}")

    summary = pipeline.get_summary()
    flow = pipeline.movement.get_flow_imbalance()
    advisory = advisory_engine.generate(summary, flow)

    print("\n=== TRAFFIC SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n=== AI ADVISORY ===")
    print(f"  Status: {advisory.status}")
    print(f"  Priority: {advisory.priority} (confidence: {advisory.confidence})")
    for rec in advisory.recommendations:
        print(f"  → {rec}")

    if args.export:
        reporter = ReportGenerator()
        csv_p = reporter.export_frame_log_csv(results)
        json_p = reporter.export_summary_json(summary, advisory.__dict__)
        daily_p = reporter.export_daily_summary(summary, advisory.__dict__)
        viol_p = reporter.export_violations_csv(
            pipeline.wrong_way_log, pipeline.overspeed_log, pipeline.tripling_log
        )
        tripling_p = reporter.export_tripling_json(pipeline.tripling_log)
        print(f"\nReports: {csv_p}, {json_p}, {daily_p}, {viol_p}, {tripling_p}")

    return {"summary": summary, "advisory": advisory, "results": results}


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
