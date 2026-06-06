"""Streamlit dashboard — Smart Traffic Intelligence System for Bengaluru."""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.advisory_engine import TrafficAdvisoryEngine
from analytics.reports import ReportGenerator
from config.settings import (
    DEFAULT_DEMO_SEQUENCE,
    OUTPUTS_DIR,
    PIXELS_PER_METER,
    SCREENSHOTS_DIR,
    SPEED_LIMIT_KMH,
)
from tracking.traffic_pipeline import TrafficPipeline
from tracking.video_source import list_visdrone_sequences, resolve_video_source
from tracking.video_utils import validate_video

st.set_page_config(
    page_title="Bengaluru Smart Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a5276; }
    .diag-box { background: #1e1e1e; color: #00ff88; font-family: monospace;
                font-size: 12px; padding: 10px; border-radius: 6px; overflow-y: auto;
                max-height: 200px; }
    [data-testid="stVerticalBlock"] { gap: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PRIORITY_CLASS = {
    "Critical": "advisory-critical",
    "High": "advisory-high",
    "Medium": "advisory-medium",
    "Low": "advisory-low",
}


def init_session():
    defaults = {
        "pipeline": None,
        "advisory_engine": TrafficAdvisoryEngine(),
        "frame_history": [],
        "running": False,
        "last_advisory": None,
        "last_summary": None,
        "video_source": None,
        "uploaded_video_path": None,
        "uploaded_video_name": None,
        "source_type": "Upload Video",
        "analysis_done": False,
        "last_annotated_frame": None,
        "diag_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def save_uploaded_video(uploaded_file) -> str | None:
    if uploaded_file is None:
        return st.session_state.get("uploaded_video_path")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    dest = OUTPUTS_DIR / f"upload_{safe_name}"
    if st.session_state.get("uploaded_video_name") != safe_name or not dest.exists():
        dest.write_bytes(uploaded_file.getbuffer())
    st.session_state.uploaded_video_path = str(dest)
    st.session_state.uploaded_video_name = safe_name
    return str(dest)


def resolve_source(source_type: str, source: str) -> str:
    if source == "0":
        return "0"
    if source_type == "VisDrone Demo":
        return resolve_video_source(source)
    return source


def resize_for_inference(frame: np.ndarray, max_width: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))


def render_advisory_panel(advisory):
    if advisory is None:
        st.info("Start analysis to receive AI recommendations.")
        return
    css = PRIORITY_CLASS.get(advisory.priority, "advisory-low")
    st.markdown(f'<div class="{css}" style="border-left:5px solid;padding:1rem;">', unsafe_allow_html=True)
    st.subheader("🤖 AI Recommendations Panel")
    c1, c2, c3 = st.columns(3)
    c1.metric("Traffic Status", advisory.status)
    c2.metric("Priority", advisory.priority)
    c3.metric("Confidence", f"{advisory.confidence * 100:.0f}%")
    for r in advisory.reasons:
        st.markdown(f"- {r}")
    for rec in advisory.recommendations:
        st.markdown(f"- ✅ {rec}")
    for alert in advisory.alerts:
        st.error(alert)
    st.markdown("</div>", unsafe_allow_html=True)


def run_analysis(
    video_path: str,
    pipeline: TrafficPipeline,
    limit: int,
    placeholders: dict,
):
    """Single-pass video processing — NO st.rerun() to avoid scroll glitch."""
    cap = cv2.VideoCapture(int(video_path) if video_path == "0" else video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
    limit = limit if limit > 0 else total_vid or 300

    out_video_path = str(OUTPUTS_DIR / "dashboard_result.webm")
    fourcc = cv2.VideoWriter_fourcc(*"vp80")
    writer = None

    frame_idx = 0
    t_start = time.perf_counter()
    hist = []
    diag_lines = []
    last_analytics = None

    progress = placeholders["progress"]
    status = placeholders["status"]
    video_ph = placeholders["video"]
    heatmap_ph = placeholders["heatmap"]

    while cap.isOpened() and frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            diag_lines.append(f"[frame {frame_idx}] VIDEO READ FAILED — end of stream")
            break

        frame = resize_for_inference(frame)
        t_inf = time.perf_counter()
        analytics = pipeline.process_frame(frame, frame_idx)
        inf_fps = 1.0 / max(time.perf_counter() - t_inf, 0.001)

        st.session_state.frame_history.append(analytics)
        hist.append(analytics)
        last_analytics = analytics

        dbg = getattr(analytics, "debug", {})
        line = (
            f"F{frame_idx}: raw={dbg.get('raw_detections',0)} veh={dbg.get('vehicle_detections',0)} "
            f"tracked={dbg.get('vehicles_with_track_id',0)} infer={dbg.get('inference_ms',0)}ms "
            f"fps={inf_fps:.1f} classes={dbg.get('classes_detected',[])}"
        )
        diag_lines.append(line)
        st.session_state.diag_log = diag_lines[-30:]

        elapsed = time.perf_counter() - t_start
        proc_fps = (frame_idx + 1) / max(elapsed, 0.001)

        status.markdown(
            f"🔴 **LIVE** Frame **{frame_idx + 1}/{limit}** | "
            f"Vehicles: **{analytics.total_vehicles}** | "
            f"Inference: **{inf_fps:.1f} FPS** | "
            f"Overall: **{proc_fps:.1f} FPS** | "
            f"Model: ✅ loaded"
        )

        placeholders["total"].metric("Vehicles (frame)", analytics.total_vehicles)
        placeholders["speed"].metric("Avg Speed", f"{analytics.avg_speed_kmh} km/h")
        placeholders["cong"].metric("Congestion", analytics.congestion_level)
        placeholders["risk"].metric("Risk", f"{analytics.risk_score:.0f}/100", analytics.risk_level)

        cc = analytics.cumulative_counts
        placeholders["counts"].markdown(
            f"🚗 {cc.get('cars',0)} | 🏍️ {cc.get('bikes',0)} | "
            f"🚌 {cc.get('buses',0)} | 🚛 {cc.get('trucks',0)} | "
            f"⛑️ {analytics.helmet_violations} | ↩️ {len(pipeline.wrong_way_log)}"
        )

        if analytics.annotated_frame is not None:
            if writer is None:
                h, w = analytics.annotated_frame.shape[:2]
                writer = cv2.VideoWriter(out_video_path, fourcc, fps_vid, (w, h))
            
            writer.write(analytics.annotated_frame)
            hm = pipeline.heatmap.render(analytics.annotated_frame)
            # Only update heatmap and small metrics to show it's alive, skip updating full video frame by frame
            if frame_idx % 5 == 0:
                heatmap_ph.image(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            st.session_state.last_annotated_frame = analytics.annotated_frame

        if frame_idx % 5 == 0:
            progress.progress(min((frame_idx + 1) / limit, 1.0), text=f"Frame {frame_idx + 1}/{limit}")

        if frame_idx % 10 == 0 and len(hist) > 1:
            fig = px.line(
                x=[a.frame_idx for a in hist],
                y=[a.congestion_score for a in hist],
                labels={"x": "Frame", "y": "Congestion"},
            )
            fig.update_traces(line_color="#e74c3c")
            placeholders["chart"].plotly_chart(fig, use_container_width=True)

        summary = pipeline.get_summary()
        summary["density"] = analytics.total_vehicles
        flow = pipeline.movement.get_flow_imbalance()
        st.session_state.last_advisory = st.session_state.advisory_engine.generate(summary, flow)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        
    progress.progress(1.0, text="Complete")
    
    # Show the final compiled video
    if Path(out_video_path).exists():
        video_ph.video(out_video_path)
        
    return last_analytics, frame_idx, diag_lines


def main():
    init_session()

    st.markdown(
        '<p class="main-header">🚦 Smart Traffic Intelligence System — Bengaluru</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        source_options = ["Upload Video", "VisDrone Demo", "Webcam", "Custom Path"]
        source_type = st.radio("Video Source", source_options, index=0)
        source = ""

        uploaded = st.file_uploader(
            "Upload video", type=["mp4", "avi", "mov", "mkv", "webm"], key="video_upload"
        )
        if uploaded:
            source = save_uploaded_video(uploaded) or ""
            ok, msg = validate_video(source)
            if ok:
                st.success(f"✅ {msg}")
            else:
                st.error(msg)
        elif st.session_state.get("uploaded_video_path"):
            source = st.session_state.uploaded_video_path
            st.caption(f"📁 {st.session_state.uploaded_video_name}")
        elif source_type == "VisDrone Demo":
            seq = st.selectbox("Sequence", list_visdrone_sequences() or [DEFAULT_DEMO_SEQUENCE])
            source = f"visdrone:{seq}"
        elif source_type == "Webcam":
            source = "0"
        else:
            source = st.text_input("Path", placeholder=r"D:\videos\traffic.mp4")

        flow_dx = st.slider("Flow X", -1.0, 1.0, 0.0, 0.1)
        flow_dy = st.slider("Flow Y", -1.0, 1.0, -1.0, 0.1)
        max_frames = st.number_input("Max Frames", 10, 500, 60)
        show_diag = st.checkbox("Show Diagnostics", value=True)

        start_btn = st.button("▶️ Start Analysis", type="primary", use_container_width=True)
        stop_btn = st.button("⏹️ Stop", use_container_width=True)
        export_btn = st.button("📥 Export Reports", use_container_width=True)

    # ── Main layout ──────────────────────────────────────────────────
    st.subheader("🎥 Live Analysis")
    status_ph = st.empty()
    progress_ph = st.progress(0, text="Ready")

    col_feed, col_metrics = st.columns([3, 1])
    with col_feed:
        video_ph = st.empty()
        heatmap_ph = st.empty()
    with col_metrics:
        st.markdown("**📊 Metrics**")
        total_ph = st.empty()
        speed_ph = st.empty()
        cong_ph = st.empty()
        risk_ph = st.empty()
        counts_ph = st.empty()

    chart_ph = st.empty()
    diag_ph = st.empty()

    placeholders = {
        "status": status_ph,
        "progress": progress_ph,
        "video": video_ph,
        "heatmap": heatmap_ph,
        "total": total_ph,
        "speed": speed_ph,
        "cong": cong_ph,
        "risk": risk_ph,
        "counts": counts_ph,
        "chart": chart_ph,
    }

    st.divider()
    render_advisory_panel(st.session_state.last_advisory)

    # ── Start analysis (single run, no rerun loop) ───────────────────
    if start_btn:
        if not source:
            st.error("Upload a video first.")
        else:
            try:
                if source_type == "VisDrone Demo":
                    resolved = resolve_source(source_type, source)
                elif source == "0":
                    resolved = "0"
                else:
                    resolved = source
                ok, msg = validate_video(resolved) if resolved != "0" else (True, "webcam")
                if not ok:
                    st.error(msg)
                else:
                    st.session_state.frame_history = []
                    st.session_state.diag_log = []
                    st.session_state.analysis_done = False

                    with st.spinner("Loading YOLO11s model (~30s first time)..."):
                        pipeline = TrafficPipeline(
                            flow_direction=(flow_dx, flow_dy),
                            pixels_per_meter=PIXELS_PER_METER,
                            speed_limit=float(SPEED_LIMIT_KMH),
                        )
                    st.session_state.pipeline = pipeline

                    status_ph.info(f"Processing: {msg}")
                    try:
                        last, n_frames, diag = run_analysis(
                            resolved, pipeline, int(max_frames), placeholders
                        )
                        st.session_state.analysis_done = True

                        cum_vehicles = sum(last.cumulative_counts.values()) if last else 0
                        if last and cum_vehicles > 0:
                            st.success(
                                f"✅ Done — {n_frames} frames | "
                                f"Total vehicles seen: {cum_vehicles} | "
                                f"Classes: {getattr(last, 'debug', {}).get('classes_detected', [])}"
                            )
                        elif last:
                            st.warning(
                                f"⚠️ {n_frames} frames processed but 0 vehicles detected. "
                                "Check diagnostics below."
                            )
                        else:
                            st.error("No frames could be read from video.")

                        if last and last.annotated_frame is not None:
                            ss = SCREENSHOTS_DIR / f"dash_{datetime.now().strftime('%H%M%S')}.jpg"
                            SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(ss), last.annotated_frame)

                    except Exception as e:
                        import traceback
                        with open("dashboard_error.txt", "w", encoding="utf-8") as f:
                            f.write(traceback.format_exc())
                        st.error("Analysis failed:")
                        st.code(traceback.format_exc())

            except Exception:
                st.error("Setup failed:")
                st.code(traceback.format_exc())

    elif st.session_state.get("last_annotated_frame") is not None:
        video_ph.image(
            cv2.cvtColor(st.session_state.last_annotated_frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            caption="Last analyzed frame",
            use_container_width=True,
        )

    # ── Diagnostics panel ────────────────────────────────────────────
    if show_diag and st.session_state.get("diag_log"):
        with st.expander("🔧 Pipeline Diagnostics", expanded=True):
            st.markdown(
                f'<div class="diag-box">{"<br>".join(st.session_state.diag_log)}</div>',
                unsafe_allow_html=True,
            )
            if st.session_state.pipeline:
                p = st.session_state.pipeline
                st.markdown(
                    f"**Model:** `{p.diagnostics.model_path}` | "
                    f"**Status:** {p.diagnostics.model_status}"
                )

    if export_btn and st.session_state.pipeline and st.session_state.frame_history:
        reporter = ReportGenerator()
        summary = st.session_state.pipeline.get_summary()
        adv = st.session_state.last_advisory
        reporter.export_frame_log_csv(st.session_state.frame_history)
        reporter.export_summary_json(summary, adv.__dict__ if adv else None)
        st.success("Reports exported to reports/")

    if not start_btn and not st.session_state.get("analysis_done"):
        st.info("👆 Upload video → click **Start Analysis**")


if __name__ == "__main__":
    main()
