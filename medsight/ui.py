from __future__ import annotations

import base64
import io
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from medsight.analytics import build_track_dataframe
from medsight.config import (
    APP_DESCRIPTION,
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEMPORAL_WINDOW,
    FRAGMENT_INTERVAL,
    IMAGE_UPLOAD_DIR,
    LOG_LIMIT,
    MAX_SNAPSHOTS,
    PAGE_DASHBOARD,
    PAGE_HOME,
    SNAPSHOT_INTERVAL_FRAMES,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
    VIDEO_UPLOAD_DIR,
)
from medsight.detection import LesionDetector
from medsight.optimization import ModelOptimizer, OptimizationResult
from medsight.pipeline import MedSightPipeline, PipelineFrameResult
from medsight.video import VideoStream, decode_uploaded_image, save_uploaded_file


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_app_shell() -> None:
    _initialize_state()
    _inject_styles()

    if st.session_state.current_page == PAGE_HOME:
        _render_home_page()
    else:
        _render_dashboard_page()


def _initialize_state() -> None:
    defaults = {
        "current_page": PAGE_HOME,
        "source_mode": "Video",
        "run_stream": False,
        "stream_object": None,
        "stream_signature": None,
        "uploaded_video_path": None,
        "uploaded_image_path": None,
        "latest_result": None,
        "latest_image_signature": None,
        "frame_count": 0,
        "total_frames": 0,
        "logs": deque(maxlen=LOG_LIMIT),
        "snapshots": [],
        "stats": {
            "fps": 0.0,
            "pipeline_ms": 0.0,
            "inference_ms": 0.0,
            "raw_detections": 0,
            "confirmed_detections": 0,
            "active_lesions": 0,
            "total_lesions": 0,
            "average_confidence": 0.0,
            "detection_frequency": 0.0,
        },
        "pipeline_settings": None,
        "pipeline": None,
        "optimization_result": None,
        "source_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        /* ── Light Healthcare Theme ─────────────────────────────────────── */
        :root {
            --ms-bg:           #f0f7ff;
            --ms-panel:        rgba(255, 255, 255, 0.92);
            --ms-panel-strong: rgba(255, 255, 255, 0.98);
            --ms-border:       rgba(99, 179, 237, 0.28);
            --ms-text:         #1a2f4a;
            --ms-muted:        #5a7a9a;
            --ms-accent:       #0ea87a;
            --ms-accent-2:     #2b7de9;
            --ms-danger:       #e53e3e;
            --ms-warning:      #d97706;
            --ms-shadow:       0 8px 32px rgba(43, 125, 233, 0.10);
            --ms-radius:       20px;
        }

        * {
            font-family: 'Space Grotesk', sans-serif !important;
        }

        /* ── Main canvas ──────────────────────────────────────────────── */
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 5%  8%,  rgba(14, 168, 122, 0.10), transparent 30%),
                radial-gradient(circle at 95% 5%,  rgba(43, 125, 233, 0.12), transparent 28%),
                radial-gradient(circle at 50% 95%, rgba(255, 214, 102, 0.08), transparent 30%),
                linear-gradient(160deg, #e8f4ff 0%, #f5fbff 45%, #edfdf6 100%);
            color: var(--ms-text);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        /* ── Sidebar ──────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f0f8ff 100%);
            border-right: 1px solid rgba(43, 125, 233, 0.15);
            box-shadow: 2px 0 16px rgba(43, 125, 233, 0.06);
        }

        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"] {
            position: fixed;
            top: 0.75rem;
            left: 0.75rem;
            z-index: 9999;
        }

        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapsedControl"] button {
            position: relative;
            width: 2.45rem;
            height: 2.45rem;
            border-radius: 999px;
            border: 1px solid rgba(43, 125, 233, 0.22);
            background: #ffffff;
            color: transparent !important;
            min-height: 2.45rem;
            padding: 0;
            box-shadow: 0 2px 8px rgba(43, 125, 233, 0.12);
        }

        [data-testid="stSidebarCollapseButton"] button *,
        [data-testid="stSidebarCollapsedControl"] button * {
            color: transparent !important;
            font-size: 0 !important;
            line-height: 0 !important;
        }

        [data-testid="stSidebarCollapseButton"] button::after,
        [data-testid="stSidebarCollapsedControl"] button::after {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #2b7de9;
            font-size: 1.15rem;
            line-height: 1;
        }

        [data-testid="stSidebarCollapseButton"] button::after {
            content: "×";
        }

        [data-testid="stSidebarCollapsedControl"] button::after {
            content: "≡";
        }

        .block-container {
            max-width: 1400px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        /* ── Card surfaces ────────────────────────────────────────────── */
        .ms-home-shell,
        .ms-panel,
        .ms-feed-shell,
        .ms-stat-card,
        .ms-log-shell,
        .ms-snapshot-card {
            background: var(--ms-panel);
            border: 1px solid var(--ms-border);
            border-radius: var(--ms-radius);
            box-shadow: var(--ms-shadow);
        }

        /* ── Hero / Home shell ────────────────────────────────────────── */
        .ms-home-shell {
            padding: 3rem;
            margin-bottom: 1.25rem;
            background:
                linear-gradient(135deg, rgba(14, 168, 122, 0.07), transparent 40%),
                linear-gradient(225deg, rgba(43, 125, 233, 0.07), transparent 40%),
                var(--ms-panel-strong);
            border: 1px solid rgba(43, 125, 233, 0.18);
            box-shadow: 0 12px 40px rgba(43, 125, 233, 0.10);
        }

        .ms-kicker {
            color: var(--ms-accent);
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.72rem;
            font-weight: 700;
        }

        .ms-title {
            font-size: 4rem;
            line-height: 1;
            margin: 0.8rem 0 1rem;
            font-weight: 700;
            background: linear-gradient(120deg, #1a3a6e 0%, #2b7de9 50%, #0ea87a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .ms-subtitle {
            max-width: 760px;
            color: var(--ms-muted);
            font-size: 1.08rem;
            line-height: 1.8;
        }

        .ms-badge-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-top: 1.5rem;
        }

        .ms-badge {
            border: 1px solid rgba(43, 125, 233, 0.25);
            border-radius: 999px;
            padding: 0.45rem 0.9rem;
            color: #1a4a8a;
            background: rgba(43, 125, 233, 0.08);
            font-size: 0.82rem;
            font-weight: 500;
        }

        /* ── Generic panel ────────────────────────────────────────────── */
        .ms-panel {
            padding: 1rem 1rem 1.15rem;
        }

        .ms-panel-title {
            color: var(--ms-accent-2);
            text-transform: uppercase;
            letter-spacing: 0.15em;
            font-size: 0.72rem;
            margin-bottom: 0.9rem;
            font-weight: 700;
        }

        /* ── Video / image feed — fixed size, never distorts ─────────── */
        .ms-feed-shell {
            /* FIXED height — must match .ms-feed-idle height exactly */
            height: 480px;
            overflow: hidden;
            padding: 0.5rem;
            background: #f8fbff;
            border: 1px solid rgba(43, 125, 233, 0.15);
            border-radius: var(--ms-radius);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* The <img> is now embedded directly in the HTML — plain selector */
        .ms-feed-shell img {
            max-width:  100%;
            max-height: 460px;
            width:  auto;
            height: auto;
            object-fit: contain;
            border-radius: 10px;
            display: block;
            margin: auto;
        }

        /* ── Idle / empty placeholder — SAME height as ms-feed-shell ─── */
        .ms-feed-idle {
            height: 480px;          /* keep in sync with ms-feed-shell */
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            gap: 1.2rem;
            border-radius: var(--ms-radius);
            background:
                repeating-linear-gradient(
                    -45deg,
                    rgba(43, 125, 233, 0.03),
                    rgba(43, 125, 233, 0.03) 10px,
                    transparent 10px,
                    transparent 20px
                ),
                linear-gradient(160deg, #f0f8ff 0%, #edfdf6 100%);
            border: 2px dashed rgba(43, 125, 233, 0.22);
        }

        .ms-feed-idle-icon {
            font-size: 3.2rem;
            line-height: 1;
            opacity: 0.55;
        }

        .ms-feed-idle-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--ms-accent-2);
        }

        .ms-feed-idle-hint {
            font-size: 0.83rem;
            color: var(--ms-muted);
            max-width: 340px;
            line-height: 1.65;
        }

        /* ── Feed outer wrapper ─────────────────────────────────────────── */
        .ms-feed-outer {
            margin-bottom: 0.5rem;
        }

        /* ── Metrics grid ─────────────────────────────────────────────── */
        .ms-metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
        }

        .ms-stat-card {
            padding: 0.9rem 1rem;
            background: linear-gradient(135deg, #f0f7ff, #ffffff);
            border: 1px solid rgba(43, 125, 233, 0.14);
        }

        .ms-stat-label {
            color: var(--ms-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        .ms-stat-value {
            color: var(--ms-text);
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }

        /* ── AI Logs ──────────────────────────────────────────────────── */
        .ms-log-shell {
            padding: 1rem;
            background: #f5faff;
            border: 1px solid rgba(43, 125, 233, 0.12);
        }

        .ms-log-line {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.76rem;
            margin: 0 0 0.45rem;
            color: #2b4a6e;
        }

        .ms-info   { color: #2b7de9; }
        .ms-detect { color: #0ea87a; }
        .ms-track  { color: #7c5cbf; }
        .ms-warn   { color: #d97706; }

        .ms-small-note {
            color: var(--ms-muted);
            font-size: 0.83rem;
            line-height: 1.7;
        }

        /* ── Buttons ──────────────────────────────────────────────────── */
        .stButton > button {
            border-radius: 14px;
            height: 2.9rem;
            font-weight: 700;
            border: 1px solid rgba(43, 125, 233, 0.22);
            background: linear-gradient(135deg, #0ea87a, #2b7de9);
            color: #ffffff;
            box-shadow: 0 4px 14px rgba(43, 125, 233, 0.20);
            transition: box-shadow 0.18s ease, transform 0.14s ease;
        }

        .stButton > button:hover {
            border-color: rgba(43, 125, 233, 0.40);
            color: #ffffff;
            box-shadow: 0 6px 20px rgba(43, 125, 233, 0.28);
            transform: translateY(-1px);
        }

        .stMetric {
            background: transparent;
        }

        /* ── Streamlit native element overrides for light mode ─────────── */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--ms-text);
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #1a2f4a !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #1a2f4a !important;
        }

        [data-testid="metric-container"] label {
            color: var(--ms-muted) !important;
        }

        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: var(--ms-text) !important;
        }

        /* ── Snapshot card ─────────────────────────────────────────────── */
        .ms-snapshot-card {
            background: linear-gradient(135deg, #f0f8ff, #edfdf6);
            border: 1px solid rgba(14, 168, 122, 0.20);
        }

        @media (max-width: 900px) {
            .ms-title {
                font-size: 2.6rem;
            }
            .ms-home-shell {
                padding: 2rem 1.35rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_home_page() -> None:
    st.markdown(
        f"""
        <section class="ms-home-shell">
            <div class="ms-kicker">Clinical AI Interface • Detection • Tracking • Optimization</div>
            <h1 class="ms-title">{APP_TITLE}</h1>
            <p class="ms-subtitle">{APP_SUBTITLE}</p>
            <p class="ms-subtitle">{APP_DESCRIPTION}</p>
            <div class="ms-badge-row">
                <span class="ms-badge">YOLOv8 detection</span>
                <span class="ms-badge">ByteTrack / BoT-SORT tracking</span>
                <span class="ms-badge">Temporal consistency filter</span>
                <span class="ms-badge">ONNX export + latency benchmark</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    action_cols = st.columns(3, gap="large")
    if action_cols[0].button("Upload Image", use_container_width=True):
        st.session_state.source_mode = "Image"
        st.session_state.current_page = PAGE_DASHBOARD
        st.rerun()
    if action_cols[1].button("Upload Video", use_container_width=True):
        st.session_state.source_mode = "Video"
        st.session_state.current_page = PAGE_DASHBOARD
        st.rerun()
    if action_cols[2].button("Start Analysis", use_container_width=True):
        st.session_state.source_mode = "Webcam"
        st.session_state.current_page = PAGE_DASHBOARD
        st.rerun()

    feature_cols = st.columns(4, gap="medium")
    features = [
        ("Structured pipeline", "video → frames → preprocessing → inference → tracking → rendering"),
        ("Temporal validation", "Confirms lesion candidates only after persistence across N frames"),
        ("Clinical telemetry", "Tracks lesion counts, durations, confidence trends, and frequency"),
        ("Deployment readiness", "Supports webcam, uploads, ONNX export, and latency benchmarking"),
    ]
    for col, (title, text) in zip(feature_cols, features):
        with col:
            st.markdown(
                f"""
                <div class="ms-panel" style="min-height: 190px;">
                    <div class="ms-panel-title">{title}</div>
                    <div class="ms-small-note">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_dashboard_page() -> None:
    detector, load_error = _load_detector_safe(DEFAULT_MODEL_PATH)
    controls = _render_sidebar(detector is not None)
    pipeline = _get_pipeline(detector, controls)

    st.markdown(
        """
        <div class="ms-panel" style="margin-bottom: 1rem;">
            <div class="ms-panel-title">Main Screen</div>
            <div class="ms-small-note">
                Real-time spatiotemporal lesion candidate analysis with live inference telemetry,
                persistent track IDs, temporal filtering, analytics, logs, and snapshots.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if load_error:
        st.error(f"Model load failed: {load_error}")
        return

    left_col, right_col = st.columns([2.4, 1], gap="large")
    with left_col:
        _render_left_panel(pipeline, controls)
    with right_col:
        _render_right_panel(pipeline)


def _render_sidebar(model_ready: bool) -> dict:
    with st.sidebar:
        st.markdown("## MedSight Controls")
        if st.button("Return Home", use_container_width=True):
            _stop_stream()
            st.session_state.current_page = PAGE_HOME
            st.rerun()

        source_mode = st.radio(
            "Source",
            ("Image", "Video", "Webcam"),
            index=("Image", "Video", "Webcam").index(st.session_state.source_mode),
        )
        confidence = st.slider("Confidence threshold", 0.10, 0.95, DEFAULT_CONFIDENCE, 0.05)
        temporal_window = st.slider("Temporal persistence (frames)", 1, 8, DEFAULT_TEMPORAL_WINDOW, 1)
        tracker_name = st.selectbox("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        enable_fp16 = st.checkbox("Enable FP16 when supported", value=True)
        loop_video = st.checkbox("Loop uploaded video", value=True)

        uploaded_image_path = None
        uploaded_video_path = None

        if source_mode == "Image":
            uploaded_image = st.file_uploader(
                "Upload image",
                type=SUPPORTED_IMAGE_TYPES,
                accept_multiple_files=False,
            )
            if uploaded_image is not None:
                uploaded_image_path = save_uploaded_file(uploaded_image, IMAGE_UPLOAD_DIR)
                st.session_state.uploaded_image_path = uploaded_image_path

        if source_mode == "Video":
            uploaded_video = st.file_uploader(
                "Upload video",
                type=SUPPORTED_VIDEO_TYPES,
                accept_multiple_files=False,
            )
            if uploaded_video is not None:
                uploaded_video_path = save_uploaded_file(uploaded_video, VIDEO_UPLOAD_DIR)
                st.session_state.uploaded_video_path = uploaded_video_path

        start_clicked = st.button("Start Analysis", use_container_width=True, disabled=not model_ready)
        stop_clicked = st.button("Stop", use_container_width=True)
        reset_clicked = st.button("Reset Session", use_container_width=True)
        export_clicked = st.button("Export ONNX + Benchmark", use_container_width=True, disabled=not model_ready)

        st.markdown("---")
        st.caption("This prototype uses a general YOLOv8 pretrained detector and presents detections as lesion candidates for workflow simulation only.")

    return {
        "source_mode": source_mode,
        "confidence": confidence,
        "temporal_window": temporal_window,
        "tracker_name": tracker_name,
        "enable_fp16": enable_fp16,
        "loop_video": loop_video,
        "uploaded_image_path": uploaded_image_path or st.session_state.uploaded_image_path,
        "uploaded_video_path": uploaded_video_path or st.session_state.uploaded_video_path,
        "start_clicked": start_clicked,
        "stop_clicked": stop_clicked,
        "reset_clicked": reset_clicked,
        "export_clicked": export_clicked,
    }


def _get_pipeline(detector: LesionDetector | None, controls: dict) -> MedSightPipeline | None:
    settings = (
        controls["temporal_window"],
        controls["tracker_name"],
        controls["enable_fp16"],
    )
    if controls["reset_clicked"]:
        _reset_session(detector, controls)

    if detector is None:
        return None

    if st.session_state.pipeline is None or st.session_state.pipeline_settings != settings:
        st.session_state.pipeline = MedSightPipeline(
            detector=detector,
            temporal_window=controls["temporal_window"],
            tracker_name=controls["tracker_name"],
            enable_fp16=controls["enable_fp16"],
        )
        st.session_state.pipeline_settings = settings
        _add_log("info", "Pipeline initialized")

    if controls["stop_clicked"]:
        _stop_stream()
        _add_log("warn", "Streaming stopped")

    if controls["source_mode"] != st.session_state.source_mode:
        _stop_stream()
        st.session_state.source_mode = controls["source_mode"]
        st.session_state.source_error = None
        _add_log("info", f"Source changed to {controls['source_mode']}")

    if controls["start_clicked"]:
        st.session_state.source_mode = controls["source_mode"]
        st.session_state.source_error = None
        if controls["source_mode"] in {"Video", "Webcam"}:
            st.session_state.run_stream = True
            _add_log("info", "Live analysis started")
        else:
            _run_image_analysis(st.session_state.pipeline, controls)

    if controls["export_clicked"]:
        _run_optimization(detector)

    return st.session_state.pipeline


def _render_left_panel(pipeline: MedSightPipeline | None, controls: dict) -> None:
    st.markdown(
        """
        <div class="ms-panel" style="margin-bottom:0.6rem; padding-bottom:0.5rem;">
            <div class="ms-panel-title" style="margin-bottom:0;">🎥 Live Feed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Always wrap feed in an outer div so the section never collapses
    st.markdown('<div class="ms-feed-outer">', unsafe_allow_html=True)
    if pipeline is None:
        st.markdown(
            _idle_placeholder_html("Model Unavailable", "The YOLOv8 model could not be loaded. Check the model path."),
            unsafe_allow_html=True,
        )
    elif controls["source_mode"] == "Image":
        _render_image_feed(pipeline, controls)
    else:
        _render_stream_feed(pipeline, controls)
    st.markdown('</div>', unsafe_allow_html=True)

    _render_pipeline_metrics()
    _render_pipeline_map()


def _idle_placeholder_html(title: str, hint: str) -> str:
    return (
        f'<div class="ms-feed-idle">'
        f'<div class="ms-feed-idle-icon">🩺</div>'
        f'<div class="ms-feed-idle-title">{title}</div>'
        f'<div class="ms-feed-idle-hint">{hint}</div>'
        f'</div>'
    )


def _frame_to_data_url(frame_rgb: np.ndarray, quality: int = 82) -> str:
    """Convert an RGB numpy frame to a JPEG data-URL for inline HTML embedding."""
    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _feed_shell_html(frame_rgb: np.ndarray) -> str:
    """Return the complete ms-feed-shell div with the frame embedded as a data-URL.
    Everything is in ONE string so the browser sees proper parent/child HTML.
    """
    data_url = _frame_to_data_url(frame_rgb)
    return (
        '<div class="ms-feed-shell">'
        f'<img src="{data_url}" alt="Live feed frame">'
        '</div>'
    )


def _render_right_panel(pipeline: MedSightPipeline | None) -> None:
    _render_analytics_panel(pipeline)
    _render_logs_panel()
    _render_snapshot_panel()
    _render_optimization_panel()


def _render_image_feed(pipeline: MedSightPipeline, controls: dict) -> None:
    image_path = controls["uploaded_image_path"]
    if not image_path:
        st.markdown(
            _idle_placeholder_html(
                "Waiting for Image",
                "Upload a medical image in the sidebar, then press Start Analysis to begin detection.",
            ),
            unsafe_allow_html=True,
        )
        return

    latest_result = st.session_state.latest_result
    latest_signature = st.session_state.latest_image_signature
    signature = f"{image_path}:{controls['confidence']}:{controls['temporal_window']}"

    if latest_result is None or latest_signature != signature:
        _run_image_analysis(pipeline, controls)

    latest_result = st.session_state.latest_result
    if latest_result is None:
        st.markdown(
            _idle_placeholder_html("Processing…", "Inference in progress — results will appear momentarily."),
            unsafe_allow_html=True,
        )
        return

    st.markdown(_feed_shell_html(latest_result.rendered_frame), unsafe_allow_html=True)


@st.fragment(run_every=FRAGMENT_INTERVAL)
def _render_stream_feed(pipeline: MedSightPipeline, controls: dict) -> None:
    if not st.session_state.run_stream:
        source_label = controls["source_mode"]
        hint_map = {
            "Video": "Upload a video file in the sidebar, then press Start Analysis to begin real-time detection.",
            "Webcam": "Grant webcam access in your browser, then press Start Analysis to begin live detection.",
        }
        hint = hint_map.get(source_label, "Select a source and press Start Analysis.")
        st.markdown(
            _idle_placeholder_html(f"{source_label} Feed Ready", hint),
            unsafe_allow_html=True,
        )
        return

    stream, error = _get_active_stream(controls)
    if error:
        st.session_state.run_stream = False
        st.session_state.source_error = error
        _add_log("warn", error)
        st.markdown(
            _idle_placeholder_html("Source Error", error),
            unsafe_allow_html=True,
        )
        return

    frame_rgb = stream.read_frame()
    if frame_rgb is None:
        st.session_state.run_stream = False
        _add_log("warn", "Stream ended")
        st.markdown(
            _idle_placeholder_html("Stream Ended", "All frames processed. Press Reset Session or Start Analysis again to continue."),
            unsafe_allow_html=True,
        )
        return

    st.session_state.frame_count += 1
    result = pipeline.process_video_frame(
        frame_rgb=frame_rgb,
        frame_index=st.session_state.frame_count,
        total_frames=stream.total_frames or st.session_state.frame_count,
        confidence=controls["confidence"],
    )
    _store_result(result)
    st.markdown(_feed_shell_html(result.rendered_frame), unsafe_allow_html=True)


def _get_active_stream(controls: dict) -> tuple[VideoStream | None, str | None]:
    source_mode = controls["source_mode"]
    source = 0 if source_mode == "Webcam" else controls["uploaded_video_path"]
    signature = f"{source_mode}:{source}:{controls['loop_video']}"

    if source_mode == "Video" and not source:
        return None, "Upload a video file to start analysis."

    if st.session_state.stream_object is not None and st.session_state.stream_signature == signature:
        return st.session_state.stream_object, None

    _release_stream()
    stream = VideoStream(source=source, loop_video=controls["loop_video"])
    if not stream.is_opened():
        stream.release()
        return None, "Unable to open the selected source. Check webcam permissions or upload integrity."

    st.session_state.stream_object = stream
    st.session_state.stream_signature = signature
    st.session_state.total_frames = stream.total_frames
    _add_log("info", f"Stream opened: {source_mode}")
    return stream, None


def _release_stream() -> None:
    stream = st.session_state.get("stream_object")
    if stream is not None:
        stream.release()
    st.session_state.stream_object = None
    st.session_state.stream_signature = None


def _stop_stream() -> None:
    st.session_state.run_stream = False
    _release_stream()


def _reset_session(detector: LesionDetector | None, controls: dict) -> None:
    _stop_stream()
    st.session_state.frame_count = 0
    st.session_state.total_frames = 0
    st.session_state.latest_result = None
    st.session_state.latest_image_signature = None
    st.session_state.snapshots = []
    st.session_state.logs = deque(maxlen=LOG_LIMIT)
    st.session_state.optimization_result = None
    st.session_state.stats = {
        "fps": 0.0,
        "pipeline_ms": 0.0,
        "inference_ms": 0.0,
        "raw_detections": 0,
        "confirmed_detections": 0,
        "active_lesions": 0,
        "total_lesions": 0,
        "average_confidence": 0.0,
        "detection_frequency": 0.0,
    }
    if detector is not None:
        st.session_state.pipeline = MedSightPipeline(
            detector=detector,
            temporal_window=controls["temporal_window"],
            tracker_name=controls["tracker_name"],
            enable_fp16=controls["enable_fp16"],
        )
        st.session_state.pipeline_settings = (
            controls["temporal_window"],
            controls["tracker_name"],
            controls["enable_fp16"],
        )
    _add_log("info", "Session reset")


def _run_image_analysis(pipeline: MedSightPipeline, controls: dict) -> None:
    image_path = controls["uploaded_image_path"]
    if not image_path:
        st.warning("Upload an image to run analysis.")
        return

    frame_rgb = decode_uploaded_image(image_path)
    result = pipeline.process_image(frame_rgb=frame_rgb, confidence=controls["confidence"])
    st.session_state.latest_image_signature = f"{image_path}:{controls['confidence']}:{controls['temporal_window']}"
    _store_result(result)


def _run_optimization(detector: LesionDetector) -> None:
    result_frame = None
    if st.session_state.latest_result is not None:
        result_frame = st.session_state.latest_result.raw_frame

    optimizer = ModelOptimizer(DEFAULT_MODEL_PATH)
    with st.spinner("Exporting ONNX model and benchmarking latency..."):
        st.session_state.optimization_result = optimizer.optimize(detector, sample_frame=result_frame)
    optimization = st.session_state.optimization_result
    if optimization.error:
        _add_log("warn", f"Optimization failed: {optimization.error}")
    else:
        _add_log("info", f"ONNX exported to {optimization.onnx_path.name}")


def _store_result(result: PipelineFrameResult) -> None:
    st.session_state.latest_result = result
    st.session_state.frame_count = result.frame_index
    st.session_state.total_frames = result.total_frames
    stats = result.analytics
    st.session_state.stats = {
        "fps": stats.fps,
        "pipeline_ms": stats.pipeline_ms,
        "inference_ms": stats.inference_ms,
        "raw_detections": stats.raw_detections,
        "confirmed_detections": stats.confirmed_detections,
        "active_lesions": stats.active_lesions,
        "total_lesions": stats.total_confirmed_lesions,
        "average_confidence": stats.average_confidence,
        "detection_frequency": stats.detection_frequency,
    }
    for level, message in result.logs:
        _add_log(level, message)

    if result.snapshot_eligible and result.confirmed_detections:
        last_frame = st.session_state.frame_count
        if last_frame % SNAPSHOT_INTERVAL_FRAMES == 0 or len(st.session_state.snapshots) == 0:
            st.session_state.snapshots.insert(
                0,
                {
                    "frame_index": result.frame_index,
                    "image": result.rendered_frame,
                    "count": len(result.confirmed_detections),
                },
            )
            st.session_state.snapshots = st.session_state.snapshots[:MAX_SNAPSHOTS]


def _render_pipeline_metrics() -> None:
    stats = st.session_state.stats
    st.markdown('<div class="ms-panel" style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="ms-panel-title">Runtime Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="ms-metric-grid">
            <div class="ms-stat-card"><div class="ms-stat-label">Current Frame</div><div class="ms-stat-value">{st.session_state.frame_count}</div></div>
            <div class="ms-stat-card"><div class="ms-stat-label">Total Frames</div><div class="ms-stat-value">{st.session_state.total_frames or '-'}</div></div>
            <div class="ms-stat-card"><div class="ms-stat-label">Inference Time</div><div class="ms-stat-value">{stats['inference_ms']:.1f} ms</div></div>
            <div class="ms-stat-card"><div class="ms-stat-label">Pipeline FPS</div><div class="ms-stat-value">{stats['fps']:.1f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_pipeline_map() -> None:
    st.markdown('<div class="ms-panel">', unsafe_allow_html=True)
    st.markdown('<div class="ms-panel-title">Frame Processing Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ```text
        video/image input
          -> frame extraction
          -> preprocessing
          -> YOLOv8 inference
          -> ByteTrack / BoT-SORT association
          -> temporal consistency filter
          -> analytics + snapshots + rendering
        ```
        """,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_analytics_panel(pipeline: MedSightPipeline | None) -> None:
    stats = st.session_state.stats
    st.markdown('<div class="ms-panel">', unsafe_allow_html=True)
    st.markdown('<div class="ms-panel-title">Lesion Analytics</div>', unsafe_allow_html=True)

    metric_cols = st.columns(2, gap="small")
    metric_cols[0].metric("Total lesions detected", int(stats["total_lesions"]))
    metric_cols[1].metric("Active lesions", int(stats["active_lesions"]))
    metric_cols[0].metric("Average confidence", f"{stats['average_confidence']:.2f}")
    metric_cols[1].metric("Detection frequency", f"{stats['detection_frequency']:.2f}/s")

    if pipeline is not None and pipeline.analytics.history:
        history_df = pd.DataFrame([item.__dict__ for item in pipeline.analytics.history])
        st.line_chart(history_df.set_index("frame_index")[["confirmed_detections", "fps", "average_confidence"]], height=220)

        track_df = build_track_dataframe(pipeline.tracker.snapshot())
        if not track_df.empty:
            st.dataframe(track_df, width="stretch", hide_index=True)
    else:
        st.caption("Analytics will populate after the first processed frame.")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_logs_panel() -> None:
    st.markdown('<div class="ms-panel">', unsafe_allow_html=True)
    st.markdown('<div class="ms-panel-title">Real-Time AI Logs</div>', unsafe_allow_html=True)
    log_lines = []
    prefix_map = {
        "info": ("[INFO]", "ms-info"),
        "detect": ("[DETECTION]", "ms-detect"),
        "track": ("[TRACKING]", "ms-track"),
        "warn": ("[WARNING]", "ms-warn"),
    }
    for level, message in list(st.session_state.logs)[-18:]:
        prefix, css_class = prefix_map.get(level, ("[LOG]", "ms-info"))
        log_lines.append(f'<p class="ms-log-line {css_class}">{prefix} {message}</p>')

    rendered_logs = "".join(log_lines) or '<p class="ms-log-line ms-info">[INFO] System idle</p>'
    st.markdown(f'<div class="ms-log-shell">{rendered_logs}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_snapshot_panel() -> None:
    st.markdown('<div class="ms-panel">', unsafe_allow_html=True)
    st.markdown('<div class="ms-panel-title">Snapshot Panel</div>', unsafe_allow_html=True)
    snapshots = st.session_state.snapshots
    if not snapshots:
        st.caption("Snapshots will appear automatically when confirmed detections persist.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for snapshot in snapshots[:4]:
        st.markdown(
            f"""
            <div class="ms-snapshot-card" style="padding: 0.75rem; margin-bottom: 0.75rem;">
                <div class="ms-small-note">Frame {snapshot['frame_index']} • {snapshot['count']} confirmed lesion candidate(s)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(snapshot["image"], channels="RGB", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_optimization_panel() -> None:
    result: OptimizationResult | None = st.session_state.optimization_result
    st.markdown('<div class="ms-panel">', unsafe_allow_html=True)
    st.markdown('<div class="ms-panel-title">Model Optimization</div>', unsafe_allow_html=True)
    if result is None:
        st.caption("Run ONNX export to compare PyTorch and optimized inference latency.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if result.error:
        st.error(result.error)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    col1, col2 = st.columns(2, gap="small")
    col1.metric("PyTorch latency", f"{result.pytorch_latency_ms:.1f} ms")
    col2.metric("ONNX latency", f"{result.onnx_latency_ms:.1f} ms")
    col1.metric("Speedup", f"{result.speedup:.2f}x")
    col2.metric("FP16", "Enabled" if result.fp16_enabled else "Unavailable")
    st.caption(f"Export: `{result.onnx_path}` via `{result.provider}`")
    st.markdown("</div>", unsafe_allow_html=True)


def _add_log(level: str, message: str) -> None:
    st.session_state.logs.append((level, message))


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def _get_detector(model_path: str) -> LesionDetector:
    return LesionDetector(model_path)


def _load_detector_safe(model_path: str) -> tuple[LesionDetector | None, str | None]:
    try:
        return _get_detector(model_path), None
    except Exception as exc:  # pragma: no cover - surfaced in UI
        return None, str(exc)
