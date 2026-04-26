from __future__ import annotations

from collections import deque
from pathlib import Path

import streamlit as st

from detector import LesionDetector
from medsight.config import (
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL_PATH,
    FRAGMENT_INTERVAL,
    SUPPORTED_VIDEO_TYPES,
    UPLOAD_DIR,
)
from video_stream import VideoStream, save_uploaded_video


def configure_page() -> None:
    st.set_page_config(
        page_title="MedSight",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_app_shell() -> None:
    _initialize_state()
    _render_styles()
    _render_hero()

    detector, model_error = _load_detector_safe(DEFAULT_MODEL_PATH)
    controls = _render_sidebar(model_ready=detector is not None)
    _sync_runtime_state(controls)

    if model_error:
        st.error(
            "The YOLO model could not be loaded. Check that `yolov8n.pt` exists and "
            "your Ultralytics setup is healthy."
        )
        st.caption(f"Details: {model_error}")

    _render_runtime(detector, model_error, controls)


def _initialize_state() -> None:
    defaults = {
        "run_stream": False,
        "stream_object": None,
        "stream_signature": None,
        "stats": {
            "detections": 0,
            "fps": 0.0,
            "frame_time_ms": 0.0,
            "inference_ms": 0.0,
            "top_confidence": 0.0,
        },
        "fps_history": deque(maxlen=12),
        "last_source_mode": "Webcam",
        "last_upload_path": None,
        "stream_generation": 0,
        "source_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ms-bg: #eef4fb;
                --ms-panel: rgba(255, 255, 255, 0.90);
                --ms-border: rgba(148, 163, 184, 0.22);
                --ms-text: #0f172a;
                --ms-muted: #475569;
                --ms-accent: #1d4ed8;
                --ms-accent-soft: #dbeafe;
                --ms-shadow: 0 20px 55px rgba(15, 23, 42, 0.08);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(191, 219, 254, 0.72), transparent 30%),
                    radial-gradient(circle at bottom right, rgba(226, 232, 240, 0.9), transparent 28%),
                    linear-gradient(180deg, #f8fbff 0%, var(--ms-bg) 100%);
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            .medsight-hero,
            .medsight-panel,
            .medsight-note {
                background: var(--ms-panel);
                border: 1px solid var(--ms-border);
                border-radius: 24px;
                box-shadow: var(--ms-shadow);
                backdrop-filter: blur(14px);
            }

            .medsight-hero {
                padding: 2rem 2.2rem;
                margin-bottom: 1.25rem;
            }

            .medsight-panel {
                padding: 1.15rem;
            }

            .medsight-note {
                padding: 1rem 1.05rem;
                margin-bottom: 0.9rem;
            }

            .medsight-kicker {
                color: var(--ms-accent);
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.65rem;
            }

            .medsight-title {
                color: var(--ms-text);
                font-size: 2.7rem;
                line-height: 1.06;
                font-weight: 800;
                margin: 0;
            }

            .medsight-subtitle {
                color: var(--ms-muted);
                font-size: 1rem;
                line-height: 1.75;
                margin-top: 0.9rem;
                max-width: 52rem;
            }

            .medsight-note-label {
                color: var(--ms-muted);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.25rem;
            }

            .medsight-note-value {
                color: var(--ms-text);
                font-size: 1.1rem;
                font-weight: 700;
                line-height: 1.35;
            }

            .medsight-feed-frame {
                border-radius: 18px;
                overflow: hidden;
                background: linear-gradient(180deg, #0f172a 0%, #162337 100%);
                min-height: 360px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px solid rgba(148, 163, 184, 0.18);
            }

            .medsight-empty-state {
                color: #cbd5e1;
                text-align: center;
                padding: 2rem;
                max-width: 26rem;
                margin: 0 auto;
            }

            .medsight-legend {
                color: var(--ms-muted);
                font-size: 0.94rem;
                line-height: 1.7;
            }

            [data-testid="stSidebar"] {
                background: rgba(255, 255, 255, 0.8);
                border-right: 1px solid rgba(148, 163, 184, 0.18);
                backdrop-filter: blur(12px);
            }

            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid rgba(191, 219, 254, 0.9);
                border-radius: 18px;
                padding: 0.7rem 0.9rem;
                box-shadow: 0 12px 32px rgba(15, 23, 42, 0.05);
            }

            .stButton > button {
                border-radius: 999px;
                height: 2.9rem;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        f"""
        <section class="medsight-hero">
            <div class="medsight-kicker">Clinical Vision Prototype</div>
            <h1 class="medsight-title">{APP_TITLE}</h1>
            <p class="medsight-subtitle">{APP_SUBTITLE}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def _get_detector(model_path: str) -> LesionDetector:
    return LesionDetector(model_path)


def _load_detector_safe(model_path: str):
    try:
        return _get_detector(model_path), None
    except Exception as error:  # pragma: no cover - UI fallback
        return None, str(error)


def _render_sidebar(model_ready: bool) -> dict:
    with st.sidebar:
        st.markdown("### Control Panel")
        st.caption("Switch sources, tune confidence, and run the live demo.")

        source_mode = st.radio(
            "Input Source",
            ("Webcam", "Upload Video"),
            index=0,
        )
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=DEFAULT_CONFIDENCE,
            step=0.05,
        )
        draw_boxes = st.toggle("Show Bounding Boxes", value=True)

        uploaded_file = None
        uploaded_path = None
        if source_mode == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload Video",
                type=SUPPORTED_VIDEO_TYPES,
                help="Supported formats: MP4, MOV, AVI, MKV",
            )
            if uploaded_file is not None:
                uploaded_path = save_uploaded_video(uploaded_file, upload_dir=UPLOAD_DIR)
                st.caption(f"Loaded: `{Path(uploaded_path).name}`")

        start_disabled = not model_ready or (
            source_mode == "Upload Video" and uploaded_path is None
        )

        start_col, stop_col = st.columns(2)
        start_clicked = start_col.button(
            "Start",
            type="primary",
            use_container_width=True,
            disabled=start_disabled,
        )
        stop_clicked = stop_col.button(
            "Stop",
            use_container_width=True,
        )
        reset_clicked = st.button("Reset Stream", use_container_width=True)

        st.markdown("---")
        st.markdown("#### Demo Notes")
        st.caption("YOLOv8 is used here as a prototype detector with simulated lesion labeling.")

    return {
        "source_mode": source_mode,
        "confidence": confidence,
        "draw_boxes": draw_boxes,
        "uploaded_path": uploaded_path,
        "start_clicked": start_clicked,
        "stop_clicked": stop_clicked,
        "reset_clicked": reset_clicked,
    }


def _sync_runtime_state(controls: dict) -> None:
    mode_changed = controls["source_mode"] != st.session_state.last_source_mode
    upload_changed = controls["uploaded_path"] != st.session_state.last_upload_path

    if mode_changed or upload_changed:
        _release_stream()
        st.session_state.stream_generation += 1
        st.session_state.last_source_mode = controls["source_mode"]
        st.session_state.last_upload_path = controls["uploaded_path"]
        st.session_state.source_error = None

    if controls["start_clicked"]:
        st.session_state.run_stream = True
        st.session_state.source_error = None

    if controls["stop_clicked"]:
        st.session_state.run_stream = False
        _release_stream()

    if controls["reset_clicked"]:
        _release_stream()
        st.session_state.stream_generation += 1
        st.session_state.fps_history.clear()
        st.session_state.stats = {
            "detections": 0,
            "fps": 0.0,
            "frame_time_ms": 0.0,
            "inference_ms": 0.0,
            "top_confidence": 0.0,
        }

    if controls["source_mode"] == "Upload Video" and controls["uploaded_path"] is None:
        st.session_state.run_stream = False


def _render_runtime(detector, model_error: str | None, controls: dict) -> None:
    stream_col, status_col = st.columns([1.75, 0.75], gap="large")

    with stream_col:
        st.markdown('<section class="medsight-panel">', unsafe_allow_html=True)
        st.subheader("Live Detection Workspace")
        _render_stream_fragment(detector, model_error, controls)
        st.markdown("</section>", unsafe_allow_html=True)

    with status_col:
        _render_status_panel(controls)


@st.fragment(run_every=FRAGMENT_INTERVAL)
def _render_stream_fragment(detector, model_error: str | None, controls: dict) -> None:
    if model_error:
        st.warning("Model unavailable. Resolve the load issue to start inference.")
        _render_metrics()
        return

    if not st.session_state.run_stream:
        _render_idle_state(controls["source_mode"], controls["uploaded_path"])
        _render_metrics()
        return

    stream, error = _get_active_stream(
        controls["source_mode"],
        controls["uploaded_path"],
    )
    if error:
        st.session_state.run_stream = False
        st.session_state.source_error = error
        st.error(error)
        _render_metrics()
        return

    frame_rgb = stream.read_frame()
    if frame_rgb is None:
        st.session_state.run_stream = False
        st.session_state.source_error = (
            "Video feed stopped unexpectedly. Reset the stream and try again."
        )
        st.error(st.session_state.source_error)
        _render_metrics()
        return

    annotated_frame, analysis = detector.process_frame(
        frame_rgb,
        conf_threshold=controls["confidence"],
        draw_boxes=controls["draw_boxes"],
    )
    fps = 1000.0 / analysis.total_ms if analysis.total_ms > 0 else 0.0
    st.session_state.fps_history.append(fps)
    smoothed_fps = sum(st.session_state.fps_history) / len(st.session_state.fps_history)

    st.session_state.stats = {
        "detections": analysis.detections,
        "fps": smoothed_fps,
        "frame_time_ms": analysis.total_ms,
        "inference_ms": analysis.inference_ms,
        "top_confidence": analysis.top_confidence,
    }

    st.image(annotated_frame, channels="RGB", use_container_width=True)
    _render_metrics()


def _render_idle_state(source_mode: str, uploaded_path: str | None) -> None:
    message = "Press Start to begin the live lesion detection demo."
    if source_mode == "Upload Video" and uploaded_path is None:
        message = "Upload a video file to preview AI detections on recorded footage."
    elif source_mode == "Webcam":
        message = "Webcam mode is ready. Press Start to open the camera feed."

    st.markdown(
        f"""
        <div class="medsight-feed-frame">
            <div class="medsight-empty-state">
                <h4 style="margin-bottom: 0.5rem;">Detection feed standby</h4>
                <p style="margin: 0; line-height: 1.7;">{message}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.source_error:
        st.warning(st.session_state.source_error)


def _render_metrics() -> None:
    stats = st.session_state.stats
    metric_cols = st.columns(4, gap="medium")
    metric_cols[0].metric("Detections", f"{stats['detections']}")
    metric_cols[1].metric("FPS", f"{stats['fps']:.1f}")
    metric_cols[2].metric("Frame Time", f"{stats['frame_time_ms']:.1f} ms")
    metric_cols[3].metric("Inference", f"{stats['inference_ms']:.1f} ms")


def _render_status_panel(controls: dict) -> None:
    stats = st.session_state.stats
    source_label = (
        Path(controls["uploaded_path"]).name
        if controls["source_mode"] == "Upload Video" and controls["uploaded_path"]
        else controls["source_mode"]
    )
    state_label = "Running" if st.session_state.run_stream else "Standby"

    st.markdown(
        f"""
        <section class="medsight-note">
            <div class="medsight-note-label">System State</div>
            <div class="medsight-note-value">{state_label}</div>
        </section>
        <section class="medsight-note">
            <div class="medsight-note-label">Active Source</div>
            <div class="medsight-note-value">{source_label}</div>
        </section>
        <section class="medsight-note">
            <div class="medsight-note-label">Highest Confidence</div>
            <div class="medsight-note-value">{stats['top_confidence']:.2f}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="medsight-panel">
            <div class="medsight-note-label">Clinical Display Notes</div>
            <div class="medsight-legend">
                Potential findings are presented with a simulated lesion label and
                a confidence score. This demo is intended for prototype visualization,
                not diagnostic use.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _get_active_stream(source_mode: str, uploaded_path: str | None):
    signature = (
        f"{source_mode}:{uploaded_path}:{st.session_state.stream_generation}"
    )
    current_signature = st.session_state.stream_signature
    stream = st.session_state.stream_object

    if stream is not None and current_signature == signature:
        return stream, None

    _release_stream()
    source = 0 if source_mode == "Webcam" else uploaded_path
    if source_mode == "Upload Video" and uploaded_path is None:
        return None, "Upload a valid video file to start playback."

    stream = VideoStream(source=source, loop_video=source_mode == "Upload Video")
    if not stream.is_opened():
        stream.release()
        if source_mode == "Webcam":
            return None, "No webcam could be opened. Check camera permissions and availability."
        return None, "The uploaded video could not be opened. Try another supported file."

    st.session_state.stream_object = stream
    st.session_state.stream_signature = signature
    return stream, None


def _release_stream() -> None:
    stream = st.session_state.get("stream_object")
    if stream is not None:
        stream.release()
    st.session_state.stream_object = None
    st.session_state.stream_signature = None
