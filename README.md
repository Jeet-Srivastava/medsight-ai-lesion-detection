# MedSight AI

MedSight AI is a computer vision prototype for real-time lesion candidate detection, spatiotemporal tracking, analytics, and model optimization. It uses a YOLO11 model to simulate a clinical-grade workflow across image uploads, recorded video, and optional webcam input. The backend is a FastAPI server and the frontend is a React + Vite dashboard.

## Features

- YOLO11 image and video inference with confidence overlays
- ByteTrack / BoT-SORT tracking with persistent lesion IDs
- Temporal consistency filtering to suppress flickering detections
- ABCDE morphological scoring and risk assessment
- Runtime analytics for total lesions, active lesions, average confidence, and detection frequency
- Saliency / XAI heatmaps for explainability
- Clinical report generation
- Real-time logs and automatic snapshots for persistent findings
- ONNX export and latency benchmarking against the PyTorch model

## Project Layout

```text
medsight/
├── api_server.py           # FastAPI backend
├── frontend/               # React + Vite dashboard
├── medsight/               # Core Python package
│   ├── abcde.py
│   ├── analytics.py
│   ├── audit.py
│   ├── config.py
│   ├── detection.py
│   ├── explainability.py
│   ├── optimization.py
│   ├── pipeline.py
│   ├── reporting.py
│   ├── risk.py
│   ├── segmentation.py
│   ├── tracking.py
│   └── video.py
├── tests/
├── requirements.txt
└── README.md
```

## Run Locally

### Backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api_server:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The React dev server proxies `/api` requests to `http://localhost:8000`.

## Optimization Notes

The ONNX export flow depends on `onnx` and `onnxruntime`. If FP16 is supported by the local accelerator, MedSight enables it automatically during benchmarking.

## Disclaimer

This project is a software engineering and computer vision prototype. The included YOLO11 checkpoint is not a medically trained lesion model and must not be used for clinical diagnosis.
