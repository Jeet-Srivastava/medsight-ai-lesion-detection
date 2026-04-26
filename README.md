# MedSight AI

MedSight AI is a Streamlit-based computer vision prototype for real-time lesion candidate detection, spatiotemporal tracking, analytics, and model optimization. It uses a pretrained YOLOv8 model to simulate a clinical-grade workflow across image uploads, recorded video, and optional webcam input.

## Features

- YOLOv8 image and video inference with confidence overlays
- ByteTrack / BoT-SORT tracking with persistent lesion IDs
- Temporal consistency filtering to suppress flickering detections
- Runtime analytics for total lesions, active lesions, average confidence, and detection frequency
- Real-time logs and automatic snapshots for persistent findings
- ONNX export and latency benchmarking against the PyTorch model
- Dark, medical-style Streamlit interface

## Project Layout

```text
medsight/
├── app.py
├── models/
├── ui/
├── utils/
├── medsight/
│   ├── analytics.py
│   ├── config.py
│   ├── detection.py
│   ├── optimization.py
│   ├── pipeline.py
│   ├── tracking.py
│   └── video.py
├── tests/
├── requirements.txt
└── README.md
```

## Run Locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Optimization Notes

The ONNX export flow depends on `onnx` and `onnxruntime`. If FP16 is supported by the local accelerator, MedSight enables it automatically during benchmarking.

## Disclaimer

This project is a software engineering and computer vision prototype. The included YOLOv8 checkpoint is not a medically trained lesion model and must not be used for clinical diagnosis.
