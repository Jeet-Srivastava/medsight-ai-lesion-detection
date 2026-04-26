import numpy as np

from detector import LesionDetector

def main():
    print("Loading detector...")
    detector = LesionDetector("yolov8n.pt")
    print("Model loaded successfully.")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    print("Running inference on a single frame...")
    _, analysis = detector.process_frame(dummy_frame, conf_threshold=0.25)

    print("\n--- Inference Results ---")
    print(f"Detections: {analysis.detections}")
    print(f"Inference time: {analysis.inference_ms:.2f} ms")
    print("Detector pipeline is working.")

if __name__ == "__main__":
    main()
