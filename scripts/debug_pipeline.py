"""Run this to debug detection on any video file."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from config.settings import CONFIDENCE_THRESHOLD, VEHICLE_CLASS_IDS, VEHICLE_MODEL
from tracking.traffic_pipeline import TrafficPipeline
from ultralytics import YOLO


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else None
    if not video:
        from tracking.video_source import resolve_video_source
        from config.settings import DEFAULT_DEMO_SEQUENCE
        video = resolve_video_source(f"visdrone:{DEFAULT_DEMO_SEQUENCE}")

    print(f"Video: {video}")
    cap = cv2.VideoCapture(video)
    print(f"Opened: {cap.isOpened()}")
    ret, frame = cap.read()
    print(f"Frame 0: ret={ret}, shape={frame.shape if ret else None}")
    cap.release()

    print(f"\nModel: {VEHICLE_MODEL}, conf={CONFIDENCE_THRESHOLD}")
    model = YOLO(VEHICLE_MODEL)
    r = model.predict(frame, conf=CONFIDENCE_THRESHOLD, classes=list(VEHICLE_CLASS_IDS.keys()), verbose=False)
    boxes = r[0].boxes
    n = len(boxes) if boxes is not None else 0
    print(f"Raw YOLO predict: {n} vehicles")
    if boxes is not None:
        for i in range(min(n, 5)):
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            print(f"  [{i}] class={VEHICLE_CLASS_IDS.get(cls,'?')} conf={conf:.2f}")

    print("\nFull pipeline frame 0:")
    pipe = TrafficPipeline()
    a = pipe.process_frame(frame, 0)
    print(f"  vehicles={a.total_vehicles}")
    print(f"  debug={a.debug}")
    if a.annotated_frame is not None:
        out = Path("outputs/debug_frame0.jpg")
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), a.annotated_frame)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
