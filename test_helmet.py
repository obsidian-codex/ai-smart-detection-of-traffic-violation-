import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import HELMET_MODEL_PATH

def main():
    model_path = HELMET_MODEL_PATH
    print(f"Helmet model loaded: {model_path}")
    
    if not model_path.exists():
        print("ERROR: Helmet model not found!")
        return

    model = YOLO(model_path)
    print("Class Names:", model.names)

    # We will test on 5 random motorcycle crops. We need some video to extract crops first.
    # Alternatively, we can use the video from VisDrone or something else.
    # To keep it standalone, let's grab a few frames from the default demo sequence.
    video_path = PROJECT_ROOT / "outputs" / "demo_uav0000071_03240_v.mp4"
    if not video_path.exists():
        print(f"ERROR: Video {video_path} not found.")
        return

    cap = cv2.VideoCapture(str(video_path))
    vehicle_model = YOLO("yolo11s.pt")

    out_dir = PROJECT_ROOT / "outputs" / "helmet_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    crops_saved = 0
    while cap.isOpened() and crops_saved < 5:
        ret, frame = cap.read()
        if not ret:
            break

        results = vehicle_model.predict(frame, classes=[3], verbose=False) # 3 is motorcycle
        if not results or results[0].boxes is None:
            continue
            
        boxes = results[0].boxes
        for i in range(len(boxes)):
            if crops_saved >= 5:
                break
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            
            # enlarge ROI by 20%
            w = x2 - x1
            h = y2 - y1
            mx1 = max(0, int(x1 - w * 0.1))
            my1 = max(0, int(y1 - h * 0.1))
            mx2 = min(frame.shape[1], int(x2 + w * 0.1))
            my2 = min(frame.shape[0], int(y2 + h * 0.1))
            
            crop = frame[my1:my2, mx1:mx2]
            if crop.size == 0:
                continue

            # Run helmet detection
            h_results = model.predict(crop, conf=0.20, verbose=False)
            h_boxes = h_results[0].boxes if h_results else []
            
            print(f"--- Crop {crops_saved + 1} ---")
            print(f"Detection count: {len(h_boxes) if h_boxes else 0}")
            
            if h_boxes:
                for j in range(len(h_boxes)):
                    hx1, hy1, hx2, hy2 = h_boxes.xyxy[j].cpu().numpy()
                    cls_id = int(h_boxes.cls[j].item())
                    conf = float(h_boxes.conf[j].item())
                    cls_name = model.names.get(cls_id, "unknown")
                    print(f"BBox: [{hx1:.1f}, {hy1:.1f}, {hx2:.1f}, {hy2:.1f}] | Class: {cls_name} ({cls_id}) | Confidence: {conf:.2f}")
                    
                    cv2.rectangle(crop, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (0, 0, 255), 2)
                    cv2.putText(crop, f"{cls_name} {conf:.2f}", (int(hx1), int(hy1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            out_path = out_dir / f"crop_{crops_saved}.jpg"
            cv2.imwrite(str(out_path), crop)
            print(f"Saved: {out_path}")
            crops_saved += 1

    cap.release()

if __name__ == "__main__":
    main()
