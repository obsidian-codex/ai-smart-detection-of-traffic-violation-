"""Fine-tune YOLO11s for helmet / no-helmet detection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from config.settings import HELMET_MODEL_PATH, MODELS_DIR, VEHICLE_MODEL
from training.prepare_helmet_dataset import prepare


def _auto_device(requested: str = "0") -> str:
    import torch
    if requested == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return requested


def train(epochs: int = 50, imgsz: int = 640, batch: int = 16, device: str = "0"):
    yaml_path = prepare()
    device = _auto_device(device)
    if device == "cpu":
        batch = min(batch, 8)
        print("CUDA not available — training on CPU (install torch+cu124 for RTX 4060 GPU acceleration)")
    model = YOLO(VEHICLE_MODEL)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(MODELS_DIR),
        name="helmet_yolo11s",
        patience=15,
        save=True,
        plots=True,
        verbose=True,
    )
    print(f"Training complete. Best weights: {HELMET_MODEL_PATH}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train helmet detection model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device)
