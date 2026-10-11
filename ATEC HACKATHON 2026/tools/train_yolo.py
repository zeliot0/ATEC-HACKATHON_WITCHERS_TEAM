"""
Trains the hazard YOLO model on the generated synthetic dataset.

Usage:  python tools/train_yolo.py [epochs]
Output: models/hazard_yolo.pt
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    from ultralytics import YOLO
    import torch

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"training on: {'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLO("yolov8n.pt")  # small pretrained backbone, fine-tuned on our classes
    model.train(
        data=str(ROOT / "datasets" / "hazards" / "data.yaml"),
        epochs=epochs, imgsz=480, batch=32 if device == 0 else 8,
        device=device, workers=0, project=str(ROOT / "runs"), name="hazard",
        exist_ok=True, verbose=True, plots=False,
        # frames are single-channel grey renders -- colour/flip augments add nothing useful
        hsv_h=0.0, hsv_s=0.0, fliplr=0.5, mosaic=0.5,
    )
    best = ROOT / "runs" / "hazard" / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / "hazard_yolo.pt")
    print("saved models/hazard_yolo.pt")
    m = model.val(data=str(ROOT / "datasets" / "hazards" / "data.yaml"), imgsz=480, device=device)
    print(f"mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")


if __name__ == "__main__":
    main()
