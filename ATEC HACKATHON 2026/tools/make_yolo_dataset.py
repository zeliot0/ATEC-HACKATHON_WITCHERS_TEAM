"""
Generates a YOLO-format training set from the project's own renderer.

Because every frame comes from render_world_view(), the ground-truth boxes
are exact -- no manual labelling. Rover poses and hazard fields are
randomised so the detector sees many viewpoints, distances, and a share of
radiation-corrupted (solar flare) frames so it learns to survive them.

Usage:  python tools/make_yolo_dataset.py [n_train] [n_val]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.localization import CameraModel, RoverPose
from src.perception import generate_world_hazard_field, render_world_view

W, H = 480, 360
CAMERA = CameraModel(fx=420, fy=420, cx=W / 2, cy=H / 2, height_m=1.2, pitch_deg=22)
CLASSES = ["crater", "rock"]
OUT = ROOT / "datasets" / "hazards"


def make_split(name: str, n: int, seed: int) -> int:
    rng = random.Random(seed)
    np.random.seed(seed)
    img_dir = OUT / "images" / name
    lbl_dir = OUT / "labels" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    boxes_total = 0
    for i in range(n):
        field = generate_world_hazard_field(n_hazards=rng.randint(8, 22), area_m=26.0,
                                            seed=rng.randint(0, 10**9))
        rover = RoverPose(x_m=rng.uniform(0, 24), y_m=rng.uniform(0, 24),
                          heading_deg=rng.uniform(0, 360))
        flare = rng.random() < 0.2
        frame, _disp, visible = render_world_view(field, CAMERA, rover, W, H, flare=flare)

        lines = []
        for v in visible:
            r = v["radius_px"]
            x1, y1 = max(0.0, v["px"] - r), max(0.0, v["py"] - r)
            x2, y2 = min(W - 1.0, v["px"] + r), min(H - 1.0, v["py"] + r)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue  # fully off-frame / degenerate
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            lines.append(f"{CLASSES.index(v['kind'])} {cx:.6f} {cy:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        cv2.imwrite(str(img_dir / f"{name}_{i:05d}.jpg"), frame)
        (lbl_dir / f"{name}_{i:05d}.txt").write_text("\n".join(lines))
        boxes_total += len(lines)
    return boxes_total


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    tb = make_split("train", n_train, seed=1)
    vb = make_split("val", n_val, seed=2)
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: crater\n  1: rock\n")
    print(f"train: {n_train} imgs / {tb} boxes   val: {n_val} imgs / {vb} boxes")
    print(f"dataset written to {OUT}")


if __name__ == "__main__":
    main()
