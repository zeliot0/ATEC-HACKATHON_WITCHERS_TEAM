"""
YOLO hazard detector wrapper.

Loads models/hazard_yolo.pt lazily. If torch/ultralytics or the weights are
missing, `available()` is False and the pipeline silently keeps using the
stereo-depth detector -- YOLO is an enhancement, never a hard dependency.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.perception import Hazard

WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "hazard_yolo.pt"
CLASSES = ["crater", "rock"]

_model = None
_failed = False


def _load():
    global _model, _failed
    if _model is not None or _failed:
        return _model
    try:
        if not WEIGHTS.exists():
            raise FileNotFoundError(WEIGHTS)
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS))
        # warm up so the first live frame isn't slow
        _model.predict(np.zeros((360, 480, 3), np.uint8), imgsz=480, verbose=False)
    except Exception as e:  # noqa: BLE001 -- any failure => graceful fallback
        print(f"[yolo] unavailable, using depth detector only: {e}")
        _failed = True
    return _model


def available() -> bool:
    return _load() is not None


def detect(frame: np.ndarray, conf: float = 0.35) -> list[Hazard]:
    model = _load()
    if model is None:
        return []
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
    res = model.predict(img, imgsz=480, conf=conf, verbose=False)[0]
    out: list[Hazard] = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append(Hazard(px=(x1 + x2) / 2, py=(y1 + y2) / 2,
                          radius_px=max(x2 - x1, y2 - y1) / 2,
                          confidence=float(b.conf[0]), kind=CLASSES[int(b.cls[0])]))
    return out


def fuse(yolo: list[Hazard], depth: list[Hazard]) -> list[Hazard]:
    """Agreement-weighted fusion. A hazard seen by BOTH detectors (centres
    within ~1 radius) gets boosted confidence; a lone detection keeps a
    reduced one. YOLO's box/class wins when both agree."""
    out: list[Hazard] = []
    used = set()
    for y in yolo:
        match = None
        for i, d in enumerate(depth):
            if i in used:
                continue
            if np.hypot(y.px - d.px, y.py - d.py) < max(y.radius_px, d.radius_px, 6):
                match = i
                break
        if match is not None:
            used.add(match)
            out.append(Hazard(y.px, y.py, y.radius_px,
                              min(1.0, 0.6 + 0.4 * max(y.confidence, depth[match].confidence)), y.kind))
        else:
            out.append(Hazard(y.px, y.py, y.radius_px, y.confidence * 0.8, y.kind))
    for i, d in enumerate(depth):
        if i not in used:
            out.append(Hazard(d.px, d.py, d.radius_px, d.confidence * 0.7, d.kind))
    return out
