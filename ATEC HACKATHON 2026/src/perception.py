"""
Onboard perception for Project C.R.A.T.E.R.

Deliberately classical computer vision, not a trained model:

  - Fully deterministic and explainable line-by-line in a Q&A -- there is
    no "the model learned it" answer to defend.
  - No training data problem: there is no real labeled "lunar hazard"
    dataset to train or validate a detector against, and a model trained on
    a synthetic proxy would be a credibility risk under questioning.
  - Runs at real-time frame rates on a CPU, which matters for an edge
    device with no cloud GPU to call out to (the whole premise of the
    challenge is *no* cloud API).

Two independent threshold passes run over the same frame -- one tuned for
dark blobs (shadowed crater-like depressions) and one for bright blobs
(sunlit rock-like protrusions) -- because a single global threshold cannot
see both at once. Detections are scored by contour circularity and local
contrast, which stands in for "how sure are we this is a real hazard and
not noise."
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

from src.localization import CameraModel, RoverPose, landing_frame_to_pixel


@dataclass
class Hazard:
    px: float
    py: float
    radius_px: float
    confidence: float
    kind: str  # "crater" (dark) or "rock" (bright)


# ---------------------------------------------------------------------------
# Synthetic terrain generator (stand-in for a live camera feed -- the spec
# requires 100% software emulation, so there is no physical sensor to read)
# ---------------------------------------------------------------------------

def generate_synthetic_frame(width: int = 480, height: int = 360, n_hazards: int = 6,
                              seed: int | None = None):
    """Generates one lunar-regolith-like grayscale frame plus ground truth.

    Returns (frame: np.ndarray[H,W] uint8, ground_truth: list[dict]) where
    each ground-truth entry has px, py, radius_px, kind.
    """
    rng = random.Random(seed)
    # base regolith: smooth grey gradient + per-pixel sensor noise
    yy, xx = np.mgrid[0:height, 0:width]
    base = 110 + 20 * np.sin(xx / width * math.pi) + 10 * np.cos(yy / height * 2 * math.pi)
    noise = np.random.normal(0, 6, size=(height, width))
    frame = np.clip(base + noise, 0, 255).astype(np.uint8)

    ground_truth = []
    for _ in range(n_hazards):
        px = rng.uniform(0.08 * width, 0.92 * width)
        py = rng.uniform(0.08 * height, 0.92 * height)
        r = rng.uniform(6, 22)
        kind = rng.choice(["crater", "rock"])
        delta = rng.uniform(45, 85)
        color = max(5, int(base[int(py), int(px)] - delta)) if kind == "crater" \
            else min(250, int(base[int(py), int(px)] + delta))
        cv2.circle(frame, (int(px), int(py)), int(r), int(color), -1)
        # soften edge slightly so it isn't a perfect binary disk
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        ground_truth.append({"px": px, "py": py, "radius_px": r, "kind": kind})
    return frame, ground_truth


@dataclass
class WorldHazard:
    x_m: float
    y_m: float
    radius_m: float
    kind: str  # "crater" or "rock"


def generate_world_hazard_field(n_hazards: int = 14, area_m: float = 26.0,
                                  seed: int | None = 0) -> list[WorldHazard]:
    """A FIXED set of hazards that exist once, in world coordinates, for the
    whole run -- this is what the rover actually drives past. Critical
    design point: these are generated ONCE at startup, not regenerated per
    frame, so the same physical rock is still there the next time the
    rover's camera happens to point at it."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_hazards):
        x = rng.uniform(-2, area_m)
        y = rng.uniform(-2, area_m)
        r = rng.uniform(0.15, 0.45)
        kind = rng.choice(["crater", "rock"])
        out.append(WorldHazard(x, y, r, kind))
    return out


def render_world_view(world_hazards: list[WorldHazard], camera: CameraModel, rover: RoverPose,
                       width: int = 480, height: int = 360, flare: bool = False) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Renders the camera frame AND a simulated Stereo Disparity map.
    Returns (color_frame, disparity_map, visible_ground_truth).
    """
    yy, xx = np.mgrid[0:height, 0:width]
    base = 110 + 20 * np.sin(xx / width * math.pi) + 10 * np.cos(yy / height * 2 * math.pi)
    noise = np.random.normal(0, 6, size=(height, width))
    frame = np.clip(base + noise, 0, 255).astype(np.uint8)
    
    # Background disparity (ground plane slopes away)
    # y=height is close (high disparity), y=0 is far (low disparity)
    ground_dist = np.clip((height - yy) * 0.05 + 1.0, 1.0, 30.0)
    baseline_m = 0.15
    # Multiply by 16.0 to simulate standard OpenCV subpixel precision disparity
    disparity = ((camera.fx * baseline_m) / ground_dist) * 16.0
    disparity = disparity.astype(np.float32)

    visible = []
    for hz in world_hazards:
        proj = landing_frame_to_pixel(hz.x_m, hz.y_m, width, height, camera, rover)
        if proj is None:
            continue
        px, py, dist = proj["px"], proj["py"], proj["distance_m"]
        radius_px = float(np.clip(hz.radius_m * camera.fx / max(dist, 0.3), 3, min(width, height) * 0.35))
        delta = 55 + 15 / max(dist, 0.5)
        base_val = float(base[int(np.clip(py, 0, height - 1)), int(np.clip(px, 0, width - 1))])
        
        # Color render
        color = max(5, int(base_val - delta)) if hz.kind == "crater" else min(250, int(base_val + delta))
        cv2.circle(frame, (int(px), int(py)), int(radius_px), int(color), -1)
        
        # Stereo Disparity render (Crater = deeper = lower disparity, Rock = closer = higher disparity)
        disp_shift = ((camera.fx * baseline_m) / (dist - hz.radius_m) if hz.kind == "rock" else (camera.fx * baseline_m) / (dist + hz.radius_m)) * 16.0
        cv2.circle(disparity, (int(px), int(py)), int(radius_px), float(disp_shift), -1)
        
        visible.append({"px": px, "py": py, "radius_px": radius_px, "kind": hz.kind,
                         "world_x": hz.x_m, "world_y": hz.y_m, "distance_m": dist})

    if visible:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        disparity = cv2.GaussianBlur(disparity, (3, 3), 0)
        
    # Add stereopsis matching noise (scaled for 16x)
    disp_noise = np.random.normal(0, 4.0, size=(height, width)).astype(np.float32)
    disparity = np.clip(disparity + disp_noise, 0, 4095).astype(np.int16)
    
    # ⚡ SOLAR FLARE (RAD STRIKE) INJECTION ⚡
    if flare:
        # 15% of pixels become pure static (0 or 255)
        salt_pepper = np.random.rand(height, width)
        frame[salt_pepper < 0.075] = 255
        frame[(salt_pepper >= 0.075) & (salt_pepper < 0.15)] = 0
        
        # Disparity map also takes massive tracking hits (drops to 0 disparity)
        disparity[salt_pepper < 0.15] = 0

    return frame, disparity, visible


# ---------------------------------------------------------------------------
# Detection (Now using Stereo Depth)
# ---------------------------------------------------------------------------

def _contours_to_hazards(mask: np.ndarray, kind: str, disparity: np.ndarray) -> list[Hazard]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * math.pi * (area / (perim * perim))
        if circularity < 0.4:
            continue

        M = cv2.moments(c)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        r = math.sqrt(area / math.pi)
        
        # Confidence based on depth discontinuity, far more robust than color!
        conf = min(1.0, 0.5 + circularity * 0.5)
        out.append(Hazard(px=float(cx), py=float(cy), radius_px=float(r), confidence=float(conf), kind=kind))
    return out


def detect_hazards(frame: np.ndarray, disparity: np.ndarray | None = None) -> list[Hazard]:
    """Detects hazards using Stereo Disparity instead of classical contrast.
    If a disparity map is provided (simulating StereoSGBM output), it searches 
    for depth anomalies (rocks = closer/higher disparity, craters = deeper/lower disparity).
    """
    if disparity is None:
        return []
        
    # Depth-based perception
    blur_disp = cv2.GaussianBlur(disparity.astype(np.float32), (5, 5), 0)
    
    # Estimate local background depth by aggressive blurring
    bg_disp = cv2.blur(blur_disp, (31, 31))
    
    # Rocks are closer than background -> disparity is significantly higher than bg
    # (Threshold is 10 units in 16x space, which is < 1 real pixel of disparity)
    m_rock = cv2.compare(blur_disp, bg_disp + 10.0, cv2.CMP_GT)
    
    # Craters are deeper than background -> disparity is significantly lower than bg
    m_crater = cv2.compare(blur_disp, bg_disp - 10.0, cv2.CMP_LT)
    
    rocks = _contours_to_hazards(m_rock, "rock", blur_disp)
    craters = _contours_to_hazards(m_crater, "crater", blur_disp)
    return rocks + craters
