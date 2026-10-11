"""
Spatial localization for Project C.R.A.T.E.R.

Converts a 2D pixel detection into a 3D coordinate relative to the landing
site -- the exact chain a real rover's nav stack uses (camera -> body ->
world), just hand-built with numpy instead of a framework like tf2, so
every step can be shown and defended on a whiteboard:

    1. Pixel -> camera-frame ray (pinhole intrinsics)
    2. Camera-frame ray -> rover body frame (fixed mounting rotation)
    3. Body frame -> world frame (rover pitch is baked into mounting; rover
       yaw/heading rotates body into world)
    4. Intersect the ray with the ground plane (z=0) to get a 3D point
    5. Translate by the rover's own world position -> landing-site frame

A first-order uncertainty estimate is attached to every point: pinhole
projection is far more sensitive to angular error at long range and
shallow viewing angles (a 1-pixel error near the horizon swings the ground
point by meters; the same error near the rover's feet swings it by
millimeters). This is a deliberately simple linearization, not a full
covariance/EKF treatment -- that's noted as future work rather than
overclaimed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float       # camera height above ground plane
    pitch_deg: float      # downward tilt from horizontal, positive = looking down


@dataclass
class RoverPose:
    x_m: float
    y_m: float
    heading_deg: float    # yaw, 0 = facing +y (landing-site "north")


def _rot_x(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])


def _rot_z(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])


# Fixed basis change: camera optical frame (x=right, y=down, z=forward)
# expressed in rover body frame (x=right, y=forward, z=up).
_CAM_TO_BODY = np.array([
    [1, 0, 0],   # body x <- cam x
    [0, 0, 1],   # body y <- cam z (forward)
    [0, -1, 0],  # body z <- -cam y (image-down is body-down)
], dtype=float)


class LocalizationError(ValueError):
    pass


def pixel_to_landing_frame(px: float, py: float, image_w: int, image_h: int,
                            camera: CameraModel, rover: RoverPose) -> dict:
    """Projects a pixel onto the ground plane and returns landing-site-frame
    coordinates plus a first-order uncertainty estimate (meters, 1-sigma-ish)."""
    xn = (px - camera.cx) / camera.fx
    yn = (py - camera.cy) / camera.fy
    ray_cam = np.array([xn, yn, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    ray_body = _CAM_TO_BODY @ ray_cam

    pitch_rad = math.radians(camera.pitch_deg)
    # tilt the body-frame ray downward by pitch_deg: rotate about body x-axis
    ray_body_tilted = _rot_x(-pitch_rad) @ ray_body

    # Compass convention: heading 0 = facing +y ("north"), heading increases
    # clockwise toward +x ("east") -- hence the negated angle, since rot_z
    # as defined is the standard counter-clockwise (right-hand) rotation.
    heading_rad = math.radians(rover.heading_deg)
    ray_world = _rot_z(-heading_rad) @ ray_body_tilted

    if ray_world[2] >= -1e-6:
        raise LocalizationError("ray does not point toward the ground plane (horizon or above)")

    origin = np.array([rover.x_m, rover.y_m, camera.height_m])
    t = -origin[2] / ray_world[2]
    point = origin + t * ray_world

    distance_m = float(np.hypot(point[0] - rover.x_m, point[1] - rover.y_m))

    # Instead of scalar uncertainty, we calculate a Covariance Ellipse.
    # Uncertainty along the ray (depth) is much larger than across the ray (lateral).
    pixel_angle_err = 1.0 / camera.fx
    # Depth uncertainty: derivative of ground distance w.r.t ray angle
    sy = (distance_m ** 2) * pixel_angle_err / max(camera.height_m, 0.05)
    sy = float(max(0.05, min(sy, 50.0)))
    
    # Lateral uncertainty: distance * tan(angle_err)
    sx = distance_m * math.tan(pixel_angle_err)
    sx = float(max(0.05, min(sx, 50.0)))
    
    # The ellipse is oriented along the ray's heading from the rover
    ray_heading_rad = math.atan2(point[1] - rover.y_m, point[0] - rover.x_m)
    ellipse_angle_deg = math.degrees(ray_heading_rad)

    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "distance_m": distance_m,
        "sx": sx,
        "sy": sy,
        "angle": ellipse_angle_deg,
    }


def landing_frame_to_pixel(x_m: float, y_m: float, image_w: int, image_h: int,
                            camera: CameraModel, rover: RoverPose) -> dict | None:
    """Inverse of pixel_to_landing_frame: given a world-frame point, returns
    the pixel it projects to from the rover's current pose, or None if the
    point is out of frame or behind the camera.

    This is what lets the pipeline render a camera frame that is actually
    consistent with where the rover is standing, instead of an independent
    random image each frame -- without it, the same physical hazard would
    appear to "teleport" every frame as the rover moves, and would never be
    recognized as the same hazard twice. It is the exact algebraic inverse
    of pixel_to_landing_frame (every rotation undone in reverse order).
    """
    vector_world = np.array([x_m - rover.x_m, y_m - rover.y_m, -camera.height_m])

    heading_rad = math.radians(rover.heading_deg)
    v1 = _rot_z(heading_rad) @ vector_world  # undo _rot_z(-heading_rad)

    pitch_rad = math.radians(camera.pitch_deg)
    v2 = _rot_x(pitch_rad) @ v1              # undo _rot_x(-pitch_rad)

    ray_cam = _CAM_TO_BODY.T @ v2             # undo _CAM_TO_BODY (orthogonal -> transpose = inverse)

    if ray_cam[2] <= 1e-6:
        return None  # point is behind the camera

    xn = ray_cam[0] / ray_cam[2]
    yn = ray_cam[1] / ray_cam[2]
    px = xn * camera.fx + camera.cx
    py = yn * camera.fy + camera.cy

    if not (0 <= px < image_w and 0 <= py < image_h):
        return None

    distance_m = float(np.hypot(x_m - rover.x_m, y_m - rover.y_m))
    return {"px": float(px), "py": float(py), "distance_m": distance_m}
