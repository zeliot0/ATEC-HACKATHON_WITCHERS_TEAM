import math
import random

from src.localization import CameraModel, RoverPose, pixel_to_landing_frame, landing_frame_to_pixel

IMAGE_W, IMAGE_H = 480, 360


def _camera(pitch_deg=30.0, height_m=1.5):
    return CameraModel(fx=400, fy=400, cx=IMAGE_W / 2, cy=IMAGE_H / 2, height_m=height_m, pitch_deg=pitch_deg)


def test_center_pixel_matches_closed_form_distance():
    cam = _camera(pitch_deg=30.0, height_m=1.5)
    rover = RoverPose(x_m=0, y_m=0, heading_deg=0)
    res = pixel_to_landing_frame(IMAGE_W / 2, IMAGE_H / 2, IMAGE_W, IMAGE_H, cam, rover)
    expected = 1.5 / math.tan(math.radians(30.0))
    assert abs(res["distance_m"] - expected) < 0.01
    assert abs(res["x"]) < 0.01
    assert res["y"] > 0


def test_heading_rotates_correctly_compass_convention():
    cam = _camera()
    rover = RoverPose(x_m=5, y_m=5, heading_deg=90)  # facing east
    res = pixel_to_landing_frame(IMAGE_W / 2, IMAGE_H / 2, IMAGE_W, IMAGE_H, cam, rover)
    assert res["x"] > rover.x_m
    assert abs(res["y"] - rover.y_m) < 0.05


def test_left_pixel_projects_left_of_forward():
    cam = _camera()
    rover = RoverPose(x_m=0, y_m=0, heading_deg=0)
    res = pixel_to_landing_frame(IMAGE_W / 2 - 140, IMAGE_H / 2, IMAGE_W, IMAGE_H, cam, rover)
    assert res["x"] < 0


def test_forward_inverse_projection_round_trip():
    """landing_frame_to_pixel must be the exact inverse of pixel_to_landing_frame
    -- this is what keeps the same physical hazard mapping to the same world
    position across frames as the rover moves, instead of appearing to jump
    to a new location (and a new tracker id) every frame."""
    cam = _camera()
    rng = random.Random(0)
    max_err = 0.0
    for _ in range(100):
        rover = RoverPose(x_m=rng.uniform(-10, 10), y_m=rng.uniform(-10, 10), heading_deg=rng.uniform(0, 360))
        px0, py0 = rng.uniform(20, IMAGE_W - 20), rng.uniform(20, IMAGE_H - 20)
        try:
            loc = pixel_to_landing_frame(px0, py0, IMAGE_W, IMAGE_H, cam, rover)
        except Exception:
            continue
        proj = landing_frame_to_pixel(loc["x"], loc["y"], IMAGE_W, IMAGE_H, cam, rover)
        assert proj is not None
        err = math.hypot(proj["px"] - px0, proj["py"] - py0)
        max_err = max(max_err, err)
    assert max_err < 1e-6


def test_point_behind_camera_or_out_of_frame_returns_none():
    cam = _camera()
    rover = RoverPose(x_m=0, y_m=0, heading_deg=0)
    # a point directly behind the rover should not be visible
    assert landing_frame_to_pixel(0, -5, IMAGE_W, IMAGE_H, cam, rover) is None


def test_uncertainty_grows_with_distance():
    # Localization now reports a covariance ellipse (sx = lateral, sy = depth-
    # along-ray uncertainty) instead of a single scalar -- sy is the one that
    # should grow with distance, since depth uncertainty from a monocular ray
    # scales with distance^2 (see pixel_to_landing_frame's docstring/comment).
    cam_shallow = _camera(pitch_deg=10.0)
    cam_steep = _camera(pitch_deg=30.0)
    rover = RoverPose(x_m=0, y_m=0, heading_deg=0)
    far = pixel_to_landing_frame(IMAGE_W / 2, IMAGE_H / 2, IMAGE_W, IMAGE_H, cam_shallow, rover)
    near = pixel_to_landing_frame(IMAGE_W / 2, IMAGE_H / 2 + 120, IMAGE_W, IMAGE_H, cam_steep, rover)
    assert far["distance_m"] > near["distance_m"]
    assert far["sy"] > near["sy"]
