"""Synchronized scene evidence remains advisory and rejects stale inputs."""

import base64
import math

import pytest

from limo_ros_mcp.workers import scene_observation_core as scene

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def capture():
    return {
        "stamps": {
            "color": 100.0,
            "scan": 100.05,
            "status": 100.10,
            "odom": 100.08,
            "depth": 100.06,
        },
        "scan": {
            "angle_min": -math.pi / 2,
            "angle_increment": math.pi / 4,
            "range_min": 0.1,
            "range_max": 12.0,
            "ranges": [2, 3, 4, 5, 6],
            "time_increment": 0.001,
            "scan_time": 0.1,
        },
        "frames": {
            "laser": "laser_link",
            "color": "camera_color_optical_frame",
            "depth": "camera_depth_optical_frame",
        },
        "laser_tf": {
            "source": "laser_link",
            "target": "base_link",
            "stamp": 100.05,
            "x": 0.105,
            "y": 0.0,
            "yaw": 0.0,
            "verified": False,
        },
        "base": {
            "error_code": 0,
            "motion_mode": 0,
            "battery_voltage": 12.1,
            "linear_x_mps": 0.0,
            "angular_z_radps": 0.0,
        },
        "depth": {
            name: {"minimum_m": 2.0, "valid_ratio": 0.5} for name in ("left", "center", "right")
        },
    }


def build(data, now=100.15):
    function = getattr(scene, "make_snapshot", None)
    assert callable(function), "synchronized scene assembly is unavailable"
    return function(data, now, now)


def test_forward_scene_preserves_sensor_times_and_unknown_rear():
    snapshot = build(capture())
    assert snapshot["laser"]["front_min_m"] == 4
    assert snapshot["laser"]["right_min_m"] == 2
    assert snapshot["laser"]["left_min_m"] == 5
    assert snapshot["laser"]["rear_min_m"] is None
    assert snapshot["image_stamp"] == 100.0
    assert snapshot["depth"]["stamp"] == 100.06
    assert snapshot["depth"]["distance_measured"] is True
    assert snapshot["laser"]["forward_view"]["center"]["total_rays"] == 1
    assert snapshot["laser"]["forward_view"]["center"]["max_unobserved_angle_rad"] >= math.pi / 6
    assert snapshot["laser"]["forward_view_stamp"] == 100.05
    assert snapshot["sensor_frames"]["laser"] == "laser_link"
    assert snapshot["sensor_frames"]["transform_verified"] is False
    assert snapshot["sensor_frames"]["laser_to_base"]["x_m"] == pytest.approx(0.105)
    assert snapshot["laser"]["scan_end_stamp"] == pytest.approx(100.054)
    assert snapshot["laser"]["deskewed"] is False
    assert snapshot["laser"]["scan_timing"] == "known"


def test_missing_lidar_returns_remain_unknown():
    data = capture()
    data["scan"]["ranges"] = [0.0, float("nan"), 4, 0, float("inf")]
    snapshot = build(data)
    assert snapshot["laser"]["right_min_m"] is None
    assert snapshot["laser"]["left_min_m"] is None
    assert snapshot["laser"]["usable_ratio"] == pytest.approx(0.2)


@pytest.mark.parametrize("stamp,now", [(99.0, 100.15), (100.5, 100.15), (float("nan"), 100.15)])
def test_old_future_or_invalid_stamp_rejects_entire_scene(stamp, now):
    data = capture()
    data["stamps"]["color"] = stamp
    with pytest.raises(ValueError):
        build(data, now)


def test_depth_invalid_values_are_not_promoted_to_distance_evidence():
    data = capture()
    data["depth"]["center"] = {"minimum_m": 0, "valid_ratio": 0.5}
    with pytest.raises(ValueError):
        build(data)


def test_missing_depth_is_available_but_not_a_distance_measurement():
    data = capture()
    data["depth"] = {
        name: {"minimum_m": None, "valid_ratio": 0.0} for name in ("left", "center", "right")
    }
    snapshot = build(data)
    assert snapshot["depth"]["available"] is True
    assert snapshot["depth"]["distance_measured"] is False


def test_zero_header_times_are_not_synchronized_sensor_evidence():
    data = capture()
    data["stamps"] = dict.fromkeys(data["stamps"], 0)
    data["laser_tf"]["stamp"] = 0
    with pytest.raises(ValueError):
        build(data, now=0)


def test_lidar_yaw_transform_prevents_back_ray_being_labeled_front():
    data = capture()
    data["laser_tf"]["yaw"] = math.pi
    snapshot = build(data)
    assert snapshot["laser"]["front_min_m"] is None
    assert snapshot["laser"]["rear_min_m"] == 3


def test_wrong_transform_frame_or_timestamp_is_rejected():
    data = capture()
    data["laser_tf"]["source"] = "some_other_frame"
    with pytest.raises(ValueError):
        build(data)


@pytest.mark.parametrize("frame", [" ", "/", "camera frame"])
def test_unusable_camera_frame_names_are_rejected(frame):
    data = capture()
    data["frames"]["color"] = frame
    with pytest.raises(ValueError):
        build(data)


def test_ros_clock_and_wall_clock_mismatch_is_rejected():
    data = capture()
    with pytest.raises(ValueError):
        scene.make_snapshot(data, 100.15, 1000.0)


def test_scan_end_in_future_is_rejected_and_zero_increment_is_unknown():
    data = capture()
    data["scan"]["time_increment"] = 1.0
    with pytest.raises(ValueError):
        build(data)
    data["scan"]["time_increment"] = 0
    snapshot = build(data)
    assert snapshot["laser"]["scan_timing"] == "unknown"
    assert snapshot["laser"]["scan_end_stamp"] is None


def test_snapshot_validator_rejects_corrupt_coverage_and_sensor_transform():
    snapshot = build(capture())
    snapshot["laser"]["forward_view"]["center"]["valid_rays"] = 1000
    with pytest.raises(ValueError):
        scene.validate_snapshot(snapshot)


def test_snapshot_validator_rejects_hits_in_unseen_directions():
    data = capture()
    data["scan"]["ranges"] = [2, 0, 0, 0, 0]
    snapshot = build(data)
    assert snapshot["laser"]["forward_view"]["center"]["valid_rays"] == 0
    snapshot["laser"]["front_min_m"] = 0.8
    with pytest.raises(ValueError):
        scene.validate_snapshot(snapshot)
    snapshot = build(data)
    snapshot["laser"]["forward_view"]["base_hits"]["right"] = {
        "nearest_m": None,
        "bearing_rad": None,
    }
    with pytest.raises(ValueError):
        scene.validate_snapshot(snapshot)
    snapshot = build(capture())
    snapshot["sensor_frames"]["laser_to_base"]["yaw_rad"] = float("nan")
    with pytest.raises(ValueError):
        scene.validate_snapshot(snapshot)
    snapshot = build(capture())
    snapshot["laser"]["forward_view"]["rear_valid_rays"] = 1
    with pytest.raises(ValueError):
        scene.validate_snapshot(snapshot)
    snapshot = build(capture())
    snapshot["laser"]["forward_view"]["base_hits"]["center"]["nearest_m"] = float("nan")
    with pytest.raises(ValueError):
        scene.validate_snapshot(snapshot)
    snapshot = build(capture())
    center = snapshot["laser"]["forward_view"]["center"]
    center.update(total_rays=0, valid_rays=0, max_unobserved_angle_rad=0, nearest_to_base_m=0.2)
    with pytest.raises(ValueError):
        scene.validate_snapshot(snapshot)
    snapshot = build(capture())
    for name in ("right", "center", "left"):
        snapshot["laser"]["forward_view"][name]["total_rays"] = 5
    with pytest.raises(ValueError):
        scene.validate_snapshot(snapshot)
    snapshot = build(capture())
    snapshot["laser"]["forward_view"]["base_hits"]["center"]["bearing_rad"] = 2.5
    with pytest.raises(ValueError):
        scene.validate_snapshot(snapshot)
    data = capture()
    data["scan"]["ranges"] = [0, 0, 0, 0, 0]
    snapshot = build(data)
    snapshot["laser"]["forward_view"]["base_hits"]["center"] = {"nearest_m": 0.2, "bearing_rad": 0}
    with pytest.raises(ValueError):
        scene.validate_snapshot(snapshot)


def test_snapshot_validator_rejects_old_complete_rounds():
    snapshot = build(capture())
    with pytest.raises(ValueError):
        scene.validate_snapshot(snapshot, now=110.0)


def test_scene_digest_binds_the_image_and_snapshot_together():
    digest = getattr(scene, "scene_digest", None)
    assert callable(digest), "scene binding digest is unavailable"
    snapshot = build(capture())
    first = digest(b"png-one", snapshot)
    assert first.startswith("sha256:")
    assert digest(b"png-two", snapshot) != first
    snapshot["image_stamp"] += 0.01
    assert digest(b"png-one", snapshot) != first


def test_png_validation_rejects_truncated_or_corrupt_image_data():
    validate = getattr(scene, "validate_png", None)
    assert callable(validate), "PNG structure validation is unavailable"
    assert validate(PNG) is True
    for png in (b"\x89PNG\r\n\x1a\nimage", PNG[:-1], PNG[:-5] + b"xxxxx"):
        with pytest.raises(ValueError):
            validate(png)


def test_depth_bands_are_preserved_without_claiming_ground_clearance():
    data = capture()
    data["depth_bands"] = {
        band: {name: {"minimum_m": 2.0, "valid_ratio": 0.5} for name in ("left", "center", "right")}
        for band in ("top", "middle", "bottom")
    }
    data["depth_bands"]["bottom"]["center"]["minimum_m"] = 0.5
    snapshot = build(data)
    assert snapshot["depth"]["bands"]["bottom"]["center"]["minimum_m"] == 0.5
    assert snapshot["depth"]["bands"]["middle"]["center"]["minimum_m"] == 2
    data["depth_bands"]["middle"]["center"]["minimum_m"] = -1
    with pytest.raises(ValueError):
        build(data)
