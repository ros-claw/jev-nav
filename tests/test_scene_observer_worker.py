from types import SimpleNamespace

import numpy as np
import pytest

from limo_ros_mcp.workers import limo_scene_observer_worker as worker


def test_request_permits_only_bounded_fixed_topic_capture():
    validate = getattr(worker, "validate_request", None)
    assert callable(validate), "scene capture request validation is unavailable"
    assert validate({"protocol": "rosclaw.limo.scene-observer.v1", "timeout_sec": 5}) == 5
    for request in (
        {"protocol": "rosclaw.limo.scene-observer.v1", "timeout_sec": 11},
        {"protocol": "rosclaw.limo.scene-observer.v1", "timeout_sec": True},
        {"protocol": "rosclaw.limo.scene-observer.v1", "timeout_sec": 5, "topic": "/cmd_vel"},
    ):
        with pytest.raises(ValueError):
            validate(request)


def test_depth_columns_preserve_invalid_samples_as_unknown():
    summarize = getattr(worker, "depth_sectors", None)
    assert callable(summarize), "scene depth summarization is unavailable"
    array = np.array([[1000, 0, 2000, 0, 0, 0]], dtype=np.uint16)
    result = summarize(array, "16UC1")
    assert result["left"] == {"minimum_m": pytest.approx(1.0), "valid_ratio": 0.5}
    assert result["center"] == {"minimum_m": pytest.approx(2.0), "valid_ratio": 0.5}
    assert result["right"] == {"minimum_m": None, "valid_ratio": 0.0}
    with pytest.raises(ValueError):
        summarize(array, "unsupported")


def test_depth_bands_distinguish_lower_pixels_from_forward_middle():
    summarize = getattr(worker, "depth_bands", None)
    assert callable(summarize), "depth image bands are unavailable"
    array = np.array([[4000] * 6, [3000] * 6, [500] * 6], dtype=np.uint16)
    bands = summarize(array, "16UC1")
    assert bands["top"]["center"]["minimum_m"] == pytest.approx(4.0)
    assert bands["middle"]["center"]["minimum_m"] == pytest.approx(3.0)
    assert bands["bottom"]["center"]["minimum_m"] == pytest.approx(0.5)


def test_scan_transform_is_tied_to_scan_frame_and_timestamp():
    lookup = getattr(worker, "laser_transform", None)
    assert callable(lookup), "timestamped laser transform is unavailable"
    scan = SimpleNamespace(
        header=SimpleNamespace(frame_id="laser_link", stamp=SimpleNamespace(to_sec=lambda: 100.05))
    )
    listener = SimpleNamespace(
        waitForTransform=lambda *a: None,
        lookupTransform=lambda *a: ((0.105, 0, 0.08), (0, 0, 0, 1)),
    )
    tf = SimpleNamespace(transformations=SimpleNamespace(euler_from_quaternion=lambda q: (0, 0, 0)))
    ros = SimpleNamespace(Duration=lambda timeout: timeout)
    transform = lookup(listener, ros, tf, scan, 0.5)
    assert transform["source"] == "laser_link"
    assert transform["target"] == "base_link"
    assert transform["stamp"] == 100.05
    assert transform["x"] == 0.105
    assert transform["verified"] is False
