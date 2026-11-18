"""The diagnostic ROS worker is exercised exclusively with synthetic streams."""

import importlib.util
from pathlib import Path

import pytest

from tests.test_local_observation_worker import fake_ros


@pytest.fixture
def worker(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "src/limo_ros_mcp/workers"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "calibration_worker", directory / "limo_calibration_diagnostic_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_uses_fixed_synchronized_topics_and_timestamped_tf(worker):
    ros, tf, filters, seen, closed = fake_ros()
    original = filters.ApproximateTimeSynchronizer

    class Synchronizer(original):
        def registerCallback(self, callback):  # noqa: N802
            def patched(scan, image, info):
                scan.time_increment, scan.scan_time = 0.001, 0.1
                image.header.frame_id = "camera_color_optical_frame"
                info.header.frame_id = ""
                info.distortion_model = "plumb_bob"
                callback(scan, image, info)

            return super().registerCallback(patched)

    filters.ApproximateTimeSynchronizer = Synchronizer
    calls = []

    class Listener:
        def waitForTransform(self, target, source, stamp, timeout):  # noqa: N802
            calls.append((target, source, stamp.to_sec()))

        def lookupTransform(self, target, source, stamp):  # noqa: N802
            return (0, 0, 0), (0, 0, 0, 1)

    tf.TransformListener = Listener
    result = worker.collect(ros, tf, filters, object, object, object, timeout=5)
    assert seen == ["/scan", "/camera/depth/image_raw", "/camera/depth/camera_info"]
    assert closed == seen
    assert calls == [("camera_color_optical_frame", "laser", 100.0), ("base_link", "laser", 100.0)]
    assert result["camera_info"]["frame_id"] == ""
    assert result["base_transform"]["target"] == "base_link"
    assert result["diagnostic_only"] is True
    assert result["usable_for_real_execution"] is False


def test_protocol_cannot_select_topics_or_change_robot(worker):
    assert worker.validate_request({"protocol": worker.PROTOCOL, "timeout_sec": 5}) == 5
    with pytest.raises(ValueError):
        worker.validate_request(
            {"protocol": worker.PROTOCOL, "timeout_sec": 5, "topic": "/cmd_vel"}
        )
