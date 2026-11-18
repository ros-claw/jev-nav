"""Exercise the fixed-topic ROS adapter using fakes only."""

import importlib.util
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def worker(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "src/limo_ros_mcp/workers"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "local_observation_worker", directory / "limo_local_observation_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_ros(*, mismatch=False, bad_endian=False, produce=True):
    def header(frame):
        return SimpleNamespace(frame_id=frame, stamp=SimpleNamespace(to_sec=lambda: 100.0))

    scan = SimpleNamespace(
        header=header("laser"),
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.1,
        range_max=10.0,
        ranges=[1.0, float("inf")],
    )
    depth = SimpleNamespace(
        header=header("optical"),
        width=3,
        height=2,
        step=6,
        encoding="16UC1",
        is_bigendian=2 if bad_endian else 0,
        data=struct.pack("<6H", 0, 1000, 0, 0, 0, 0),
    )
    info = SimpleNamespace(
        header=header("wrong" if mismatch else "optical"),
        width=3,
        height=2,
        K=[1, 0, 1, 0, 1, 1, 0, 0, 1],
        D=[0, 0, 0, 0, 0],
        R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        P=[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        binning_x=0,
        binning_y=0,
        roi=SimpleNamespace(x_offset=0, y_offset=0, width=0, height=0, do_rectify=False),
    )
    messages = {"/scan": scan, "/camera/depth/image_raw": depth, "/camera/depth/camera_info": info}
    seen = []

    closed = []

    class Subscriber:
        def __init__(self, topic, kind):
            seen.append(topic)
            self.topic = topic
            self.sub = self

        def unregister(self):
            closed.append(self.topic)

    class Synchronizer:
        def __init__(self, subscribers, queue_size, slop):
            assert queue_size == 10 and slop == 0.1
            self.subscribers = subscribers

        def registerCallback(self, callback):  # noqa: N802 - ROS API
            if produce:
                callback(*(messages[sub.topic] for sub in self.subscribers))

    ros = SimpleNamespace(
        Duration=lambda x: x,
        Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 100.1)),
    )

    class Listener:
        def waitForTransform(self, target, source, stamp, timeout):  # noqa: N802 - ROS API
            assert target == "base_link" and source in {"laser", "optical"}

        def lookupTransform(self, target, source, stamp):  # noqa: N802 - ROS API
            return ((0, 0, 0), (0, 0, 0, 1))

    tf = SimpleNamespace(
        TransformListener=Listener,
        transformations=SimpleNamespace(
            quaternion_matrix=lambda q: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
    )
    filters = SimpleNamespace(Subscriber=Subscriber, ApproximateTimeSynchronizer=Synchronizer)
    return ros, tf, filters, seen, closed


def test_collect_uses_only_fixed_read_only_topics(worker):
    ros, tf, filters, seen, closed = fake_ros()
    bundle = worker.collect(ros, tf, filters, object, object, object, timeout=5)
    assert seen == ["/scan", "/camera/depth/image_raw", "/camera/depth/camera_info"]
    assert bundle["scan"]["frame_id"] == "laser"
    assert bundle["depth"]["frame_id"] == "optical"
    assert bundle["usable_for_real_execution"] is False
    assert closed == seen


def test_collect_rejects_mismatched_camera_frame(worker):
    ros, tf, filters, _, closed = fake_ros(mismatch=True)
    with pytest.raises(ValueError, match="mismatch") as exc:
        worker.collect(ros, tf, filters, object, object, object, timeout=5)
    assert "optical" in str(exc.value)
    assert "wrong" in str(exc.value)
    assert "3x2" in str(exc.value)
    assert "rectification_max_error=0" in str(exc.value)
    assert "K(fx,fy,cx,cy)=[1,1,1,1]" in str(exc.value)
    assert "P(fx,fy,cx,cy)=[1,1,1,1]" in str(exc.value)
    assert len(closed) == 3


def test_collect_rejects_invalid_endian_flag(worker):
    ros, tf, filters, _, _ = fake_ros(bad_endian=True)
    with pytest.raises(ValueError, match="endian"):
        worker.collect(ros, tf, filters, object, object, object, timeout=5)


def test_collect_timeout_releases_all_subscriptions(worker):
    ros, tf, filters, seen, closed = fake_ros(produce=False)
    with pytest.raises(ValueError, match="timed out"):
        worker.collect(ros, tf, filters, object, object, object, timeout=0.1)
    assert closed == seen


def test_calibration_diagnostic_does_not_accept_nonfinite_values(worker):
    info = SimpleNamespace(R=[float("nan")] * 9, K=[1] * 9, P=[1] * 12)
    assert worker._calibration_diagnostic(info) == "calibration_diagnostic=unavailable"


def test_worker_request_has_no_arbitrary_ros_topic_or_command(worker):
    with pytest.raises(ValueError):
        worker.validate_request(
            {"protocol": worker.PROTOCOL, "timeout_sec": 5, "topic": "/cmd_vel"}
        )
    assert worker.validate_request({"protocol": worker.PROTOCOL, "timeout_sec": 5}) == 5
