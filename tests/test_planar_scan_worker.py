"""Fixed LiDAR-only ROS worker tested with fake scan and TF providers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def worker(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "src/limo_ros_mcp/workers"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "planar_scan_worker", directory / "limo_planar_scan_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_fixed_scan_and_base_transform_are_observed(worker):
    stamp = SimpleNamespace(to_sec=lambda: 100.0)
    scan = SimpleNamespace(
        header=SimpleNamespace(frame_id="laser", stamp=stamp),
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.1,
        range_max=10.0,
        time_increment=0.001,
        scan_time=0.1,
        ranges=[1.0, float("inf")],
    )
    calls = []
    ros = SimpleNamespace(
        wait_for_message=lambda topic, kind, timeout: calls.append(topic) or scan,
        Duration=lambda value: value,
        Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 100.1)),
    )

    class Listener:
        def waitForTransform(self, target, source, at, timeout):  # noqa: N802
            calls.append((target, source, at.to_sec()))

        def lookupTransform(self, target, source, at):  # noqa: N802
            return (0, 0, 0), (0, 0, 0, 1)

    tf = SimpleNamespace(
        TransformListener=Listener,
        transformations=SimpleNamespace(
            quaternion_matrix=lambda q: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
    )
    output = worker.collect(ros, tf, object, timeout=5)
    assert calls == ["/scan", ("base_link", "laser", 100.0)]
    assert output["usable_for_real_execution"] is False
    assert "depth" not in output


def test_protocol_refuses_caller_selected_topic(worker):
    assert worker.validate_request({"protocol": worker.PROTOCOL, "timeout_sec": 5}) == 5
    with pytest.raises(ValueError):
        worker.validate_request(
            {"protocol": worker.PROTOCOL, "timeout_sec": 5, "topic": "/cmd_vel"}
        )


def test_new_scan_is_retried_when_tf_buffer_starts_after_first_stamp(worker):
    stamps = [100.0, 100.1]
    calls = []

    def wait(topic, kind, timeout):
        calls.append(topic)
        value = stamps.pop(0)
        stamp = SimpleNamespace(to_sec=lambda: value)
        return SimpleNamespace(
            header=SimpleNamespace(frame_id="laser", stamp=stamp),
            angle_min=0.0,
            angle_increment=0.1,
            range_min=0.1,
            range_max=10.0,
            time_increment=0.001,
            scan_time=0.1,
            ranges=[1.0],
        )

    ros = SimpleNamespace(
        wait_for_message=wait,
        Duration=lambda value: value,
        Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda: 100.2)),
    )

    class Listener:
        def waitForTransform(self, target, source, stamp, timeout):  # noqa: N802
            if stamp.to_sec() == 100.0:
                raise RuntimeError("TF history starts later")

        def lookupTransform(self, target, source, stamp):  # noqa: N802
            return (0, 0, 0), (0, 0, 0, 1)

    tf = SimpleNamespace(
        TransformListener=Listener,
        transformations=SimpleNamespace(
            quaternion_matrix=lambda q: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
    )
    output = worker.collect(ros, tf, object, timeout=5, clock=lambda: 200.0)
    assert calls == ["/scan", "/scan"]
    assert output["scan"]["stamp"] == 100.1
    assert output["scan"]["transform"]["stamp"] == 100.1
