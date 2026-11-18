import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "worker" / "limo_mapless_worker.py"
spec = importlib.util.spec_from_file_location("limo_mapless_worker", path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def request():
    return {
        "protocol": "rosclaw.limo.worker.v1",
        "operation": "MAPLESS_EXPLORE",
        "schema_version": "limo.explore.v1",
        "action_id": "explore-1",
        "body_id": "limo",
        "body_snapshot_hash": "a" * 64,
        "duration_sec": 10.0,
        "daemon_pid": 123,
    }


def test_worker_accepts_only_exact_bounded_exploration_contract():
    assert worker.validate_request(request())["duration_sec"] == 10.0
    for extra in ({"duration_sec": 31}, {"body_id": "other"}, {"unexpected": 1}, {"daemon_pid": 1}):
        proposal = request()
        proposal.update(extra)
        with pytest.raises(ValueError):
            worker.validate_request(proposal)


def test_reparented_worker_refuses_to_initialize_ros(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(worker, "os", SimpleNamespace(getppid=lambda: 1))
    with pytest.raises(ValueError, match="parent"):
        worker.execute(request())


def test_laser_endpoints_are_transformed_before_direction_sectors():
    scan = worker.summarize_scan([3.0] * 360, -math.pi, math.pi / 180, 0.1, 12.0, 0.1, 0.0, 0.0)
    assert 3.08 < scan["front"] <= 3.1
    assert scan["left"] is not None and scan["right"] is not None


def test_laser_sparse_or_rear_obstacle_observations_are_not_treated_as_clear():
    ranges = [float("nan")] * 360
    ranges[180] = 3.0
    sparse = worker.summarize_scan(ranges, -math.pi, math.pi / 180, 0.1, 12.0, 0.0, 0.0, 0.0)
    assert sparse["front"] is None
    ranges = [3.0] * 360
    ranges[0] = 0.2
    obstacle = worker.summarize_scan(ranges, -math.pi, math.pi / 180, 0.1, 12.0, 0.0, 0.0, 0.0)
    assert obstacle["rear"] == pytest.approx(0.2)


def test_depth_frame_and_camera_info_must_match():
    image = {"frame_id": "color_optical", "width": 640, "height": 400, "stamp": 10.0}
    info = {"frame_id": "color_optical", "width": 640, "height": 400, "stamp": 10.05}
    assert worker.depth_metadata_matches(image, info)
    for patch in ({"frame_id": ""}, {"width": 320}, {"stamp": 11.0}):
        invalid = dict(info)
        invalid.update(patch)
        assert not worker.depth_metadata_matches(image, invalid)


def test_stop_requires_multiple_new_fresh_odometry_samples():
    verifier = worker.StoppedOdomVerifier(stop_requested_at=10.0, stop_requested_ros=100.0)
    assert not verifier.observe(
        0.0, 0.0, 0.0, observed_at=10.1, now=10.1, source_stamp=99.0, source_now=100.1
    )
    assert not verifier.observe(
        0.0, 0.0, 0.0, observed_at=10.1, now=10.1, source_stamp=100.1, source_now=100.1
    )
    assert not verifier.observe(
        0.1, 0.0, 0.0, observed_at=10.2, now=10.2, source_stamp=100.2, source_now=100.2
    )
    assert not verifier.observe(
        0.0, 0.0, 0.0, observed_at=10.3, now=10.3, source_stamp=100.3, source_now=100.3
    )
    assert not verifier.observe(
        0.0, 0.0, 0.0, observed_at=10.4, now=10.4, source_stamp=100.4, source_now=100.4
    )
    assert verifier.observe(
        0.0, 0.0, 0.0, observed_at=10.5, now=10.5, source_stamp=100.5, source_now=100.5
    )


def test_motion_evidence_counts_translation_and_wrapped_yaw():
    assert worker.motion_delta((0.0, 0.0, 3.13), (0.03, 0.04, -3.13))[0] == pytest.approx(0.05)
    assert worker.motion_delta((0.0, 0.0, 3.13), (0.0, 0.0, -3.13))[1] == pytest.approx(
        0.0231853, abs=1e-6
    )


def test_motion_evidence_ignores_poses_before_a_relay_acknowledgement():
    evidence = worker.MotionEvidence()
    evidence.observe((0.0, 0.0, 0.0))
    evidence.observe((1.0, 0.0, 0.0))
    evidence.acknowledge((1.0, 0.0, 0.0))
    evidence.observe((1.0, 0.0, 0.0))
    assert evidence.max_translation == 0.0
    evidence.observe((1.05, 0.0, 0.0))
    assert evidence.max_translation == pytest.approx(0.05)


def test_stop_evidence_requires_a_time_span_not_just_a_burst():
    verifier = worker.StoppedOdomVerifier(stop_requested_at=10.0, stop_requested_ros=100.0)
    for step in (0.01, 0.02, 0.03):
        assert not verifier.observe(
            0.0,
            0.0,
            0.0,
            observed_at=10.0 + step,
            now=10.0 + step,
            source_stamp=100.0 + step,
            source_now=100.0 + step,
        )
    assert verifier.observe(
        0.0, 0.0, 0.0, observed_at=10.22, now=10.22, source_stamp=100.22, source_now=100.22
    )


def test_delayed_or_incoherent_sensor_source_times_are_rejected():
    assert worker.sensor_stamps_current(
        {"scan": 100.0, "odom": 100.0, "status": 99.8, "depth": 100.0, "color": 100.0},
        ros_now=100.1,
    )
    assert not worker.sensor_stamps_current(
        {"scan": 98.7, "odom": 100.0, "status": 99.8, "depth": 100.0, "color": 100.0}, ros_now=100.1
    )
    assert not worker.sensor_stamps_current(
        {"scan": 100.0, "odom": 100.0, "status": 99.0, "depth": 100.0, "color": 100.0},
        ros_now=100.1,
    )


def test_model_direction_is_rejected_after_large_pose_change():
    assert worker.decision_pose_current((0.0, 0.0, 0.0), (0.05, 0.0, 0.1))
    assert not worker.decision_pose_current((0.0, 0.0, 0.0), (0.2, 0.0, 0.0))
    assert not worker.decision_pose_current((0.0, 0.0, 0.0), (0.0, 0.0, 1.25))


def test_hung_decision_process_is_terminated_after_its_total_deadline():
    started = time.monotonic()
    response = worker.run_decision_process(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        {},
        cancelled=lambda: False,
        clock=time.monotonic,
        timeout=0.2,
    )
    assert response is None
    assert time.monotonic() - started < 2.0


def test_depth_obstacle_outside_center_columns_is_not_discarded():
    image = np.ones((4, 6), dtype=np.float32)
    image[1, 0] = 0.15
    minimum, ratio = worker.summarize_depth(image, "32FC1")
    assert minimum == pytest.approx(0.15)
    assert ratio == 1.0
