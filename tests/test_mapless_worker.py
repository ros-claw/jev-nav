import importlib.util
import io
import math
import os
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


def test_forward_only_lidar_reports_front_directions_and_unknown_rear():
    ranges = [2.0] * 181
    ranges[30] = 0.4
    ranges[90:95] = [0.0] * 5
    view = worker.summarize_forward_view(
        ranges, -math.pi / 2, math.pi / 180, 0.1, 12.0, 0.0, 0.0, 0.0
    )
    assert view["right"]["nearest_to_base_m"] == pytest.approx(0.4)
    assert view["left"]["nearest_to_base_m"] == pytest.approx(2.0)
    assert view["center"]["max_invalid_return_run_rad"] == pytest.approx(5 * math.pi / 180)
    assert view["rear_total_rays"] == 0
    assert view["rear_valid_rays"] == 0
    assert view["rear_nearest_to_base_m"] is None


def test_forward_view_distinguishes_missing_coverage_from_invalid_returns():
    partial = worker.summarize_forward_view([2.0] * 81, -0.4, 0.01, 0.1, 12.0, 0, 0, 0)
    assert partial["left"]["total_rays"] == 0
    assert partial["left"]["max_invalid_return_run_rad"] is None
    full = worker.summarize_forward_view([0.0] * 360, -math.pi, math.pi / 180, 0.1, 12.0, 0, 0, 0)
    assert full["rear_total_rays"] > 0
    assert full["rear_valid_rays"] == 0
    assert full["rear_nearest_to_base_m"] is None


def test_forward_view_exposes_unscanned_edges_and_base_relative_hit_bearing():
    single = worker.summarize_forward_view([2.0], 0, 0.01, 0.1, 12.0, 0, 0, 0)
    assert single["center"]["observed_angle_min_rad"] == 0
    assert single["center"]["observed_angle_max_rad"] == 0
    assert single["center"]["max_unobserved_angle_rad"] >= math.pi / 6
    shifted = worker.summarize_forward_view([0.2], -0.7, 0.01, 0.1, 12.0, 0.1, 0, 0)
    assert shifted["right"]["nearest_bearing_from_base_rad"] == pytest.approx(-0.473, abs=0.01)


def test_forward_view_preserves_nearest_obstacle_for_each_base_direction():
    view = worker.summarize_forward_view([0.09, 0.12], -1.2, 0.5, 0.05, 12.0, 0.1, 0, 0)
    assert view["right"]["valid_rays"] == 2
    assert view["base_hits"]["right"]["nearest_m"] == pytest.approx(0.1569, abs=0.001)
    assert view["base_hits"]["center"]["nearest_m"] == pytest.approx(0.2067, abs=0.001)


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
    assert not worker.decision_pose_current((0.0, 0.0, 0.0), (0.14, 0.0, 0.0), 0.02, 0.0)
    assert not worker.decision_pose_current((0.0, 0.0, 0.0), (0.0, 0.0, 0.34), 0.0, 0.02)


def test_camera_odometry_misalignment_is_bounded_and_charged_to_pose_budget():
    assert worker.pose_alignment_uncertainty(100.0, 100.08) == pytest.approx((0.016, 0.048))
    assert worker.pose_alignment_uncertainty(100.0, 100.75) is None
    assert worker.pose_alignment_uncertainty(float("nan"), 100.0) is None


def test_delayed_model_decision_retains_original_anchor_and_requires_fresh_sample():
    captured = {"pose": (0.0, 0.0, 0.0), "pose_uncertainty": (0.02, 0.02)}
    sample = {"pose": (0.1, 0.0, 0.05), "scan_at": 12.0, "odom_at": 12.0}

    def fresh(observation, now):
        return now - observation["scan_at"] <= 0.1

    anchor = worker.validated_decision_anchor(captured, sample, 12.0, fresh)
    assert anchor == ((0.0, 0.0, 0.0), 0.02, 0.02)
    assert not worker.decision_pose_current(anchor[0], (0.14, 0.0, 0.0), anchor[1], anchor[2])
    assert worker.validated_decision_anchor(captured, sample, 12.2, fresh) is None
    sample["pose"] = (0.2, 0.0, 0.0)
    assert worker.validated_decision_anchor(captured, sample, 12.0, fresh) is None


def test_full_relay_pipe_fails_fast_instead_of_blocking_control():
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(write_fd, "wb", buffering=0)
    try:
        writer = worker.NonblockingPipeWriter(stream)
        with pytest.raises(OSError):
            while True:
                writer.write(b"x" * 512)
        started = time.monotonic()
        with pytest.raises(OSError):
            writer.write(b"stop\n")
        assert time.monotonic() - started < 0.5
    finally:
        stream.close()
        os.close(read_fd)


def test_exited_relay_reports_its_stop_reason():
    from types import SimpleNamespace

    relay = SimpleNamespace(poll=lambda: 0, stdout=io.BytesIO(b"MOTION_ACK 3\nSTOP_LATCHED\n"))
    assert worker.relay_exit_reason(relay) == "relay_stop_latched"
    relay = SimpleNamespace(poll=lambda: None, stdout=io.BytesIO(b""))
    assert worker.relay_exit_reason(relay) is None


def test_closed_relay_writer_never_writes_to_a_reused_file_descriptor():
    read_a, write_a = os.pipe()
    read_b, write_b = os.pipe()
    stream = os.fdopen(write_a, "wb", buffering=0)
    writer = worker.NonblockingPipeWriter(stream)
    try:
        writer.close()
        os.dup2(write_b, write_a)
        with pytest.raises((OSError, ValueError)):
            writer.write(b"stop\n")
        os.set_blocking(read_b, False)
        with pytest.raises(BlockingIOError):
            os.read(read_b, 100)
    finally:
        os.close(write_a)
        os.close(write_b)
        os.close(read_a)
        os.close(read_b)


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
