"""Isolated checks for the standalone controller's data alignment."""

import math

import numpy as np
from explore import (
    DecisionGate,
    build_decision_evidence,
    depth_grid,
    nearest_odometry,
    physical_forward_view,
    snapshot_is_fresh,
    source_time,
    transform_matches,
)


def test_nearest_odometry_requires_a_close_timestamp():
    samples = [{"stamp": 10.0, "pose": (0.0, 0.0, 0.0)},
               {"stamp": 10.04, "pose": (0.01, 0.0, 0.0)}]
    assert nearest_odometry(samples, 10.035)["stamp"] == 10.04
    assert nearest_odometry(samples, 10.2) is None


def test_model_result_must_match_capture_and_remain_near_the_observation_pose():
    gate = DecisionGate()
    assert gate.accept({"ok": True, "choice": "forward", "captured_at": 10.0},
                       10.0, (0.0, 0.0, 0.0), 12.0, (0.03, 0.0, 0.0))
    assert gate.choice == "forward"
    assert not gate.accept({"ok": True, "choice": "left", "captured_at": 10.1},
                           10.0, (0.0, 0.0, 0.0), 12.0, (0.03, 0.0, 0.0))
    assert not gate.accept({"ok": True, "choice": "right", "captured_at": 10.2},
                           10.2, (0.0, 0.0, 0.0), 12.0, (0.2, 0.0, 0.0))
    assert not gate.accept({"ok": True, "choice": "forward", "captured_at": 10.2},
                           10.2, (0.0, 0.0, 0.0), 20.0, (0.0, 0.0, 0.0))


def test_model_stop_overrides_without_a_valid_current_pose():
    gate = DecisionGate()
    assert gate.accept({"ok": True, "choice": "stop", "captured_at": 10.0},
                       10.0, (0.0, 0.0, 0.0), 12.0, None)
    assert gate.choice == "stop"


def test_pose_validation_rejects_nonfinite_and_handles_wrapped_heading():
    gate = DecisionGate()
    decision = {"ok": True, "choice": "forward", "captured_at": 10.0}
    assert gate.accept(decision, 10.0, (0.0, 0.0, math.pi - 0.01),
                       10.5, (0.0, 0.0, -math.pi + 0.01))
    assert not gate.accept(decision, 10.0, (0.0, 0.0, 0.0),
                           10.5, (float("nan"), 0.0, 0.0))


def test_source_timestamp_uses_sensor_age_and_rejects_future_or_stale():
    assert abs(source_time(100.0, 50.0, 99.8, 0.5) - 49.8) < 1e-9
    assert source_time(100.0, 50.0, 99.0, 0.5) is None
    assert source_time(100.0, 50.0, 100.1, 0.5) is None
    assert source_time(100.0, 50.0, float('nan'), 0.5) is None


def test_depth_grid_reports_spatial_gaps_not_just_global_fraction():
    array = np.full((9, 9), 2000, dtype=np.uint16)
    array[0:3, 3:6] = 0
    result = depth_grid(array, '16UC1', 10.0)
    assert result['stamp'] == 10.0
    assert result['min_m'] == 2.0
    assert result['valid_grid'] == [[1.0, 0.0, 1.0], [1.0] * 3, [1.0] * 3]
    assert depth_grid(array, 'invalid', 10.0) is None
    assert depth_grid(array, '16UC1', float('nan')) is None


def test_decision_expires_when_robot_has_moved_since_image():
    gate = DecisionGate()
    assert gate.accept({'ok': True, 'choice': 'forward', 'captured_at': 10.0},
                       10.0, (0.0, 0.0, 0.0), 10.1, (0.0, 0.0, 0.0))
    assert gate.current(10.2, (0.1, 0.0, 0.0))['choice'] == 'forward'
    assert gate.current(10.2, (0.16, 0.0, 0.0)) is None
    assert gate.current(18.1, (0.0, 0.0, 0.0)) is None
    assert gate.current(10.2, None) is None


def test_model_evidence_is_bound_to_oldest_sensor_and_does_not_include_raw_message():
    sample = {'scan_at': 9.9, 'odom_at': 10.0, 'status_at': 9.7, 'depth_at': 9.8,
              'color_at': 9.95, 'forward_view': {'coverage': 'partial'},
              'state': {'linear_x_mps': 0.0, 'angular_z_radps': 0.0, 'battery_voltage': 12.0}}
    evidence = build_decision_evidence(sample, b'\xff\xd8jpeg')
    assert evidence['captured_at'] == 9.7
    assert evidence['laser']['forward_view'] == {'coverage': 'partial'}
    assert evidence['base']['battery_voltage'] == 12.0
    assert 'color' not in evidence
    assert build_decision_evidence(sample, b'not-jpeg') is None


def test_model_failure_invalidates_previous_forward_decision():
    gate = DecisionGate()
    assert gate.accept({'ok': True, 'choice': 'forward', 'captured_at': 10.0},
                       10.0, (0.0, 0.0, 0.0), 10.1, (0.0, 0.0, 0.0))
    gate.invalidate()
    assert gate.current(10.2, (0.0, 0.0, 0.0)) is None


def test_decision_cannot_revive_after_pose_or_odometry_was_lost():
    gate = DecisionGate()
    assert gate.accept({'ok': True, 'choice': 'forward', 'captured_at': 10.0},
                       10.0, (0.0, 0.0, 0.0), 10.1, (0.0, 0.0, 0.0))
    assert gate.current(10.2, (0.2, 0.0, 0.0)) is None
    assert gate.current(10.3, (0.0, 0.0, 0.0)) is None


def test_snapshot_requires_current_pose_even_if_other_sensors_are_fresh():
    sample = {'snapshot_at': 10.05, 'odom_at': 10.0, 'pose': (0.0, 0.0, 0.0)}
    assert snapshot_is_fresh(sample, 10.08, (0.001, 0.0, 0.0))
    assert not snapshot_is_fresh(sample, 10.11, (0.001, 0.0, 0.0))
    assert not snapshot_is_fresh(sample, 10.08, (0.02, 0.0, 0.0))
    assert not snapshot_is_fresh(sample, 10.08, None)
    assert not snapshot_is_fresh(sample, 10.0, (0.0, 0.0, 0.0))


def test_model_scan_summary_does_not_turn_max_range_or_rear_into_hits():
    view = physical_forward_view([2.0, 10.0, 10.0, 10.0, 2.0], -math.pi,
                                 math.pi / 2, 0.1, 10.0, 0.105, 0.0, 0.0)
    assert view['rear_valid_rays'] == 0
    assert view['center']['valid_rays'] == 0


def test_extrinsic_change_in_any_translation_or_rotation_invalidates_snapshot():
    original = (0.105, 0.0, 0.08, 0.0, 0.0, 0.0)
    assert transform_matches(original, original)
    for changed in [(0.105, 0.0, 0.09, 0.0, 0.0, 0.0),
                    (0.105, 0.0, 0.08, 0.01, 0.0, 0.0),
                    (0.105, 0.0, 0.08, 0.0, 0.01, 0.0),
                    (0.105, 0.0, 0.08, 0.0, 0.0, 0.01)]:
        assert not transform_matches(original, changed)
