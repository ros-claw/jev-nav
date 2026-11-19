"""Isolated checks for the standalone controller's data alignment."""

import math
import threading

import numpy as np
from explore import (
    DecisionGate,
    RosObservation,
    _monotonic,
    base_state,
    build_decision_evidence,
    depth_grid,
    nearest_odometry,
    physical_forward_view,
    refreshed_odometry,
    sample_clocks,
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


def test_delayed_clock_pair_is_discarded_instead_of_false_clock_jump():
    times = iter([10.0, 10.004, 10.1, 10.14])
    first = sample_clocks(lambda: next(times), lambda: 100.0)
    assert first[0] == 100.0 and abs(first[1] - 10.002) < 1e-9
    assert sample_clocks(lambda: next(times), lambda: 100.1) is None


def test_observation_refresh_uses_latest_valid_odometry_after_slow_processing():
    original = {'stamp': 10.0, 'pose': (0.0, 0.0, 0.0)}
    latest = {'stamp': 10.08, 'pose': (0.001, 0.0, 0.0)}
    assert refreshed_odometry(original, latest, 10.09) is latest
    assert refreshed_odometry(original, latest, 10.19) is None
    assert refreshed_odometry(original, {'stamp': 9.9, 'pose': (0, 0, 0)}, 10.09) is None
    assert refreshed_odometry(original, {'stamp': 10.08, 'pose': None}, 10.09) is None


def test_continuity_snapshot_atomically_checks_epoch_guards_and_odom_freshness():
    class Guard(object):
        faulted = False
        fault_reason = None
    observer = RosObservation.__new__(RosObservation)
    observer.lock = threading.Lock()
    observer.epoch = 9
    observer.clock_guard, observer.odom_guard = Guard(), Guard()
    observer.latest = {'odom': {'stamp': _monotonic(), 'pose': (0.0, 0.0, 0.0)}}
    state = observer.continuity_snapshot()
    assert state['epoch'] == 9 and state['pose'] == (0.0, 0.0, 0.0)
    observer.latest['odom']['stamp'] -= 0.2
    assert observer.continuity_snapshot()['pose'] is None
    observer.odom_guard.faulted = True
    observer.odom_guard.fault_reason = 'odometry_sequence_reset'
    assert observer.continuity_snapshot()['odom_fault'] == 'odometry_sequence_reset'


def test_stationary_trace_keeps_sample_before_start_and_all_later_samples():
    class Value(object):
        pass
    class Guard(object):
        faulted = False
    def entry(seq, stamp):
        msg = Value()
        msg.header = Value()
        msg.header.seq = seq
        msg.twist = Value()
        msg.twist.twist = Value()
        msg.twist.twist.linear = Value()
        msg.twist.twist.angular = Value()
        msg.twist.twist.linear.x = 0.0
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.angular.z = 0.0
        return {'stamp': stamp, 'pose': (0.0, 0.0, 0.0), 'epoch': 3, 'message': msg}
    observer = RosObservation.__new__(RosObservation)
    observer.lock = threading.Lock()
    observer.epoch = 3
    observer.clock_guard, observer.odom_guard = Guard(), Guard()
    observer.odom_history = [entry(1, 10.0), entry(2, 10.05), entry(3, 10.1), entry(4, 10.15)]
    trace = observer.stationary_trace(10.075, 10.15)
    assert [item['seq'] for item in trace] == [2, 3, 4]
    assert all(item['epoch'] == 3 and item['linear_x_mps'] == 0
               and item['linear_y_mps'] == 0 for item in trace)
    assert observer.stationary_trace(9.0, 10.15) == []
    observer.odom_guard.faulted = True
    assert observer.stationary_trace(10.075, 10.15) == []


def test_base_state_keeps_mode_and_all_planar_velocities():
    class Value(object):
        pass
    status, odom = Value(), Value()
    status.motion_mode, status.error_code, status.vehicle_state, status.control_mode = 0, 0, 0, 1
    status.battery_voltage = 12.2
    odom.twist = Value()
    odom.twist.twist = Value()
    odom.twist.twist.linear = Value()
    odom.twist.twist.angular = Value()
    odom.twist.twist.linear.x, odom.twist.twist.linear.y = 0.02, 0.001
    odom.twist.twist.angular.z = 0.0
    state = base_state(odom, status, 10.0)
    assert state['vehicle_state'] == 0 and state['control_mode'] == 1
    assert state['linear_y_mps'] == 0.001 and state['stamp'] == 10.0
    assert base_state(odom, status, float('nan')) is None


def test_current_sample_rejects_new_scan_odom_drift_or_base_fault():
    class Value(object):
        pass
    class Guard(object):
        faulted = False
    now = _monotonic()
    odom, status = Value(), Value()
    odom.twist = Value()
    odom.twist.twist = Value()
    odom.twist.twist.linear, odom.twist.twist.angular = Value(), Value()
    odom.twist.twist.linear.x = odom.twist.twist.linear.y = odom.twist.twist.angular.z = 0.0
    status.motion_mode = status.error_code = status.vehicle_state = 0
    status.control_mode, status.battery_voltage = 1, 12.2
    observer = RosObservation.__new__(RosObservation)
    observer.lock = threading.Lock()
    observer.epoch = 2
    observer.clock_guard, observer.odom_guard = Guard(), Guard()
    observer.latest = {'odom': {'stamp': now, 'pose': (0.0, 0.0, 0.0), 'message': odom},
                       'status': {'stamp': now, 'message': status}, 'scan': {'stamp': now}}
    sample = {'epoch': 2, 'snapshot_at': now, 'scan_at': now, 'status_at': now, 'depth_at': now,
              'color_at': now, 'pose': (0.0, 0.0, 0.0)}
    assert observer.is_current_sample(sample)
    observer.latest['scan']['stamp'] += 0.01
    assert not observer.is_current_sample(sample)
    observer.latest['scan']['stamp'] = now
    observer.latest['odom']['pose'] = (0.01, 0.0, 0.0)
    assert not observer.is_current_sample(sample)
    observer.latest['odom']['pose'] = (0.0, 0.0, 0.0)
    status.error_code = 1
    assert not observer.is_current_sample(sample)
    status.error_code = 0
    scan = Value()
    scan.header = Value()
    scan.header.frame_id = 'laser_link'
    scan.angle_min, scan.angle_increment = -1.57079632679, 0.01745329252
    scan.range_min, scan.range_max, scan.ranges = 0.1, 8.0, [2.0] * 181
    observer.extrinsics = (0.1, 0.0, 0.1, 0.0, 0.0, 0.0)
    observer.latest['scan']['message'] = scan
    sample['scan_at'] = now - 0.01
    assert observer.is_current_sample(sample)
    scan.ranges[90] = 0.3
    assert not observer.is_current_sample(sample)
    scan.ranges[90] = 2.0
    sample['snapshot_at'] = now - 0.15
    assert observer.is_current_sample(sample)
    sample['snapshot_at'] = now - 0.26
    assert not observer.is_current_sample(sample)
    sample['snapshot_at'] = now - 0.23
    assert observer.is_current_sample(sample)
    sample['snapshot_at'] = now - 0.15
    sample['scan_at'] = now - 0.35
    assert observer.is_current_sample(sample)
    scan.ranges[90] = 0.3
    assert not observer.is_current_sample(sample)
    scan.ranges[90] = 2.0
    observer.latest['scan']['stamp'] = now - 0.31
    assert not observer.is_current_sample(sample)
    observer.latest['scan']['stamp'] = now
    sample['snapshot_at'] = now
    observer.epoch = 1
    for malformed in (True, 1.0, '1'):
        sample['epoch'] = malformed
        assert not observer.is_current_sample(sample)
        assert not observer.is_current_epoch(malformed)


def test_observation_constructor_accepts_optional_projection_configuration():
    import inspect
    args = inspect.getcallargs(RosObservation.__init__, None)
    assert args['depth_projection_config'] is None
    args = inspect.getcallargs(RosObservation.__init__, None, depth_projection_config={'sample_stride': 4})
    assert args['depth_projection_config']['sample_stride'] == 4


def test_odometry_subscription_can_buffer_short_cpu_scheduling_delays():
    from explore import sensor_queue_size
    assert sensor_queue_size('odom') >= 10
    assert sensor_queue_size('scan') == 1
    assert sensor_queue_size('color') == 1


def test_newer_scan_is_checked_for_obstacles_instead_of_unconditionally_discarding_sample():
    from explore import newest_scan_veto

    class Value(object):
        pass
    scan = Value()
    scan.header = Value()
    scan.header.frame_id = 'laser_link'
    scan.angle_min, scan.angle_increment = -1.57079632679, 0.01745329252
    scan.range_min, scan.range_max = 0.1, 8.0
    scan.ranges = [2.0] * 181
    extrinsics = (0.1, 0.0, 0.1, 0.0, 0.0, 0.0)
    assert newest_scan_veto(scan, extrinsics) is True
    scan.ranges[90] = 0.3
    assert newest_scan_veto(scan, extrinsics) is False
    scan.ranges = [float('nan')] * 181
    assert newest_scan_veto(scan, extrinsics) is False
    scan.ranges = [2.0] * 181
    scan.header.frame_id = 'unexpected_frame'
    assert newest_scan_veto(scan, extrinsics) is False
    scan.header.frame_id = 'laser_link'
    scan.angle_increment = 1.0
    assert newest_scan_veto(scan, extrinsics) is False


def test_depth_grid_reports_spatial_gaps_not_just_global_fraction():
    array = np.full((9, 9), 2000, dtype=np.uint16)
    array[0:3, 3:6] = 0
    result = depth_grid(array, '16UC1', 10.0)
    assert result['stamp'] == 10.0
    assert result['min_m'] == 2.0
    assert result['valid_grid'] == [[1.0, 0.0, 1.0], [1.0] * 3, [1.0] * 3]
    assert depth_grid(array, 'invalid', 10.0) is None
    assert depth_grid(array, '16UC1', float('nan')) is None


def test_depth_grid_rejects_values_outside_the_supplied_camera_range():
    array = np.full((3, 3), 2000, dtype=np.uint16)
    array[0, 0] = 299
    array[1, 1] = 3001
    summary = depth_grid(array, '16UC1', 10.0, min_depth_m=0.3, max_depth_m=3.0)
    assert summary['valid_grid'][0][0] == 0.0
    assert summary['valid_grid'][1][1] == 0.0
    assert summary['valid_grid'][2][2] == 1.0
    assert depth_grid(array, '16UC1', 10.0, min_depth_m=3.0, max_depth_m=0.3) is None


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
