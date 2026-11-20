"""Isolated checks for the standalone controller's data alignment."""

import math
import threading

import numpy as np
import pytest
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


def test_callback_retains_receipt_from_same_clock_pair_as_sensor_stamp(monkeypatch):
    import explore
    class Guard(object):
        faulted = False
        def observe(self, *_args):
            return True
    class Stamp(object):
        def to_sec(self):
            return 99.9
    class Obj(object):
        pass
    observer = RosObservation.__new__(RosObservation)
    observer.lock = threading.Lock()
    observer.clock_guard = observer.odom_guard = Guard()
    observer.latest, observer.epoch, observer.limits = {}, 0, {'scan': .5}
    observer.rospy = Obj()
    observer.rospy.Time = Obj()
    observer.rospy.Time.now = lambda: None
    message = Obj()
    message.header = Obj()
    message.header.stamp = Stamp()
    monkeypatch.setattr(explore, 'sample_clocks', lambda *_: (100., 50.))
    observer._receive(message, 'scan')
    assert observer.latest['scan']['received_at'] == 50.
    assert observer.latest['scan']['stamp'] == pytest.approx(49.9)
    message.header.stamp.to_sec = lambda: 99.8
    observer._receive(message, 'scan')
    assert observer.epoch == 1
    assert observer.epoch_issue == {'reason': 'sensor_time_regression', 'sensor': 'scan'}


def test_callback_records_sensor_age_when_source_time_is_invalid(monkeypatch):
    import explore
    class Guard(object):
        def observe(self, *_args):
            return True
    class Obj(object):
        pass
    observer = RosObservation.__new__(RosObservation)
    observer.lock, observer.clock_guard = threading.Lock(), Guard()
    observer.latest, observer.epoch, observer.limits = {}, 0, {'odom': .25}
    observer.rospy = Obj()
    observer.rospy.Time = Obj()
    observer.rospy.Time.now = lambda: None
    message = Obj()
    message.header = Obj()
    message.header.stamp = Obj()
    message.header.stamp.to_sec = lambda: 100.02
    monkeypatch.setattr(explore, 'sample_clocks', lambda *_: (100., 50.))
    observer._receive(message, 'odom')
    assert observer.epoch_issue['reason'] == 'source_time_invalid'
    assert observer.epoch_issue['sensor_age_sec'] == pytest.approx(-.02)
    message.header.stamp.to_sec = lambda: -1e308
    monkeypatch.setattr(explore, 'sample_clocks', lambda *_: (1e308, 50.))
    observer._receive(message, 'odom')
    assert 'sensor_age_sec' not in observer.epoch_issue


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


def test_current_sample_rejects_evidence_that_expires_during_check(monkeypatch):
    import explore
    class Value(object):
        pass
    class Guard(object):
        faulted = False
    clock = iter((100.0, 100.2, 100.2))
    monkeypatch.setattr(explore, '_monotonic', lambda: next(clock))
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
    observer.latest = {'odom': {'stamp': 100.0, 'pose': (0.0, 0.0, 0.0), 'message': odom},
                       'status': {'stamp': 100.0, 'message': status}, 'scan': {'stamp': 100.0},
                       'depth': {'stamp': 100.0}}
    sample = {'epoch': 2, 'snapshot_at': 100.0, 'scan_at': 100.0, 'status_at': 100.0,
              'depth_at': 100.0, 'color_at': 100.0, 'pose': (0.0, 0.0, 0.0)}
    assert observer.current_valid_until(sample) is None


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
    observer.odom_history = [{'stamp': now, 'pose': (0.0, 0.0, 0.0), 'epoch': 2}]
    observer.latest = {'odom': {'stamp': now, 'pose': (0.0, 0.0, 0.0), 'message': odom},
                       'status': {'stamp': now, 'message': status}, 'scan': {'stamp': now},
                       'depth': {'stamp': now}}
    sample = {'epoch': 2, 'snapshot_at': now, 'scan_at': now, 'status_at': now, 'depth_at': now,
              'color_at': now, 'pose': (0.0, 0.0, 0.0),
              'scene': {'status': 'ok', 'coverage': {'sensor_fresh': True}, 'rays': [], 'occupied_hits': []}}
    assert observer.is_current_sample(sample)
    observer.latest['odom']['stamp'] = now - 0.12
    assert observer.is_current_sample(sample)
    observer.latest['odom']['stamp'] = now - 0.16
    assert not observer.is_current_sample(sample)
    observer.latest['odom']['stamp'] = now
    observer.latest['depth']['stamp'] = now + 0.01
    assert not observer.is_current_sample(sample)
    assert observer.current_last_issue == 'new_depth_unchecked'
    observer.latest['depth']['stamp'] = now
    observer.latest['scan']['stamp'] += 0.01
    assert not observer.is_current_sample(sample)
    observer.latest['scan']['stamp'] = now
    observer.latest['odom']['pose'] = (0.01, 0.0, 0.0)
    assert not observer.is_current_sample(sample)
    observer.latest['odom']['pose'] = (0.0, 0.0, 0.0)
    status.error_code = 1
    assert not observer.is_current_sample(sample)
    assert observer.current_last_issue == 'base_state'
    status.error_code = 0
    scan = Value()
    scan.header = Value()
    scan.header.frame_id = 'laser_link'
    scan.angle_min, scan.angle_increment = -1.57079632679, 0.01745329252
    scan.time_increment = 0.0
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


def test_missing_sensor_snapshot_has_a_non_sensitive_reason():
    import collections
    class Guard(object):
        faulted = False
    observer = RosObservation.__new__(RosObservation)
    observer.lock = threading.Lock()
    observer.clock_guard = observer.odom_guard = Guard()
    observer.latest = {}
    observer.odom_history = collections.deque()
    observer.epoch = 0
    observer.limits = {'scan': 0.5}
    assert observer.snapshot() is None
    assert observer.snapshot_last_issue == 'missing_scan'


def test_depth_transform_fallback_requires_known_unchanged_recent_rigid_transform():
    from explore import lookup_rigid_transform

    class Stamp(object):
        def __init__(self, value):
            self.value = value
        def to_sec(self):
            return self.value

    class Transformations(object):
        @staticmethod
        def euler_from_quaternion(rotation):
            return rotation[:3]

    class Tf(object):
        class ExtrapolationException(Exception):  # noqa: N818 - matches ROS API
            pass
        transformations = Transformations()

    class Ros(object):
        Time = Stamp

    class Listener(object):
        common = 99.94
        translation = (0.1, 0.0, 0.1)
        def lookupTransform(self, target, frame, stamp):  # noqa: N802 - matches ROS API
            if stamp.value:
                raise Tf.ExtrapolationException()
            return self.translation, (0.0, 0.0, 0.0, 1.0)
        def getLatestCommonTime(self, target, frame):  # noqa: N802 - matches ROS API
            return Stamp(self.common)

    listener = Listener()
    known = (0.1, 0.0, 0.1, 0.0, 0.0, 0.0)
    assert lookup_rigid_transform(listener, Tf, Ros, 'camera_depth_optical_frame',
                                  Stamp(100.0), known)[0] == listener.translation
    with pytest.raises(Tf.ExtrapolationException):
        lookup_rigid_transform(listener, Tf, Ros, 'camera_depth_optical_frame', Stamp(100.0), None)
    listener.common = 99.7
    with pytest.raises(ValueError):
        lookup_rigid_transform(listener, Tf, Ros, 'camera_depth_optical_frame', Stamp(100.0), known)
    listener.common = 99.94
    listener.translation = (0.2, 0.0, 0.1)
    with pytest.raises(ValueError):
        lookup_rigid_transform(listener, Tf, Ros, 'camera_depth_optical_frame', Stamp(100.0), known)


def test_projection_only_summary_keeps_stamp_without_running_full_image_statistics(monkeypatch):
    import explore
    def unexpected(*args):
        raise AssertionError('full image statistics are not needed by the projected gate')
    monkeypatch.setattr(explore, 'depth_grid', unexpected)
    assert explore.depth_summary_for_snapshot(None, '16UC1', 10.0, True) == {'stamp': 10.0}
    assert explore.depth_summary_for_snapshot(None, '16UC1', float('nan'), True) is None


def test_newer_depth_frame_requires_projection_and_rejects_obstacles(monkeypatch):
    import explore
    from test_projected_depth_gate import sample as projected_sample
    class Value(object):
        pass
    class Transformations(object):
        @staticmethod
        def quaternion_from_euler(*args):
            return (0.0, 0.0, 0.0, 1.0)
        @staticmethod
        def quaternion_matrix(_rotation):
            return np.eye(4)
    class Bridge(object):
        def imgmsg_to_cv2(self, message, desired_encoding):
            return np.ones((6, 6), dtype=np.uint16)
    depth, info = Value(), Value()
    depth.header, info.header = Value(), Value()
    depth.header.frame_id = info.header.frame_id = 'camera_depth_optical_frame'
    depth.width = info.width = depth.height = info.height = 6
    depth.encoding = '16UC1'
    depth.step, depth.is_bigendian, depth.data = 12, 0, b'\x01\x00' * 36
    info.K = [1.0, 0, 2.0, 0, 1.0, 2.0, 0, 0, 1.0]
    observer = RosObservation.__new__(RosObservation)
    from collections import deque
    observer._depth_projections = deque(maxlen=4)
    observer.bridge = Bridge()
    observer.tf = Value()
    observer.tf.transformations = Transformations()
    observer.depth_projection_config = {'sample_stride': 4}
    observer.depth_extrinsics = (0.1, 0.0, 0.1, 0.0, 0.0, 0.0)
    observer.epoch = 2
    observer.odom_history = [{'stamp': 100.0, 'pose': (0.0, 0.0, 0.0), 'epoch': 2}]
    observer.latest = {'info': {'stamp': 100.0, 'message': info}}
    frame = {'stamp': 100.0, 'message': depth}
    odom = {'pose': (0.0, 0.0, 0.0)}
    original = {'pose': (0.0, 0.0, 0.0)}
    corridor = projected_sample()['depth']['corridor']
    calls = []
    def project(*args):
        calls.append(True)
        return corridor
    monkeypatch.setattr(explore, 'project_depth_frame', project)
    assert observer._newest_depth_safe_locked(original, frame, odom, 100.01)
    assert observer._newest_depth_safe_locked(original, frame, odom, 100.01)
    assert len(calls) == 1
    frame['stamp'] = observer.latest['info']['stamp'] = 100.01
    observer.odom_history[0]['stamp'] = 100.01
    corridor['stamp'] = 100.01
    corridor['obstacle_count'] = 1
    corridor['outside_corridor_count'] -= 1
    assert not observer._newest_depth_safe_locked(original, frame, odom, 100.02)
    corridor['obstacle_count'] = 0
    corridor['outside_corridor_count'] += 1
    observer.odom_history[0]['stamp'] = 99.8
    assert not observer._newest_depth_safe_locked(original, frame, odom, 100.01)
def test_newer_scan_is_checked_for_obstacles_instead_of_unconditionally_discarding_sample():
    from explore import newest_scan_veto

    class Value(object):
        pass
    scan = Value()
    scan.header = Value()
    scan.header.frame_id = 'laser_link'
    scan.angle_min, scan.angle_increment = -1.57079632679, 0.01745329252
    scan.time_increment = 0.0
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
    scan.angle_increment = 0.01745329252
    scan.time_increment = -0.01
    assert newest_scan_veto(scan, extrinsics) is False


def test_newer_scan_cannot_replace_corridor_coverage_with_twenty_side_rays():
    from explore import newest_scan_veto
    class Value(object):
        pass
    scan = Value()
    scan.header = Value()
    scan.header.frame_id = 'laser_link'
    scan.angle_min, scan.angle_increment = -math.pi, 2 * math.pi / 719.0
    scan.time_increment = 0.0
    scan.range_min, scan.range_max = 0.1, 8.0
    scan.ranges = [0.0] * 720
    for index in range(440, 460):
        scan.ranges[index] = 2.0
    assert newest_scan_veto(scan, (0.1, 0.0, 0.1, 0.0, 0.0, 0.0)) is False


def test_newer_scan_can_combine_recent_pose_aligned_history_but_not_stale_history():
    from explore import newest_scan_veto
    class Value(object):
        pass
    scan = Value()
    scan.header = Value()
    scan.header.frame_id = 'laser_link'
    scan.angle_min, scan.angle_increment = -0.3, 0.6 / 32.0
    scan.time_increment = 0.0
    scan.range_min, scan.range_max = 0.1, 8.0
    latest = (0, 6, 12, 18, 24, 30, 32)
    scan.ranges = [2.0 if index in latest else 0.0 for index in range(33)]
    rays = []
    for index in (0, 3, 9, 15, 21, 27, 32):
        angle = scan.angle_min + index * scan.angle_increment
        rays.append({'start_base': (0.1, 0.0),
                     'hit_base': (0.1 + 2.0 * math.cos(angle), 2.0 * math.sin(angle)),
                     'stamp': 10.0})
    scene = {'status': 'ok', 'coverage': {'sensor_fresh': True, 'scan_stamp': 10.0},
             'rays': rays, 'occupied_hits': [{'point_base': ray['hit_base'], 'stamp': ray['stamp']}
                                              for ray in rays]}
    extrinsics, pose = (0.1, 0.0, 0.1, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    assert not newest_scan_veto(scan, extrinsics)
    assert newest_scan_veto(scan, extrinsics, scene, 10.1, 10.1, pose, pose, pose)
    assert not newest_scan_veto(scan, extrinsics, scene, 10.8, 10.8, pose, pose, pose)


def test_fast_recent_history_agrees_with_scalar_new_scan_check(monkeypatch):
    import explore
    from test_policy import fusion_scene
    class Value(object):
        pass
    scan = Value()
    scan.header = Value()
    scan.header.frame_id = 'laser_link'
    scan.angle_min, scan.angle_increment = -0.3, 0.6 / 32.0
    scan.time_increment = 0.0
    scan.range_min, scan.range_max = 0.1, 8.0
    pose = (0.0, 0.0, 0.0)
    scene = fusion_scene(10.0, 10.1, [2.0] * 33, [2.0] * 33)
    original_numpy = explore._np
    assert original_numpy is not None
    for missing in (0, 2, 3, 4, 5, 6, 7):
        scan.ranges = [2.0 if missing == 0 or i % missing != 0 else 0.0 for i in range(33)]
        monkeypatch.setattr(explore, '_np', None)
        expected = explore.newest_scan_veto(scan, (0.105, 0.0, 0.1, 0.0, 0.0, 0.0),
                                            scene, 10.2, 10.2, pose, pose, pose)
        monkeypatch.setattr(explore, '_np', original_numpy)
        assert explore.newest_scan_veto(scan, (0.105, 0.0, 0.1, 0.0, 0.0, 0.0),
                                       scene, 10.2, 10.2, pose, pose, pose) == expected


def test_new_scan_cache_reuses_only_same_scene_pose_and_unexpired_rays(monkeypatch):
    import explore
    calls = []
    def check_scan(*args):
        calls.append(True)
        return args[0].header.frame_id == 'laser_link'
    monkeypatch.setattr(explore, 'newest_scan_veto', check_scan)
    observer = RosObservation.__new__(RosObservation)
    observer.epoch = 2
    observer.extrinsics = (0.1, 0.0, 0.1, 0.0, 0.0, 0.0)
    scene = {'rays': [{'start_base': (0.1, 0.0), 'hit_base': (2.0, 0.0), 'stamp': 10.0}],
             'occupied_hits': [{'point_base': (2.0, 0.0), 'stamp': 10.0}]}
    sample = {'scene': scene, 'pose': (0.0, 0.0, 0.0)}
    class Scan(object):
        header = type('Header', (object,), {'frame_id': 'laser_link'})()
        angle_min = 0.0
        angle_increment = 0.1
        range_min = 0.1
        range_max = 8.0
        time_increment = 0.0
        ranges = [2.0] * 33
    scan = {'stamp': 10.1, 'message': Scan()}
    aligned = {'pose': (0.0, 0.0, 0.0)}
    odom = {'pose': (0.0, 0.0, 0.0)}
    assert observer._cached_newest_scan_safe_locked(sample, scan, aligned, odom, 10.1)
    assert observer._cached_newest_scan_safe_locked(sample, scan, aligned, odom, 10.11)
    assert len(calls) == 1
    scan['message'].header.frame_id = 'unexpected_frame'
    assert not observer._cached_newest_scan_safe_locked(sample, scan, aligned, odom, 10.115)
    assert len(calls) == 2
    scan['message'].header.frame_id = 'laser_link'
    odom['pose'] = (0.001, 0.0, 0.0)
    assert observer._cached_newest_scan_safe_locked(sample, scan, aligned, odom, 10.12)
    assert len(calls) == 3
    scene['rays'][0]['hit_base'] = (1.9, 0.0)
    assert observer._cached_newest_scan_safe_locked(sample, scan, aligned, odom, 10.13)
    assert len(calls) == 4
    assert observer._cached_newest_scan_safe_locked(sample, scan, aligned, odom, 10.66)
    assert len(calls) == 5


def test_new_scan_veto_distinguishes_sparse_coverage_from_measured_blockage():
    from explore import newest_scan_veto

    class Scan(object):
        header = type('Header', (object,), {'frame_id': 'laser_link'})()
        angle_min, angle_increment = -.3, .02
        range_min, range_max, time_increment = .1, 8., 0.
        ranges = [0.] * 33

    scan, diagnostic = Scan(), {}
    extrinsics = (.1, 0., .08, 0., 0., 0.)
    scan.ranges[15] = 2.
    assert not newest_scan_veto(scan, extrinsics, diagnostic=diagnostic)
    assert diagnostic['reason'] == 'scan_coverage_pending'
    scan.ranges[15] = .2
    assert not newest_scan_veto(scan, extrinsics, diagnostic=diagnostic)
    assert diagnostic['reason'] == 'front_blocked'


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
def test_depth_projection_reuses_frame_with_equivalent_calibration(monkeypatch):
    import collections
    import copy
    import numpy as np
    import explore

    class Obj(object):
        pass

    observer = explore.RosObservation.__new__(explore.RosObservation)
    observer._depth_projections = collections.deque(maxlen=4)
    observer.depth_projection_config = {'sample_stride': 4}
    observer.tf, observer.bridge = Obj(), Obj()
    observer.tf.transformations = Obj()
    observer.tf.transformations.quaternion_matrix = lambda _: np.eye(4)
    observer.bridge.imgmsg_to_cv2 = lambda *a, **k: None
    image, info = Obj(), Obj()
    image.data, image.encoding = b'\x00\x01', '16UC1'
    image.width, image.height, image.step, image.is_bigendian = 1, 1, 2, 0
    image.header = Obj()
    image.header.frame_id = 'camera_depth_optical_frame'
    info.K = (1., 0., .5, 0., 1., .5, 0., 0., 1.)
    calls = []

    def project(_array, _encoding, _intrinsics, _matrix, stamp, _config):
        calls.append(stamp)
        return {'stamp': stamp, 'nearest_obstacle': {'x': .5}}

    monkeypatch.setattr(explore, 'project_depth_frame', project)
    trans, rotation = (0., 0., 0.), (0., 0., 0., 1.)
    assert observer._depth_projection(image, info, 10., 0, trans, rotation, cache_only=True) is None
    assert calls == []
    first = observer._depth_projection(image, info, 10., 0, trans, rotation)
    first['nearest_obstacle']['x'] = 1.
    assert copy.copy(observer)._depth_projection(image, copy.copy(info), 10., 0, trans, rotation
                                                 )['nearest_obstacle']['x'] == .5
    assert calls == [10.]
    assert observer._depth_projection(image, info, 10., 0, trans, rotation, cache_only=True)['stamp'] == 10.
    observer._depth_projection(image, info, 10., 1, trans, rotation)
    observer._depth_projection(image, info, 10., 1, (.01, 0., 0.), rotation)
    assert calls == [10., 10., 10.]


def test_depth_worker_processes_same_calibrated_frame_once_and_closes(monkeypatch):
    import explore

    class Obj(object):
        def __init__(self, **fields):
            self.__dict__.update(fields)

    observer = explore.RosObservation.__new__(explore.RosObservation)
    observer._depth_worker = None
    observer.lock = threading.Lock()
    observer.epoch = 1
    observer.clock_guard = observer.odom_guard = Obj(faulted=False)
    observer.depth_extrinsics = (0., 0., 0., 0., 0., 0.)
    observer.depth_projection_config = {'sample_stride': 4}
    observer.listener = observer.rospy = None
    rotations = Obj(euler_from_quaternion=lambda _: (0., 0., 0.),
                    quaternion_from_euler=lambda *_: (0., 0., 0., 1.))
    observer.tf = Obj(transformations=rotations)
    frame = Obj(frame_id='camera_depth_optical_frame', stamp=None)
    image = Obj(header=frame, width=10, height=10)
    info = Obj(header=frame, width=10, height=10, K=[1.] * 9)
    observer.latest = {'depth': {'stamp': 10., 'message': image},
                       'info': {'stamp': 10., 'message': info}}
    monkeypatch.setattr(explore, '_monotonic', lambda: 10.01)
    monkeypatch.setattr(explore, 'lookup_rigid_transform', lambda *_: ((0., 0., 0.), (0., 0., 0., 1.)))
    calls = []
    def project(*args):
        calls.append(args)
        return {'stamp': 10.}

    observer._depth_projection = project
    observer.start_depth_preprojection()
    try:
        assert observer._depth_ready.wait(.3)
        # Simulate a fresh CameraInfo header with identical calibration.
        with observer.lock:
            observer.latest['info'] = {'stamp': 10.01, 'message': info}
        observer._depth_event.set()
    finally:
        assert observer.close_depth_preprojection()
    assert observer.depth_preprojection_counts['processed'] == 1
    assert len(calls) == 1 and calls[0][:4] == (image, info, 10., 1)
