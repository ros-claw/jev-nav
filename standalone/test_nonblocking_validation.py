from __future__ import division

import threading

import pytest
from explore import RosObservation, _monotonic


class Value(object):
    pass


class Guard(object):
    faulted = False


def observer_and_sample():
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
    observer.odom_history = [{'stamp': now, 'pose': (0., 0., 0.), 'epoch': 2}]
    observer.latest = {'odom': {'stamp': now, 'pose': (0., 0., 0.), 'message': odom},
                       'status': {'stamp': now, 'message': status}, 'scan': {'stamp': now},
                       'depth': {'stamp': now}}
    sample = {'epoch': 2, 'snapshot_at': now, 'scan_at': now - .01, 'status_at': now,
              'depth_at': now, 'color_at': now, 'pose': (0., 0., 0.), 'scene': {}}
    return observer, sample


@pytest.mark.parametrize('changed,reason', [
    ('scan', 'prepared_scan_advanced'), ('depth', 'prepared_depth_advanced'),
    ('epoch', 'epoch_or_guard'), ('pose', 'pose_changed'),
    ('state', 'base_state'), ('expiry', 'prepared_pending'),
])
def test_prepared_command_requires_unchanged_sensor_generation_and_current_state(changed, reason, monkeypatch):
    import explore
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    observer.extrinsics = observer.depth_extrinsics = observer.depth_projection_config = None
    sample['scan_at'] = now
    monkeypatch.setattr(explore, '_monotonic', lambda: now)
    evidence, issue = observer.prepare_command_evidence(sample)
    assert issue is None and evidence is not None
    assert observer.read_command_evidence(sample, evidence) == (evidence['valid_until'], None)
    if changed in ('scan', 'depth'):
        observer.latest[changed] = dict(observer.latest[changed])
    elif changed == 'epoch':
        observer.epoch += 1
        assert observer.read_command_evidence(sample, None) == (None, 'epoch_or_guard')
    elif changed == 'pose':
        observer.latest['odom']['pose'] = (.006, 0., 0.)
    elif changed == 'state':
        observer.latest['status']['message'].error_code = 1
    else:
        monkeypatch.setattr(explore, '_monotonic', lambda: evidence['valid_until'])
    assert observer.read_command_evidence(sample, evidence) == (None, reason)


def test_split_proof_combines_current_parts_with_original_deadlines(monkeypatch):
    import explore
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    observer.extrinsics = observer.depth_extrinsics = observer.depth_projection_config = None
    sample['scan_at'] = now
    sample['_front_coverage_valid_until'] = now + .22
    monkeypatch.setattr(explore, '_monotonic', lambda: now)
    monkeypatch.setattr(RosObservation, '_newest_depth_safe_locked', lambda *_: True)
    _, scan_input = observer._capture_split_input('scan', sample)
    scan, issue = observer._check_split_input('scan', sample, scan_input)
    assert issue is None
    _, depth_input = observer._capture_split_input('depth', sample)
    depth, issue = observer._check_split_input('depth', sample, depth_input)
    assert issue is None and depth['expires'] < scan['expires']
    proof, issue = observer._combine_split_proof(sample, scan, depth)
    assert issue is None and proof['valid_until'] == min(depth['expires'], now + .070)
    observer.latest['depth'] = dict(observer.latest['depth'])
    assert observer._combine_split_proof(sample, scan, depth) == (None, 'prepared_depth_advanced')


def test_split_pending_frame_does_not_extend_prior_proof(monkeypatch):
    import explore
    from relay import RelayGate
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    observer.extrinsics = observer.depth_extrinsics = observer.depth_projection_config = None
    sample['scan_at'] = now
    observer.latest['depth']['generation'] = 4
    monkeypatch.setattr(explore, '_monotonic', lambda: now)
    evidence, issue = observer.prepare_command_evidence(sample)
    assert issue is None
    evidence['valid_until'] = now + .04
    observer.latest['depth'] = {'generation': 5, 'stamp': now + .01, 'received_at': now + .02}
    monkeypatch.setattr(explore, '_monotonic', lambda: now + .03)
    deadline, issue = observer.read_command_evidence(sample, evidence, True)
    assert deadline == now + .04 and issue is None
    relay = RelayGate(now, max_motion_sec=2., require_intent_deadline=True)
    assert relay.accept({'sequence': 0, 'stamp': now + .03, 'linear': .02, 'angular': 0.,
                         'expires_at': deadline}, now + .03)
    assert observer.read_command_evidence(sample, evidence) == (None, 'prepared_depth_advanced')
    observer.latest['depth']['generation'] = 6
    assert observer.read_command_evidence(sample, evidence, True) == (None, 'prepared_depth_advanced')
    assert relay.command(deadline) == (0., 0.) and relay.reason == 'intent_deadline'


def test_split_worker_checks_owned_frame_after_new_depth_arrives(monkeypatch):
    import explore
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    observer.extrinsics = observer.depth_extrinsics = observer.depth_projection_config = None
    sample['scan_at'] = now
    sample['_front_coverage_valid_until'] = now + .22
    observer.latest['depth']['generation'] = 4
    monkeypatch.setattr(explore, '_monotonic', lambda: now + .02)
    examined = []

    def check_depth(_probe, _sample, frame, _odom, _at):
        examined.append(frame)
        return True

    monkeypatch.setattr(RosObservation, '_newest_depth_safe_locked', check_depth)
    _, captured_depth = observer._capture_split_input('depth', sample)
    original = observer.latest['depth']
    observer.latest['depth'] = {'stamp': now + .01, 'received_at': now + .015, 'generation': 5}
    _, captured_scan = observer._capture_split_input('scan', sample)
    scan, issue = observer._check_split_input('scan', sample, captured_scan)
    assert issue is None
    depth, issue = observer._check_split_input('depth', sample, captured_depth)
    assert issue is None and examined == [original] and depth['entry'] is original
    proof, issue = observer._combine_split_proof(sample, scan, depth)
    assert issue is None and proof['valid_until'] == now + .015 + .07

    observer.latest['depth'] = dict(observer.latest['depth'], generation=7)
    assert observer._combine_split_proof(sample, scan, depth) == (None, 'prepared_depth_advanced')


def test_snapshot_uses_completed_depth_with_its_original_timestamp_and_calibration():
    from explore import CompletedDepth
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    depth = {'stamp': now - .04}
    calibration = {'message': Value()}
    calibration['message'].K = (1.,) * 9
    observer.depth_extrinsics = (.1, 0., .1, 0., 0., 0.)
    observer.depth_projection_config = {'sample_stride': 4}
    completed = CompletedDepth(2, depth, calibration, observer.depth_extrinsics,
                               (('sample_stride', 4),), {'stamp': depth['stamp']})
    observer._completed_depth_frames = (completed,)
    current = {'depth': {'stamp': now}, 'info': calibration}
    assert observer._fresh_completed_depth(current, 2, now) is completed
    assert completed.projection['stamp'] == depth['stamp'] < current['depth']['stamp']
    assert observer._fresh_completed_depth(current, 3, now) is None
    assert observer._fresh_completed_depth(current, 2, now + .12) is None
    assert observer._fresh_completed_depth(dict(current, info=None), 2, now) is None
    observer.depth_projection_config['sample_stride'] = 8
    assert observer._fresh_completed_depth(current, 2, now) is None


def test_split_capture_uses_aligned_completed_depth_while_newer_frame_waits(monkeypatch):
    import explore
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    observer._depth_projector = observer._depth_worker = Value()
    observer.depth_extrinsics = (0., 0., 0., 0., 0., 0.)
    observer.depth_projection_config = {'sample_stride': 4}
    observer.latest['odom']['stamp'] = now - .069
    observer.odom_history[0]['stamp'] = now - .069
    old = {'stamp': now - .068, 'received_at': now - .06, 'generation': 10}
    newer = {'stamp': now - .010, 'received_at': now - .004, 'generation': 11}
    observer.latest['depth'] = newer
    header = Value()
    header.frame_id = 'camera_depth_optical_frame'
    info = Value()
    info.header, info.width, info.height, info.K = header, 320, 240, (1.,) * 9
    info_entry = {'stamp': now - .011, 'message': info}
    observer.latest['info'] = info_entry
    observer._completed_depth_frames = tuple(
        explore.CompletedDepth(2, frame, info_entry, observer.depth_extrinsics,
                               (('sample_stride', 4),), {'stamp': frame['stamp']})
        for frame in (newer, old))
    monkeypatch.setattr(explore, '_monotonic', lambda: now)
    _, captured = observer._capture_split_input('depth', sample)
    assert captured[1] is old
    assert captured[3].projection['stamp'] == old['stamp']
    observer.odom_history.append({'stamp': now - .008, 'pose': (0., 0., 0.), 'epoch': 2})
    _, captured = observer._capture_split_input('depth', sample)
    assert captured[1] is newer


def test_split_worker_uses_published_odometry_while_ros_lock_is_held(monkeypatch):
    import explore
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    observer.extrinsics = observer.depth_extrinsics = observer.depth_projection_config = None
    observer._sensor_receipts = {}
    observer._publish_split_inputs()
    monkeypatch.setattr(explore, '_monotonic', lambda: now)

    def depth_check(probe, _sample, _frame, _odom, _at):
        return probe.odom_history == tuple(observer.odom_history)

    monkeypatch.setattr(RosObservation, '_newest_depth_safe_locked', depth_check)
    result = []
    def check():
        _, captured = observer._capture_split_input('depth', sample)
        result.append(observer._check_split_input('depth', sample, captured))

    worker = threading.Thread(target=check)
    with observer.lock:
        worker.start()
        worker.join(.25)
        completed_under_lock = not worker.is_alive()
    worker.join(1.)
    assert completed_under_lock and result[0][1] is None


def test_command_evidence_does_not_wait_for_a_busy_ros_lock():
    observer, sample = observer_and_sample()
    with observer.lock:
        assert observer.read_command_evidence(sample, None, nonblocking=True) == (None, 'prepared_state_busy')


def test_published_command_view_can_be_read_under_ros_lock_and_rejects_epoch_change(monkeypatch):
    import explore
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    sample['scan_at'] = now
    observer.extrinsics = observer.depth_extrinsics = observer.depth_projection_config = None
    observer._sensor_receipts = {}
    monkeypatch.setattr(explore, '_monotonic', lambda: now)
    evidence, issue = observer.prepare_initial_command_evidence(sample)
    assert issue is None
    with observer.lock:
        observer._publish_split_inputs()
        assert observer.read_command_evidence(sample, evidence, nonblocking=True) == (
            evidence['valid_until'], None)
        observer.epoch += 1
        assert observer.read_command_evidence(sample, evidence, nonblocking=True) == (None, 'epoch_or_guard')


@pytest.mark.parametrize('update,expected', [
    ('fresh_odom', None),
    ('outside_pose', 'pose_changed'),
    ('continuous', 'prepared_state_busy'),
    ('epoch', 'epoch_or_guard'),
    ('unsafe_old_state', 'base_state'),
])
def test_published_command_view_rechecks_one_concurrent_update(update, expected, monkeypatch):
    import explore
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    sample['scan_at'] = now
    observer.extrinsics = observer.depth_extrinsics = observer.depth_projection_config = None
    observer._sensor_receipts = {}
    monkeypatch.setattr(explore, '_monotonic', lambda: now)
    evidence, issue = observer.prepare_initial_command_evidence(sample)
    assert issue is None
    evidence['valid_until'] = now + .045
    observer._publish_split_inputs()
    original_base_state = explore.base_state
    checked = []

    def publish_during_check(odom, status, checked_at):
        checked.append(odom)
        if len(checked) == 1 or update == 'continuous':
            previous = observer.latest['odom']
            pose = (.006, 0., 0.) if update == 'outside_pose' else previous['pose']
            observer.latest['odom'] = dict(previous, message=Value(), pose=pose)
            observer.latest['odom']['message'].twist = odom.twist
            if update == 'epoch':
                observer.epoch += 1
            observer._publish_split_inputs()
        state = original_base_state(odom, status, checked_at)
        if update == 'unsafe_old_state' and len(checked) == 1:
            state['error_code'] = 1
        return state

    monkeypatch.setattr(explore, 'base_state', publish_during_check)
    result = observer.read_command_evidence(sample, evidence, allow_pending=True, nonblocking=True)
    assert result == (evidence['valid_until'] if expected is None else None, expected)
    assert len(checked) == (2 if update in ('fresh_odom', 'continuous') else 1)
    if len(checked) == 2:
        assert checked[0] is not checked[1]


def test_seeded_sensors_keep_their_individual_expiry(monkeypatch):
    import explore
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    observer.extrinsics = observer.depth_extrinsics = observer.depth_projection_config = None
    sample['scan_at'] = now
    sample['_front_coverage_valid_until'] = now + .22
    monkeypatch.setattr(explore, '_monotonic', lambda: now)
    evidence, issue = observer.prepare_initial_command_evidence(sample)
    assert issue is None
    evidence['valid_until'] = now + .08
    captured = []
    observer._split_validation = Value()
    observer._split_validation.seed = lambda *args: captured.extend(args)
    observer.seed_split_validation(sample, evidence)
    assert captured[0] is sample
    assert captured[1]['expires'] == now + .22
    assert captured[2]['expires'] == now + .15


@pytest.mark.parametrize('changed', ('odom', 'scan', 'depth', 'status', 'epoch', 'guard', 'transform', 'calibration'))
def test_slow_scan_check_does_not_block_sensors_or_accept_unchecked_updates(changed, monkeypatch):
    observer, sample = observer_and_sample()
    observer.extrinsics = (.1, 0., .1, 0., 0., 0.)
    observer.depth_projection_config = {'ground_z_m': -.1}
    entered, release, updated = threading.Event(), threading.Event(), threading.Event()
    result = []

    original_scan = observer.latest['scan']

    def slow_scan(probe, _sample, scan, *_args):
        entered.set()
        assert release.wait(2)
        probe._scan_veto_valid_until = _monotonic() + .2
        return scan is original_scan

    monkeypatch.setattr(RosObservation, '_cached_newest_scan_safe_locked', slow_scan)
    checker = threading.Thread(target=lambda: result.append(observer.current_valid_until(sample)))

    def update():
        with observer.lock:
            if changed == 'epoch':
                observer.epoch += 1
            elif changed == 'guard':
                observer.odom_guard.faulted = True
            elif changed == 'transform':
                observer.extrinsics = (.2, 0., .1, 0., 0., 0.)
            elif changed == 'calibration':
                observer.depth_projection_config['ground_z_m'] = -.2
            else:
                old = observer.latest[changed]
                replacement = dict(old)
                replacement['stamp'] = _monotonic()
                if changed == 'status':
                    bad = Value()
                    bad.motion_mode = bad.vehicle_state = 0
                    bad.control_mode, bad.battery_voltage, bad.error_code = 1, 12.2, 1
                    replacement['message'] = bad
                observer.latest[changed] = replacement
        updated.set()

    checker.start()
    try:
        assert entered.wait(1)
        updater = threading.Thread(target=update)
        updater.start()
        completed_during_check = updated.wait(.05)
    finally:
        release.set()
        checker.join(2)
        if 'updater' in locals():
            updater.join(2)
    assert completed_during_check
    assert not checker.is_alive()
    if changed == 'odom':
        assert result and result[0] is not None and result[0] > _monotonic()
    else:
        assert result == [None]


@pytest.mark.parametrize('replace_again', (False, True, 'info'))
def test_new_depth_is_checked_and_a_later_frame_is_rechecked(replace_again, monkeypatch):
    observer, sample = observer_and_sample()
    info = Value()
    info.header = Value()
    info.header.frame_id = 'camera_depth_optical_frame'
    info.width, info.height, info.K = 640, 480, [1.] * 9
    observer.latest['info'] = {'stamp': _monotonic(), 'message': info}
    scan_entered, scan_release = threading.Event(), threading.Event()
    depth_entered, depth_release = threading.Event(), threading.Event()
    checked, result = [], []
    scan_computations = []

    def scan_check(probe, *_args):
        scan_computations.append(True)
        scan_entered.set()
        accepted = scan_release.wait(2)
        probe._scan_veto_valid_until = _monotonic() + .2
        return accepted

    def depth_check(_self, _sample, depth, _odom, _now):
        checked.append(depth)
        depth_entered.set()
        return depth_release.wait(2)

    monkeypatch.setattr(RosObservation, '_cached_newest_scan_safe_locked', scan_check)
    monkeypatch.setattr(RosObservation, '_newest_depth_safe_locked', depth_check)
    checker = threading.Thread(target=lambda: result.append(observer.current_valid_until(sample)))
    checker.start()
    try:
        assert scan_entered.wait(1)
        with observer.lock:
            depth = {'stamp': _monotonic()}
            observer.latest['depth'] = depth
        scan_release.set()
        assert depth_entered.wait(1)
        if replace_again is True:
            with observer.lock:
                observer.latest['depth'] = {'stamp': _monotonic()}
        elif replace_again == 'info':
            replacement = Value()
            replacement.header = info.header
            replacement.width, replacement.height, replacement.K = 640, 480, [2.] * 9
            with observer.lock:
                observer.latest['info'] = {'stamp': _monotonic(), 'message': replacement}
        depth_release.set()
    finally:
        scan_release.set()
        depth_release.set()
        checker.join(2)
    assert checked and checked[0] is depth
    assert not checker.is_alive()
    if replace_again is True:
        assert len(checked) == 2 and checked[-1] is observer.latest['depth']
        assert len(scan_computations) == 1
        assert result and result[0] is not None and result[0] > _monotonic()
    elif replace_again == 'info':
        assert result == [None]
    else:
        assert len(checked) == 1
        assert result and result[0] is not None and result[0] > _monotonic()


def test_depth_projection_uses_newest_frame_after_scan_check(monkeypatch):
    observer, sample = observer_and_sample()
    info = Value()
    info.header = Value()
    info.header.frame_id = 'camera_depth_optical_frame'
    info.width, info.height, info.K = 640, 480, [1.] * 9
    observer.latest['info'] = {'stamp': _monotonic(), 'message': info}
    observer.latest['depth'] = {'stamp': _monotonic()}
    entered, release = threading.Event(), threading.Event()
    checked, result = [], []
    def scan_check(probe, *_args):
        entered.set()
        assert release.wait(2)
        probe._scan_veto_valid_until = _monotonic() + .2
        return True
    def depth_check(_probe, _sample, depth, _odom, _now):
        checked.append(depth)
        return True
    monkeypatch.setattr(RosObservation, '_cached_newest_scan_safe_locked', scan_check)
    monkeypatch.setattr(RosObservation, '_newest_depth_safe_locked', depth_check)
    checker = threading.Thread(target=lambda: result.append(observer.current_valid_until(sample)))
    checker.start()
    try:
        assert entered.wait(1)
        with observer.lock:
            newest = {'stamp': _monotonic()}
            observer.latest['depth'] = newest
    finally:
        release.set()
        checker.join(2)
    assert checked == [newest]
    assert result and result[0] is not None


def test_expired_new_scan_history_cannot_authorize_motion(monkeypatch):
    observer, sample = observer_and_sample()

    def scan_check(probe, *_args):
        probe._scan_veto_valid_until = _monotonic() - .01
        return True

    monkeypatch.setattr(RosObservation, '_cached_newest_scan_safe_locked', scan_check)
    assert observer.current_valid_until(sample) is None


def test_validated_scan_cache_is_reused_for_same_unchanged_scene(monkeypatch):
    import explore
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    observer.extrinsics = (.1, 0., .1, 0., 0., 0.)
    ray = {'start_base': (.1, 0.), 'hit_base': (2., 0.), 'stamp': now}
    sample['scene'] = {'status': 'ok', 'coverage': {'sensor_fresh': True}, 'rays': [ray],
                       'occupied_hits': [{'point_base': (2., 0.), 'stamp': now}]}
    message = Value()
    message.header = Value()
    message.header.frame_id = 'laser_link'
    message.angle_min, message.angle_increment = -.3, .02
    message.range_min, message.range_max, message.time_increment = .1, 8., 0.
    message.ranges = [2.] * 33
    observer.latest['scan']['message'] = message
    calls = []

    def assess(*_args):
        calls.append(True)
        return len(calls) == 1

    monkeypatch.setattr(explore, 'newest_scan_veto', assess)
    assert observer.current_valid_until(sample) is not None
    assert observer.current_valid_until(sample) is not None
    assert len(calls) == 1
    ray['hit_base'] = (1., 0.)
    assert observer.current_valid_until(sample) is None
    assert len(calls) == 2


@pytest.mark.parametrize('future_covered', (True, False))
def test_scan_proof_can_outlive_irrelevant_old_rays_only_after_future_recheck(monkeypatch, future_covered):
    import explore
    observer, sample = observer_and_sample()
    now = sample['snapshot_at']
    observer.extrinsics = (.1, 0., .1, 0., 0., 0.)
    ray = {'start_base': (.1, 0.), 'hit_base': (2., 0.), 'stamp': now - .62}
    sample['scene'] = {'status': 'ok', 'coverage': {'sensor_fresh': True}, 'rays': [ray],
                       'occupied_hits': [{'point_base': (2., 0.), 'stamp': ray['stamp']}]}
    message = Value()
    message.header = Value()
    message.header.frame_id = 'laser_link'
    message.angle_min, message.angle_increment = -.3, .02
    message.range_min, message.range_max, message.time_increment = .1, 8., 0.
    message.ranges = [2.] * 33
    observer.latest['scan']['message'] = message
    observer.latest['scan']['stamp'] = now - .24
    calls = []
    def check(_scan, _extrinsics, _scene, checked_at, *_args):
        calls.append(checked_at)
        return len(calls) == 1 or future_covered
    monkeypatch.setattr(explore, 'newest_scan_veto', check)
    scan = observer.latest['scan']
    odom = observer.latest['odom']
    assert observer._cached_newest_scan_safe_locked(sample, scan, odom, odom, now)
    assert len(calls) == 2 and calls[1] > now + .05
    if future_covered:
        assert observer._scan_veto_valid_until > now + .05
    else:
        assert observer._scan_veto_valid_until <= now + .031


@pytest.mark.parametrize('future_covered', (True, False))
def test_split_scan_covers_measured_scan_period_only_with_live_ray_support(monkeypatch, future_covered):
    import explore
    observer, sample = observer_and_sample()
    observer._split_validation = Value()
    observer.extrinsics = (.1, 0., .1, 0., 0., 0.)
    now = sample['snapshot_at']
    stamp = now - .11
    # The ray expires before the acquisition deadline and was seen within
    # the historical ray window of the new scan.
    ray = {'start_base': (.1, 0.), 'hit_base': (2., 0.), 'stamp': now - .50}
    sample['scene'] = {'status': 'ok', 'coverage': {'sensor_fresh': True}, 'rays': [ray],
                       'occupied_hits': [{'point_base': ray['hit_base'], 'stamp': ray['stamp']}]}
    message = Value()
    message.header = Value()
    message.header.frame_id = 'laser_link'
    message.angle_min, message.angle_increment = -.3, .02
    message.range_min, message.range_max, message.time_increment = .1, 8., 0.
    message.ranges = [2.] * 33
    calls = []

    def check(_scan, _extrinsics, _scene, at, *_args, **_kwargs):
        calls.append(at)
        return len(calls) == 1 or future_covered

    monkeypatch.setattr(explore, 'newest_scan_veto', check)
    scan = {'stamp': stamp, 'message': message}
    odom = observer.latest['odom']
    assert observer._cached_newest_scan_safe_locked(sample, scan, odom, odom, now)
    assert len(calls) == 2
    assert calls[-1] < stamp + .30 and calls[-1] > now + .15
    if future_covered:
        assert observer._scan_veto_valid_until == calls[-1]
    else:
        assert observer._scan_veto_valid_until == ray['stamp'] + .65


@pytest.mark.parametrize('dense', (True, False))
def test_real_scan_recheck_only_extends_proof_when_remaining_rays_cover_corridor(dense):
    import math
    observer, _sample = observer_and_sample()
    observer.extrinsics = (.1, 0., .1, 0., 0., 0.)
    message = Value()
    message.header = Value()
    message.header.frame_id = 'laser_link'
    message.angle_min, message.angle_increment = -.3, .6 / 32.
    message.range_min, message.range_max, message.time_increment = .1, 8., 0.
    latest = (0, 6, 12, 18, 24, 30, 32)
    message.ranges = [2. if dense or i in latest else 0. for i in range(33)]
    rays = []
    for i in (0, 3, 9, 15, 21, 27, 32):
        angle = message.angle_min + i * message.angle_increment
        endpoint = (.1 + 2. * math.cos(angle), 2. * math.sin(angle))
        rays.append({'start_base': (.1, 0.), 'hit_base': endpoint, 'stamp': 10.0})
    scene = {'status': 'ok', 'coverage': {'sensor_fresh': True}, 'rays': rays,
             'occupied_hits': [{'point_base': ray['hit_base'], 'stamp': ray['stamp']} for ray in rays]}
    pose = (0., 0., 0.)
    sample = {'scene': scene, 'pose': pose}
    scan = {'stamp': 10.38, 'message': message}
    odom = {'pose': pose}
    assert observer._cached_newest_scan_safe_locked(sample, scan, odom, odom, 10.6)
    if dense:
        assert observer._scan_veto_valid_until > 10.67
    else:
        assert observer._scan_veto_valid_until <= 10.65


@pytest.mark.parametrize('replace_again', (False, True))
def test_new_scan_is_checked_and_a_later_scan_is_rechecked(replace_again, monkeypatch):
    observer, sample = observer_and_sample()
    first_entered, first_release = threading.Event(), threading.Event()
    second_entered, second_release = threading.Event(), threading.Event()
    checked, result = [], []

    def scan_check(probe, _sample, scan, *_args):
        checked.append(scan)
        if len(checked) == 1:
            first_entered.set()
            accepted = first_release.wait(2)
        else:
            second_entered.set()
            accepted = second_release.wait(2)
        probe._scan_veto_valid_until = _monotonic() + .2
        return accepted

    monkeypatch.setattr(RosObservation, '_cached_newest_scan_safe_locked', scan_check)
    checker = threading.Thread(target=lambda: result.append(observer.current_valid_until(sample)))
    checker.start()
    try:
        assert first_entered.wait(1)
        with observer.lock:
            new_scan = {'stamp': _monotonic()}
            observer.latest['scan'] = new_scan
        first_release.set()
        assert second_entered.wait(1)
        if replace_again:
            with observer.lock:
                observer.latest['scan'] = {'stamp': _monotonic()}
        second_release.set()
    finally:
        first_release.set()
        second_release.set()
        checker.join(2)
    assert len(checked) >= 2 and checked[1] is new_scan
    assert not checker.is_alive()
    if replace_again:
        assert len(checked) == 3 and checked[-1] is observer.latest['scan']
        assert result and result[0] is not None and result[0] > _monotonic()
    else:
        assert len(checked) == 2
        assert result and result[0] is not None and result[0] > _monotonic()


def test_scan_and_depth_updates_are_both_checked_before_acceptance(monkeypatch):
    observer, sample = observer_and_sample()
    first_entered, first_release = threading.Event(), threading.Event()
    seen_scans, seen_depth, result = [], [], []

    def scan_check(probe, _sample, scan, *_args):
        seen_scans.append(scan)
        if len(seen_scans) == 1:
            first_entered.set()
            assert first_release.wait(2)
        probe._scan_veto_valid_until = _monotonic() + .2
        return True

    def depth_check(_probe, _sample, depth, _odom, _now):
        seen_depth.append(depth)
        return True

    monkeypatch.setattr(RosObservation, '_cached_newest_scan_safe_locked', scan_check)
    monkeypatch.setattr(RosObservation, '_newest_depth_safe_locked', depth_check)
    checker = threading.Thread(target=lambda: result.append(observer.current_valid_until(sample)))
    checker.start()
    try:
        assert first_entered.wait(1)
        with observer.lock:
            new_scan = {'stamp': _monotonic()}
            new_depth = {'stamp': _monotonic()}
            observer.latest['scan'], observer.latest['depth'] = new_scan, new_depth
    finally:
        first_release.set()
        checker.join(2)
    assert not checker.is_alive()
    assert len(seen_scans) == 2 and seen_scans[-1] is new_scan
    assert seen_depth == [new_depth]
    assert result and result[0] is not None and result[0] > _monotonic()


@pytest.mark.parametrize('reason', ('scan_generation_changed', 'depth_generation_changed'))
def test_changed_generation_can_be_rechecked_within_original_snapshot_lifetime(reason):
    observer = RosObservation.__new__(RosObservation)
    sample = {'snapshot_at': _monotonic()}
    calls = []

    def assess(_sample, failure):
        calls.append(True)
        if len(calls) == 1:
            failure[0] = reason
            return None
        return _monotonic() + .05

    observer._current_valid_until_once = assess
    result = observer.current_valid_until(sample)
    assert result is not None and len(calls) == 2


def test_hard_sensor_rejection_is_not_retried_even_if_shared_diagnostic_changes():
    observer = RosObservation.__new__(RosObservation)
    sample = {'snapshot_at': _monotonic()}
    calls = []

    def assess(_sample, failure):
        calls.append(True)
        failure[0] = 'new_scan'
        observer.current_last_issue = 'scan_generation_changed'
        return None

    observer._current_valid_until_once = assess
    assert observer.current_valid_until(sample) is None
    assert len(calls) == 1


def test_generation_rechecks_are_bounded_and_never_extend_snapshot_lifetime():
    observer = RosObservation.__new__(RosObservation)
    calls = []

    def assess(_sample, failure):
        calls.append(True)
        failure[0] = 'scan_generation_changed'
        return None

    observer._current_valid_until_once = assess
    assert observer.current_valid_until({'snapshot_at': _monotonic()}) is None
    assert len(calls) == 3
    calls[:] = []
    assert observer.current_valid_until({'snapshot_at': _monotonic() - .3}) is None
    assert len(calls) == 1
