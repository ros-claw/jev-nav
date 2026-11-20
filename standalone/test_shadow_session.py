"""Offline shadow control flow checks: no rospy, network, or device access."""
from __future__ import division

import threading
import time

from relay import _monotonic
from test_control_session import evidence

try:
    import shadow_session
except ImportError:
    shadow_session = None


class FakeSource(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.epoch = 7
        self.pose = (0.0, 0.0, 0.0)
        self.available = True
        self.snapshot_missing = False
        self.pose_available = True
        self.clock_fault = None
        self.odom_fault = None
        self.error = False
        self.block = None

    def stationary_trace(self, start, end):
        with self.lock:
            pose, epoch = self.pose, self.epoch
        stamps = [start]
        index = 1
        while start + index * 0.05 < end:
            stamps.append(start + index * 0.05)
            index += 1
        if end > start:
            stamps.append(end)
        return [{'stamp': stamp, 'seq': index, 'pose': pose, 'epoch': epoch,
                 'linear_x_mps': 0.0, 'linear_y_mps': 0.0, 'angular_z_radps': 0.0}
                for index, stamp in enumerate(stamps)]

    def snapshot(self):
        if self.block is not None:
            self.block.wait(3.0)
        with self.lock:
            if self.error:
                raise RuntimeError('injected observation error')
            if not self.available or self.snapshot_missing:
                return None
            epoch, pose = self.epoch, self.pose
        now = _monotonic()
        value = evidence(now, pose, epoch)
        value.update(image_pose=pose, scan_at=now, status_at=now,
                     depth_at=now, color_at=now)
        return value

    def current_pose(self):
        with self.lock:
            return self.pose if self.available and self.pose_available else None

    def is_current_epoch(self, epoch):
        with self.lock:
            return (self.available and self.epoch == epoch and
                    self.clock_fault is None and self.odom_fault is None)

    def continuity_snapshot(self):
        with self.lock:
            now = _monotonic()
            return {'epoch': self.epoch, 'pose': self.pose if self.pose_available else None,
                    'odom_at': now, 'checked_at': now,
                    'clock_fault': self.clock_fault, 'odom_fault': self.odom_fault}

    def break_epoch(self):
        with self.lock:
            self.epoch += 1


def capture_time(snapshot):
    return min(snapshot[name] for name in ('scan_at', 'odom_at', 'status_at', 'depth_at', 'color_at'))


def forward(snapshot, cancelled):
    assert not cancelled()
    return {'ok': True, 'choice': 'forward', 'captured_at': capture_time(snapshot)}


def wait_for(predicate, timeout=2.0):
    deadline = _monotonic() + timeout
    while _monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError('timed out waiting for shadow condition')


def create(source, model=forward, report=lambda _item: None, **kwargs):
    assert shadow_session is not None, 'shadow session implementation is required'
    return shadow_session.ShadowSession(source, model, report, duration_sec=4.0, **kwargs)


def test_fresh_end_to_end_decision_reports_bounded_would_command():
    source = FakeSource()
    reports = []
    session = create(source, report=reports.append)
    session.start()
    try:
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        state = session.status()
        assert 0 < state['would_command']['linear'] <= 0.03
        assert state['would_command']['angular'] == 0.0
        assert state['heartbeat_count'] >= 2 and state['heartbeat_lateness_sec'] >= 0
        assert not state['latched']
        wait_for(lambda: bool(reports))
        assert all(item['mode'] == 'shadow' and 'frame' not in item for item in reports)
    finally:
        session.stop(timeout=0.8)


def test_blocked_model_never_blocks_heartbeat_and_stop_is_bounded():
    source = FakeSource()
    entered, release = threading.Event(), threading.Event()
    def blocked(snapshot, cancelled):
        entered.set()
        release.wait(3.0)
        return forward(snapshot, cancelled)
    session = create(source, model=blocked, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        before = session.status()['heartbeat_count']
        wait_for(lambda: session.status()['heartbeat_count'] >= before + 4)
        assert session.status()['would_command']['linear'] == 0
        started = _monotonic()
        result = session.stop(timeout=0.4)
        assert _monotonic() - started < 0.8
        assert 'model' in result['alive']
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_decision_timeout_latches_zero_despite_a_stuck_model():
    entered, release = threading.Event(), threading.Event()
    def blocked(snapshot, cancelled):
        entered.set()
        release.wait(3.0)
        return forward(snapshot, cancelled)
    session = create(FakeSource(), model=blocked, decision_timeout_sec=0.2)
    session.start()
    try:
        assert entered.wait(1.0)
        wait_for(lambda: session.status()['latched'])
        assert session.status()['would_command']['linear'] == 0
        assert session.status()['reason'] == 'decision_timeout'
        counters = session.status()['model_diagnostic']
        assert counters['callbacks_started'] == 1 and counters['callbacks_returned'] == 0
        assert counters['session_timeout'] == 1 and counters['discarded_stale'] == 0
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_model_result_from_old_epoch_is_never_used():
    source = FakeSource()
    entered, release = threading.Event(), threading.Event()
    def delayed(snapshot, cancelled):
        entered.set()
        release.wait(2.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed)
    session.start()
    try:
        assert entered.wait(1.0)
        with source.lock:
            source.epoch += 1
        wait_for(lambda: session.status()['latched'])
        release.set()
        time.sleep(0.15)
        assert session.status()['would_command']['linear'] == 0
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_capture_pose_drift_latches_even_with_fresh_observations():
    source = FakeSource()
    session = create(source)
    session.start()
    try:
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        with source.lock:
            source.pose = (0.2, 0.0, 0.0)
        wait_for(lambda: session.status()['latched'])
        assert session.status()['would_command']['linear'] == 0
    finally:
        session.stop(timeout=0.8)


def test_observation_loss_after_motion_latches_and_does_not_resume():
    source = FakeSource()
    session = create(source)
    session.start()
    try:
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        with source.lock:
            source.available = False
        wait_for(lambda: session.status()['latched'])
        with source.lock:
            source.available = True
        time.sleep(0.15)
        assert session.status()['latched']
        assert session.status()['would_command']['linear'] == 0
    finally:
        session.stop(timeout=0.8)


def test_wrong_capture_echo_or_model_exception_latches_zero():
    def wrong(snapshot, cancelled):
        return {'ok': True, 'choice': 'forward', 'captured_at': capture_time(snapshot) - 0.01}
    def raising(snapshot, cancelled):
        raise RuntimeError('injected model failure')
    for model in (wrong, raising):
        session = create(FakeSource(), model=model)
        session.start()
        try:
            wait_for(lambda current=session: current.status()['latched'])
            assert session.status()['would_command']['linear'] == 0
        finally:
            session.stop(timeout=0.8)


def test_blocked_reporter_does_not_stop_the_heartbeat():
    entered, release = threading.Event(), threading.Event()
    def blocked(_item):
        entered.set()
        release.wait(3.0)
    session = create(FakeSource(), report=blocked)
    session.start()
    try:
        assert entered.wait(1.0)
        before = session.status()['heartbeat_count']
        wait_for(lambda: session.status()['heartbeat_count'] >= before + 4)
        result = session.stop(timeout=0.4)
        assert 'report' in result['alive']
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_clock_regression_latches_zero_and_cannot_restart():
    lock = threading.Lock()
    offset = [0.0]
    def clock():
        with lock:
            return _monotonic() + offset[0]
    session = create(FakeSource(), clock=clock)
    session.start()
    try:
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        with lock:
            offset[0] = -1.0
        wait_for(lambda: session.status()['latched'])
        assert session.status()['would_command']['linear'] == 0
    finally:
        session.stop(timeout=0.8)
    try:
        session.start()
    except RuntimeError:
        pass
    else:
        raise AssertionError('a stopped session must not restart')


def test_observation_error_immediately_clears_cached_would_command():
    source = FakeSource()
    session = create(source)
    session.start()
    entered, release = threading.Event(), threading.Event()
    try:
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        real_step = session.control.step
        def blocked_step(*args):
            entered.set()
            release.wait(2.0)
            return real_step(*args)
        session.control.step = blocked_step
        assert entered.wait(1.0)
        with source.lock:
            source.error = True
        wait_for(lambda: session.latched)
        state = session.status()
        assert state['latched'] and state['would_command']['linear'] == 0
        assert state['reason'] == 'observation_error'
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_heartbeat_schedules_against_absolute_deadlines_during_slow_processing():
    source = FakeSource()
    entered, release = threading.Event(), threading.Event()
    def pending(snapshot, cancelled):
        entered.set()
        release.wait(3.0)
        return forward(snapshot, cancelled)
    session = create(source, model=pending, decision_timeout_sec=3.0)
    session.start()
    times = []
    try:
        assert entered.wait(1.0)
        real_step = session.control.step
        def slow_step(*args):
            times.append(_monotonic())
            time.sleep(0.035)
            return real_step(*args)
        session.control.step = slow_step
        wait_for(lambda: len(times) >= 9, timeout=2.0)
        intervals = [times[i] - times[i - 1] for i in range(1, len(times))]
        assert sum(intervals) / len(intervals) < 0.075
        state = session.status()
        assert state['heartbeat_interval_samples'] >= 2
        assert 0 < state['heartbeat_interval_mean_sec'] <= state['heartbeat_interval_max_sec']
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_delayed_report_cannot_emit_queued_forward_after_fault():
    source = FakeSource()
    entered, release_report = threading.Event(), threading.Event()
    entered_step, release_step = threading.Event(), threading.Event()
    delivered = []
    def report(item):
        delivered.append(item)
        if not entered.is_set():
            entered.set()
            release_report.wait(2.0)
    session = create(source, report=report)
    session.start()
    try:
        assert entered.wait(1.0)
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        real_step = session.control.step
        def blocked_step(*args):
            entered_step.set()
            release_step.wait(2.0)
            return real_step(*args)
        session.control.step = blocked_step
        assert entered_step.wait(1.0)
        with source.lock:
            source.error = True
        wait_for(lambda: session.latched)
        release_report.set()
        wait_for(lambda: len(delivered) >= 2)
        assert delivered[-1]['latched'] and delivered[-1]['would_command']['linear'] == 0
    finally:
        release_step.set()
        release_report.set()
        session.stop(timeout=0.8)


def test_late_model_result_cannot_win_a_race_against_timeout():
    source = FakeSource()
    session = create(source, decision_timeout_sec=0.2, clock=lambda: 10.3)
    snapshot = evidence(10.0, epoch=7)
    snapshot.update(image_pose=(0.0, 0.0, 0.0), scan_at=10.0, status_at=10.0,
                    depth_at=10.0, color_at=10.0)
    record = {'sample': snapshot, 'epoch': 7, 'image_pose': (0.0, 0.0, 0.0),
              'captured_at': 10.0, 'generation': 0}
    session.session_epoch = 7
    session.pending = (1, 10.0)
    session.model_jobs.put_nowait((1, 0, record))
    thread = threading.Thread(target=session._model)
    thread.daemon = True
    thread.start()
    try:
        wait_for(lambda: session.pending is None)
        assert session.latched and session.latch_reason == 'decision_timeout'
        assert session.decision is None
    finally:
        session.stop_event.set()
        thread.join(1.0)


def test_stop_before_start_prevents_any_later_start():
    session = create(FakeSource())
    session.stop(timeout=0.1)
    try:
        session.start()
    except RuntimeError:
        pass
    else:
        raise AssertionError('a stopped session must not start')


def test_decisions_adapter_builds_evidence_without_starting_external_work():
    class Bridge(object):
        def imgmsg_to_cv2(self, image, desired_encoding):
            assert image == 'fake-color' and desired_encoding == 'bgr8'
            return 'fake-array'
    class Source(object):
        bridge = Bridge()
    class Encoded(object):
        def tostring(self):
            return b'\xff\xd8fake-jpeg'
    class Cv(object):
        IMWRITE_JPEG_QUALITY = 1
        @staticmethod
        def imencode(extension, image, parameters):
            assert extension == '.jpg' and image == 'fake-array' and parameters == [1, 75]
            return True, Encoded()
    class Scan(object):
        ranges = [2.0] * 33
        angle_min = -0.3
        angle_increment = 0.6 / 32
        range_min = 0.1
        range_max = 10.0
    now = _monotonic()
    snapshot = evidence(now)
    snapshot.update(image_pose=(0, 0, 0), scan_at=now, status_at=now,
                    depth_at=now, color_at=now, color='fake-color', scan=Scan(),
                    laser_extrinsics=(0.105, 0.0, 0.0))
    calls = []
    def invoke(payload, cancelled):
        calls.append(payload)
        assert not cancelled()
        return {'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']}
    request = shadow_session.make_decisions_request(Source(), invoke, cv2_module=Cv())
    result = request(snapshot, lambda: False)
    assert result['captured_at'] == now
    assert len(calls) == 1 and calls[0]['captured_at'] == now
    assert calls[0]['laser']['forward_view']['rear_total_rays'] == 0
    assert request(snapshot, lambda: True) is None
    assert len(calls) == 1


def test_observer_uses_deadlines_without_adding_a_fixed_wait_to_slow_samples():
    class SlowSource(FakeSource):
        def __init__(self):
            FakeSource.__init__(self)
            self.starts = []
        def snapshot(self):
            self.starts.append(_monotonic())
            time.sleep(0.035)
            return FakeSource.snapshot(self)
    source = SlowSource()
    session = create(source, model=lambda snapshot, cancelled: None)
    session.start()
    try:
        wait_for(lambda: len(source.starts) >= 9, timeout=2.0)
        intervals = [source.starts[i] - source.starts[i - 1] for i in range(1, len(source.starts))]
        assert 0.08 < sum(intervals) / len(intervals) < 0.105
    finally:
        session.stop(timeout=0.8)


def test_overrun_sampler_continues_without_an_extra_full_period():
    class SlowSource(FakeSource):
        def __init__(self):
            FakeSource.__init__(self)
            self.starts = []
        def snapshot(self):
            self.starts.append(_monotonic())
            time.sleep(0.065)
            return FakeSource.snapshot(self)
    source = SlowSource()
    session = create(source, model=lambda snapshot, cancelled: None)
    session.start()
    try:
        wait_for(lambda: len(source.starts) >= 8, timeout=2.0)
        intervals = [source.starts[i] - source.starts[i - 1] for i in range(1, len(source.starts))]
        assert 0.06 <= sum(intervals) / len(intervals) < 0.095
    finally:
        session.stop(timeout=0.8)


def test_repeated_observation_errors_do_not_busy_loop():
    class FaultSource(FakeSource):
        def __init__(self):
            FakeSource.__init__(self)
            self.calls = 0
        def snapshot(self):
            self.calls += 1
            raise RuntimeError('sensor failed')
    source = FaultSource()
    session = create(source)
    session.start()
    try:
        wait_for(lambda: session.status()['latched'])
        time.sleep(0.25)
        assert 3 <= source.calls <= 9
        assert session.status()['would_command']['linear'] == 0
    finally:
        session.stop(timeout=0.8)


class OccasionallyStaleSource(FakeSource):
    def __init__(self):
        FakeSource.__init__(self)
        self.stale = False
    def snapshot(self):
        value = FakeSource.snapshot(self)
        if value is not None:
            with self.lock:
                stale = self.stale
            if stale:
                value['snapshot_at'] -= 0.2
                value['odom_at'] -= 0.2
        return value


def test_stale_sample_before_any_decision_or_motion_waits_for_fresh_observation():
    source = OccasionallyStaleSource()
    entered, release = threading.Event(), threading.Event()
    release_followup = threading.Event()
    calls = []
    def delayed_once(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        if len(calls) == 1:
            entered.set()
            release.wait(1.0)
        else:
            release_followup.wait(1.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed_once, decision_interval_sec=0.05, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with session.lock:
            original_generation = session.generation
        with source.lock:
            source.stale = True
        wait_for(lambda: session.status()['reason'] == 'waiting_for_fresh_observation')
        assert not session.status()['latched']
        assert session.status()['would_command']['linear'] == 0
        release.set()
        wait_for(lambda: session.status()['model_status'] == 'waiting_for_fresh_observation')
        assert session.status()['would_command']['linear'] == 0
        assert len(calls) == 1
        with source.lock:
            source.stale = False
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        assert session.decision['captured_at'] == calls[0]
        assert session.generation == original_generation and not session.status()['latched']
    finally:
        release.set()
        release_followup.set()
        session.stop(timeout=0.8)


def test_stale_after_accepted_forward_blocked_by_depth_keeps_waiting():
    class BlockedDepth(OccasionallyStaleSource):
        def snapshot(self):
            value = OccasionallyStaleSource.snapshot(self)
            value['depth']['min_m'] = 0.4
            return value
    source = BlockedDepth()
    session = create(source)
    session.start()
    try:
        wait_for(lambda: session.status()['model_status'] == 'accepted')
        assert session.status()['would_command']['linear'] == 0
        assert session.status()['reason'] == 'depth_unavailable_or_close'
        with source.lock:
            source.stale = True
        wait_for(lambda: session.status()['reason'] == 'waiting_for_fresh_observation')
        assert not session.status()['latched']
        assert session.status()['would_command']['linear'] == 0
    finally:
        session.stop(timeout=0.8)


def test_epoch_change_while_waiting_for_fresh_data_still_latches():
    source = OccasionallyStaleSource()
    entered, release = threading.Event(), threading.Event()
    def delayed(snapshot, cancelled):
        entered.set()
        release.wait(1.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with source.lock:
            source.stale = True
        wait_for(lambda: session.status()['reason'] == 'waiting_for_fresh_observation')
        with source.lock:
            source.epoch += 1
        wait_for(lambda: session.status()['latched'])
        with source.lock:
            source.stale = False
        release.set()
        time.sleep(0.1)
        assert session.status()['latched'] and session.status()['would_command']['linear'] == 0
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_stale_and_pose_mismatch_is_not_treated_as_recoverable_staleness():
    class MismatchedSource(OccasionallyStaleSource):
        def snapshot(self):
            value = OccasionallyStaleSource.snapshot(self)
            if value is not None:
                with self.lock:
                    stale = self.stale
                if stale:
                    value['pose'] = (0.02, 0.0, 0.0)
            return value
    source = MismatchedSource()
    entered, release = threading.Event(), threading.Event()
    def delayed(snapshot, cancelled):
        entered.set()
        release.wait(1.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with source.lock:
            source.stale = True
        wait_for(lambda: session.status()['latched'], timeout=0.7)
        assert session.status()['reason'] == 'observation_pose_changed'
        assert session.status()['would_command']['linear'] == 0
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_model_error_while_waiting_for_fresh_observation_latches():
    source = OccasionallyStaleSource()
    entered, release = threading.Event(), threading.Event()
    def failing(snapshot, cancelled):
        entered.set()
        release.wait(1.0)
        raise RuntimeError('injected model error')
    session = create(source, model=failing, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with source.lock:
            source.stale = True
        wait_for(lambda: session.status()['reason'] == 'waiting_for_fresh_observation')
        release.set()
        wait_for(lambda: session.status()['latched'])
        assert session.status()['reason'] == 'decision_error'
        assert session.status()['would_command']['linear'] == 0
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_stale_sample_after_motion_still_latches():
    source = OccasionallyStaleSource()
    session = create(source)
    session.start()
    try:
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        with source.lock:
            source.stale = True
        wait_for(lambda: session.status()['latched'])
        assert session.status()['reason'] == 'observation_stale'
        assert session.status()['would_command']['linear'] == 0
    finally:
        session.stop(timeout=0.8)


def test_blocked_sampler_does_not_hold_heartbeat_or_status_lock():
    entered, release = threading.Event(), threading.Event()
    class BlockedSource(FakeSource):
        def snapshot(self):
            entered.set()
            release.wait(2.0)
            return FakeSource.snapshot(self)
    session = create(BlockedSource())
    session.start()
    try:
        assert entered.wait(1.0)
        before = session.status()['heartbeat_count']
        wait_for(lambda: session.status()['heartbeat_count'] >= before + 4)
        started = _monotonic()
        stopped = session.stop(timeout=0.4)
        assert _monotonic() - started < 0.8
        assert 'observation' in stopped['alive']
    finally:
        release.set()
        session.stop(timeout=0.8)



def test_prolonged_stale_observation_discards_inflight_decision():
    source = OccasionallyStaleSource()
    source.stationary_trace = None
    entered, release = threading.Event(), threading.Event()
    calls = []
    def delayed_once(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        if len(calls) == 1:
            entered.set()
            release.wait(1.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed_once, decision_interval_sec=0.05, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with session.lock:
            generation = session.generation
        with source.lock:
            source.stale = True
        wait_for(lambda: session.status()['reason'] == 'waiting_for_fresh_observation')
        wait_for(lambda: session.generation > generation, timeout=0.8)
        assert not session.status()['latched']
        source.stationary_trace = lambda start, end: FakeSource.stationary_trace(source, start, end)
        with source.lock:
            source.stale = False
        release.set()
        wait_for(lambda: len(calls) >= 2 and session.status()['would_command']['linear'] > 0)
        assert not session.status()['latched']
        assert session.status()['model_diagnostic']['discarded_stale'] >= 1
        assert session.status()['model_diagnostic']['session_timeout'] == 0
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_pending_decision_pose_over_budget_latches_before_model_returns():
    source = FakeSource()
    entered, release = threading.Event(), threading.Event()
    def delayed(snapshot, cancelled):
        entered.set()
        release.wait(1.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with source.lock:
            source.pose = (0.2, 0.0, 0.0)
        wait_for(lambda: session.status()['latched'], timeout=0.7)
        assert session.status()['reason'] == 'decision_pose_changed'
        release.set()
        time.sleep(0.1)
        assert session.status()['would_command']['linear'] == 0
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_two_second_model_survives_intermittent_short_stale_samples():
    class IntermittentSource(FakeSource):
        def __init__(self):
            FakeSource.__init__(self)
            self.count = 0
        def snapshot(self):
            value = FakeSource.snapshot(self)
            self.count += 1
            if value is not None and self.count % 3 == 0:
                value['snapshot_at'] -= 0.2
                value['odom_at'] -= 0.2
            return value
    source = IntermittentSource()
    calls = []
    def slow_model(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        time.sleep(2.1)
        return forward(snapshot, cancelled)
    session = create(source, model=slow_model, decision_timeout_sec=3.0)
    session.start()
    try:
        wait_for(lambda: session.status()['would_command']['linear'] > 0, timeout=3.5)
        assert not session.status()['latched']
        assert session.decision['captured_at'] == calls[0]
        assert session.generation == 0
    finally:
        session.stop(timeout=0.8)



def test_brief_none_preserves_inflight_model_but_waits_for_fresh_snapshot():
    source = FakeSource()
    entered, release, release_followup = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def delayed(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        if len(calls) == 1:
            entered.set()
            release.wait(1.0)
        else:
            release_followup.wait(1.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_interval_sec=0.05, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with session.lock:
            generation = session.generation
        with source.lock:
            source.snapshot_missing = True
        wait_for(lambda: session.status()['reason'] == 'waiting_for_fresh_observation')
        assert not session.status()['latched'] and session.status()['would_command']['linear'] == 0
        release.set()
        wait_for(lambda: session.status()['model_status'] == 'waiting_for_fresh_observation')
        assert session.status()['would_command']['linear'] == 0
        with source.lock:
            source.snapshot_missing = False
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        assert session.decision['captured_at'] == calls[0]
        assert session.generation == generation and not session.status()['latched']
    finally:
        release.set()
        release_followup.set()
        session.stop(timeout=0.8)


def test_snapshot_none_after_nonzero_always_latches():
    source = FakeSource()
    session = create(source)
    session.start()
    try:
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        with source.lock:
            source.snapshot_missing = True
        wait_for(lambda: session.status()['latched'])
        assert session.status()['would_command']['linear'] == 0
        with source.lock:
            source.snapshot_missing = False
        time.sleep(0.1)
        assert session.status()['latched'] and session.status()['would_command']['linear'] == 0
    finally:
        session.stop(timeout=0.8)


def test_prolonged_none_discards_inflight_result_and_restarts_waiting():
    source = FakeSource()
    source.stationary_trace = None
    entered, release = threading.Event(), threading.Event()
    calls = []
    def delayed(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        if len(calls) == 1:
            entered.set()
            release.wait(1.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_interval_sec=0.05, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with session.lock:
            generation = session.generation
        with source.lock:
            source.snapshot_missing = True
        wait_for(lambda: session.generation > generation, timeout=0.8)
        assert not session.status()['latched'] and session.status()['would_command']['linear'] == 0
        source.stationary_trace = lambda start, end: FakeSource.stationary_trace(source, start, end)
        with source.lock:
            source.snapshot_missing = False
        release.set()
        wait_for(lambda: len(calls) >= 2 and session.status()['would_command']['linear'] > 0)
        assert session.decision['captured_at'] != calls[0]
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_none_epoch_change_discards_old_model_and_restarts_with_new_epoch():
    source = FakeSource()
    entered, release = threading.Event(), threading.Event()
    calls = []
    def delayed(snapshot, cancelled):
        calls.append((snapshot['epoch'], capture_time(snapshot)))
        if len(calls) == 1:
            entered.set()
            release.wait(1.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_interval_sec=0.05, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with session.lock:
            generation = session.generation
        with source.lock:
            source.snapshot_missing = True
            source.epoch += 1
        wait_for(lambda: session.generation > generation, timeout=0.8)
        assert not session.status()['latched'] and session.status()['would_command']['linear'] == 0
        with source.lock:
            source.snapshot_missing = False
        release.set()
        wait_for(lambda: len(calls) >= 2 and session.status()['would_command']['linear'] > 0)
        assert calls[0][0] == 7 and session.decision['epoch'] == 8
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_none_without_valid_pose_discards_model_before_waiting_again():
    source = FakeSource()
    entered, release = threading.Event(), threading.Event()
    calls = []
    def delayed(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        if len(calls) == 1:
            entered.set()
            release.wait(1.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_interval_sec=0.05, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with session.lock:
            generation = session.generation
        with source.lock:
            source.snapshot_missing = True
            source.pose_available = False
        wait_for(lambda: session.generation > generation, timeout=0.8)
        assert not session.status()['latched'] and session.status()['would_command']['linear'] == 0
        with source.lock:
            source.pose_available = True
            source.snapshot_missing = False
        release.set()
        wait_for(lambda: len(calls) >= 2 and session.status()['would_command']['linear'] > 0)
        assert session.decision['captured_at'] != calls[0]
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_guard_fault_during_none_latches_even_before_motion():
    for name in ('clock_fault', 'odom_fault'):
        source = FakeSource()
        entered, release = threading.Event(), threading.Event()
        def delayed(snapshot, cancelled, ready=entered, finish=release):
            ready.set()
            finish.wait(1.0)
            return forward(snapshot, cancelled)
        session = create(source, model=delayed, decision_timeout_sec=1.0)
        session.start()
        try:
            assert entered.wait(1.0)
            with source.lock:
                source.snapshot_missing = True
                setattr(source, name, 'injected')
            wait_for(lambda current=session: current.status()['latched'])
            assert session.status()['would_command']['linear'] == 0
        finally:
            release.set()
            session.stop(timeout=0.8)


def test_none_pose_over_budget_discards_before_restarting():
    source = FakeSource()
    entered, release = threading.Event(), threading.Event()
    calls = []
    def delayed(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        if len(calls) == 1:
            entered.set()
            release.wait(1.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_interval_sec=0.05, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with session.lock:
            generation = session.generation
        with source.lock:
            source.snapshot_missing = True
            source.pose = (0.2, 0.0, 0.0)
        wait_for(lambda: session.generation > generation, timeout=0.8)
        assert not session.status()['latched'] and session.status()['would_command']['linear'] == 0
        with source.lock:
            source.pose = (0.0, 0.0, 0.0)
            source.snapshot_missing = False
        release.set()
        wait_for(lambda: len(calls) >= 2 and session.status()['would_command']['linear'] > 0)
        assert session.decision['captured_at'] != calls[0]
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_legacy_source_without_atomic_continuity_cannot_authorize_motion():
    class LegacySource(FakeSource):
        continuity_snapshot = None
    source = LegacySource()
    calls = []
    def model(snapshot, cancelled):
        calls.append(1)
        return forward(snapshot, cancelled)
    session = create(source, model=model)
    session.start()
    try:
        wait_for(lambda: session.status()['latched'])
        assert session.status()['would_command']['linear'] == 0
        assert not calls
    finally:
        session.stop(timeout=0.8)


def test_two_second_model_survives_intermittent_proven_none_snapshots():
    class IntermittentNone(FakeSource):
        def __init__(self):
            FakeSource.__init__(self)
            self.count = 0
            self.missing_count = 0
            self.inject_missing = True
        def snapshot(self):
            self.count += 1
            with self.lock:
                inject = self.inject_missing
            if inject and self.count % 3 == 0:
                self.missing_count += 1
                return None
            return FakeSource.snapshot(self)
    source = IntermittentNone()
    calls, reports = [], []
    def slow_model(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        time.sleep(2.1)
        with source.lock:
            source.inject_missing = False
        return forward(snapshot, cancelled)
    session = create(source, model=slow_model, report=reports.append, decision_timeout_sec=3.0)
    session.start()
    try:
        wait_for(lambda: any(item['would_command']['linear'] > 0 for item in reports), timeout=3.5)
        assert len(calls) >= 1 and source.missing_count >= 5
        assert any(item['would_command']['linear'] > 0 and not item['latched'] for item in reports)
    finally:
        session.stop(timeout=0.8)


def test_incomplete_continuity_proof_fails_closed():
    class BadContinuity(FakeSource):
        def __init__(self):
            FakeSource.__init__(self)
            self.bad = False
        def continuity_snapshot(self):
            proof = FakeSource.continuity_snapshot(self)
            with self.lock:
                bad = self.bad
            if bad:
                del proof['clock_fault']
            return proof
    source = BadContinuity()
    entered, release = threading.Event(), threading.Event()
    def delayed(snapshot, cancelled):
        entered.set()
        release.wait(1.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_timeout_sec=1.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with source.lock:
            source.snapshot_missing = True
            source.bad = True
        wait_for(lambda: session.status()['latched'], timeout=0.7)
        assert session.status()['reason'] == 'observation_error'
        assert session.status()['would_command']['linear'] == 0
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_observation_exception_diagnostic_contains_only_safe_stage_and_type():
    import json
    secret = 'private-api-key-and-image-bytes-must-not-appear'
    class FaultSnapshot(FakeSource):
        def snapshot(self):
            raise RuntimeError(secret)
    class FaultPose(FakeSource):
        def current_pose(self):
            raise KeyError(secret)
    class FaultEpoch(FakeSource):
        def is_current_epoch(self, epoch):
            raise OSError(secret)
    class FaultContinuity(FakeSource):
        def snapshot(self):
            return None
        def continuity_snapshot(self):
            raise ValueError(secret)
    class SecretNamedError(Exception):
        pass
    class FaultUnknown(FakeSource):
        def snapshot(self):
            raise SecretNamedError(secret)
    cases = ((FaultSnapshot, 'snapshot', 'RuntimeError'),
             (FaultPose, 'current_pose', 'KeyError'),
             (FaultEpoch, 'source_epoch_check', 'OSError'),
             (FaultContinuity, 'continuity_snapshot', 'ValueError'),
             (FaultUnknown, 'snapshot', 'OtherException'))
    for source_type, stage, exception_type in cases:
        reports = []
        session = create(source_type(), report=reports.append)
        session.start()
        try:
            wait_for(lambda current=session: current.status()['latched'])
            state = session.status()
            assert state['diagnostic'] == {'stage': stage, 'exception_type': exception_type}
            assert secret not in json.dumps(state)
            assert all(secret not in json.dumps(item) for item in reports)
        finally:
            session.stop(timeout=0.8)


def test_missing_current_pose_before_motion_waits_and_can_recover():
    source = FakeSource()
    with source.lock:
        source.pose_available = False
    session = create(source)
    session.start()
    try:
        wait_for(lambda: session.status()['reason'] == 'waiting_for_fresh_observation')
        assert not session.status()['latched']
        assert session.status()['would_command']['linear'] == 0
        assert session.status()['diagnostic'] == {'stage': 'current_pose', 'exception_type': None}
        with source.lock:
            source.pose_available = True
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        assert not session.status()['latched']
    finally:
        session.stop(timeout=0.8)


def test_missing_current_pose_after_nonzero_latches():
    source = FakeSource()
    session = create(source)
    session.start()
    try:
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        with source.lock:
            source.pose_available = False
        wait_for(lambda: session.status()['latched'])
        assert session.status()['would_command']['linear'] == 0
        assert session.status()['reason'] == 'observation_unavailable'
        assert session.status()['diagnostic'] == {'stage': 'current_pose', 'exception_type': None}
        with source.lock:
            source.pose_available = True
        time.sleep(0.1)
        assert session.status()['latched'] and session.status()['would_command']['linear'] == 0
    finally:
        session.stop(timeout=0.8)


def test_shadow_accepts_015_second_sample_only_with_fresh_live_odom():
    class AgedSample(FakeSource):
        def snapshot(self):
            value = FakeSource.snapshot(self)
            value['snapshot_at'] -= 0.11
            value['odom_at'] -= 0.11
            return value
    source = AgedSample()
    session = create(source)
    session.start()
    try:
        wait_for(lambda: session.status()['would_command']['linear'] > 0, timeout=3.0)
        assert session.control.snapshot_max_age_sec == 0.15
        assert not session.status()['latched']
    finally:
        session.stop(timeout=0.8)


def test_shadow_never_authorizes_from_live_odom_older_than_01_second():
    class OldLiveOdom(FakeSource):
        def continuity_snapshot(self):
            proof = FakeSource.continuity_snapshot(self)
            proof['odom_at'] -= 0.11
            return proof
    calls = []
    def model(snapshot, cancelled):
        calls.append(1)
        return forward(snapshot, cancelled)
    session = create(OldLiveOdom(), model=model)
    session.start()
    try:
        time.sleep(0.4)
        assert session.status()['would_command']['linear'] == 0
        assert not calls
    finally:
        session.stop(timeout=0.8)


def test_shadow_never_authorizes_from_sample_older_than_015_second():
    class TooOldSample(FakeSource):
        def snapshot(self):
            value = FakeSource.snapshot(self)
            value['snapshot_at'] -= 0.16
            value['odom_at'] -= 0.16
            return value
    calls = []
    def model(snapshot, cancelled):
        calls.append(1)
        return forward(snapshot, cancelled)
    session = create(TooOldSample(), model=model)
    session.start()
    try:
        time.sleep(0.4)
        assert session.status()['would_command']['linear'] == 0
        assert not calls
    finally:
        session.stop(timeout=0.8)


def test_shadow_still_rejects_sample_pose_difference_over_001_metre():
    class DisplacedSample(FakeSource):
        def snapshot(self):
            value = FakeSource.snapshot(self)
            value['snapshot_at'] -= 0.11
            value['odom_at'] -= 0.11
            value['pose'] = (0.011, 0.0, 0.0)
            return value
    session = create(DisplacedSample())
    session.start()
    try:
        wait_for(lambda: session.status()['latched'])
        assert session.status()['would_command']['linear'] == 0
    finally:
        session.stop(timeout=0.8)



def test_live_odom_age_is_rechecked_after_slow_policy_work():
    session = create(FakeSource())
    session.start()
    try:
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        real_step = session.control.step
        def slow_step(*args):
            time.sleep(0.12)
            return real_step(*args)
        session.control.step = slow_step
        wait_for(lambda: session.status()['latched'], timeout=1.0)
        assert session.status()['would_command']['linear'] == 0
        assert session.status()['reason'] == 'observation_stale'
    finally:
        session.stop(timeout=0.8)


def test_sampler_target_reduces_duty_cycle_without_slowing_heartbeat():
    class SlowSource(FakeSource):
        def __init__(self):
            FakeSource.__init__(self)
            self.starts = []
        def snapshot(self):
            self.starts.append(_monotonic())
            time.sleep(0.065)
            return FakeSource.snapshot(self)
    source = SlowSource()
    release = threading.Event()
    def pending(snapshot, cancelled):
        release.wait(2.0)
        return forward(snapshot, cancelled)
    session = create(source, model=pending, decision_timeout_sec=2.0)
    session.start()
    try:
        wait_for(lambda: len(source.starts) >= 8)
        intervals = [source.starts[i] - source.starts[i - 1] for i in range(1, len(source.starts))]
        assert 0.085 <= sum(intervals) / len(intervals) < 0.12
        assert 0.04 <= session.status()['heartbeat_interval_mean_sec'] <= 0.065
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_sampler_skips_missed_slots_to_avoid_saturating_cpu():
    class OverrunSource(FakeSource):
        def __init__(self):
            FakeSource.__init__(self)
            self.starts = []
        def snapshot(self):
            self.starts.append(_monotonic())
            time.sleep(0.125)
            return FakeSource.snapshot(self)
    source = OverrunSource()
    session = create(source, model=lambda snapshot, cancelled: None)
    session.start()
    try:
        wait_for(lambda: len(source.starts) >= 5)
        intervals = [source.starts[i] - source.starts[i - 1] for i in range(1, len(source.starts))]
        assert 0.16 <= sum(intervals) / len(intervals) < 0.22
    finally:
        session.stop(timeout=0.8)


def test_long_stale_gap_retains_model_only_until_fresh_sample_and_trace():
    source = OccasionallyStaleSource()
    entered, release, followup = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def delayed(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        if len(calls) == 1:
            entered.set()
            release.wait(2.0)
        else:
            followup.wait(2.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_interval_sec=0.05, decision_timeout_sec=2.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with session.lock:
            generation = session.generation
        with source.lock:
            source.stale = True
        wait_for(lambda: session.status()['reason'] == 'waiting_for_fresh_observation')
        time.sleep(0.4)
        assert session.generation == generation
        release.set()
        wait_for(lambda: session.status()['model_status'] == 'waiting_for_fresh_observation')
        assert session.status()['would_command']['linear'] == 0
        with source.lock:
            source.stale = False
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        assert session.decision['captured_at'] == calls[0]
        assert session.generation == generation
    finally:
        release.set()
        followup.set()
        session.stop(timeout=0.8)


def test_long_none_gap_waits_for_fresh_sample_and_full_trace():
    source = FakeSource()
    entered, release, followup = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def delayed(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        if len(calls) == 1:
            entered.set()
            release.wait(2.0)
        else:
            followup.wait(2.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_interval_sec=0.05, decision_timeout_sec=2.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with session.lock:
            generation = session.generation
        with source.lock:
            source.snapshot_missing = True
        wait_for(lambda: session.status()['reason'] == 'waiting_for_fresh_observation')
        time.sleep(0.4)
        assert session.generation == generation
        release.set()
        wait_for(lambda: session.status()['model_status'] == 'waiting_for_fresh_observation')
        assert session.status()['would_command']['linear'] == 0
        with source.lock:
            source.snapshot_missing = False
        wait_for(lambda: session.status()['would_command']['linear'] > 0)
        assert session.decision['captured_at'] == calls[0]
        assert session.generation == generation
    finally:
        release.set()
        followup.set()
        session.stop(timeout=0.8)


def test_incomplete_or_moving_history_never_authorizes_retained_model():
    class BadTrace(FakeSource):
        def __init__(self, fault):
            FakeSource.__init__(self)
            self.fault = fault
        def stationary_trace(self, start, end):
            trace = FakeSource.stationary_trace(self, start, end)
            if self.fault == 'missing':
                return []
            if self.fault == 'gap':
                return [trace[0], trace[-1]] if len(trace) > 3 else []
            if self.fault == 'velocity':
                trace[len(trace) // 2]['linear_y_mps'] = 0.006
            if self.fault == 'pose':
                trace[len(trace) // 2]['pose'] = (0.01, 0.0, 0.0)
            if self.fault == 'epoch':
                trace[len(trace) // 2]['epoch'] += 1
            return trace
    for fault in ('missing', 'gap', 'velocity', 'pose', 'epoch'):
        release = threading.Event()
        def delayed(snapshot, cancelled, resume=release):
            resume.wait(2.0)
            return forward(snapshot, cancelled)
        session = create(BadTrace(fault), model=delayed, decision_timeout_sec=2.0)
        session.start()
        try:
            wait_for(lambda current=session: current.status()['model_status'] == 'pending')
            time.sleep(0.25)
            release.set()
            time.sleep(0.4)
            assert session.status()['would_command']['linear'] == 0, fault
            assert session.status()['model_status'] != 'accepted', fault
        finally:
            release.set()
            session.stop(timeout=0.8)


def test_missing_continuity_field_after_trace_never_authorizes_model():
    class IncompleteAfterTrace(FakeSource):
        def __init__(self):
            FakeSource.__init__(self)
            self.traced = False
        def stationary_trace(self, start, end):
            self.traced = True
            return FakeSource.stationary_trace(self, start, end)
        def continuity_snapshot(self):
            proof = FakeSource.continuity_snapshot(self)
            if self.traced:
                del proof['clock_fault']
            return proof
    source = IncompleteAfterTrace()
    reports = []
    session = create(source, report=reports.append)
    session.start()
    try:
        wait_for(lambda: source.traced)
        wait_for(lambda: session.status()['latched'])
        assert session.status()['would_command']['linear'] == 0
        assert all(item['would_command']['linear'] == 0 for item in reports)
    finally:
        session.stop(timeout=0.8)


def test_blocked_trace_does_not_block_heartbeat_or_accept_epoch_change():
    entered, release = threading.Event(), threading.Event()
    class BlockedTrace(FakeSource):
        def stationary_trace(self, start, end):
            entered.set()
            release.wait(2.0)
            return FakeSource.stationary_trace(self, start, end)
    source = BlockedTrace()
    session = create(source)
    session.start()
    try:
        assert entered.wait(1.0)
        count = session.status()['heartbeat_count']
        wait_for(lambda: session.status()['heartbeat_count'] >= count + 4)
        assert session.status()['would_command']['linear'] == 0
        with source.lock:
            source.epoch += 1
        release.set()
        wait_for(lambda: session.status()['latched'])
        assert session.status()['would_command']['linear'] == 0
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_depth_diagnostics_count_coverage_and_near_returns_independently():
    class SparseDepth(FakeSource):
        def snapshot(self):
            value = FakeSource.snapshot(self)
            value['depth']['min_m'] = 2.0
            value['depth']['valid_grid'] = [[0.4] * 3 for _ in range(3)]
            return value
    session = create(SparseDepth())
    session.start()
    try:
        wait_for(lambda: session.status()['model_status'] == 'accepted')
        wait_for(lambda: session.status()['reason'] == 'depth_unavailable_or_close')
        status = session.status()
        counts = status['depth_diagnostic_counts']
        assert counts['samples'] > 0
        assert counts['coverage_insufficient'] == counts['samples']
        assert counts['minimum_at_or_below_0_5m'] == 0
        assert counts['minimum_unavailable'] == 0
        assert counts['timestamp_stale_or_invalid'] == 0
        assert status['would_command']['linear'] == 0
    finally:
        session.stop(timeout=0.8)


def test_depth_diagnostics_count_stale_near_and_malformed_grid_together():
    class BadDepth(FakeSource):
        def snapshot(self):
            value = FakeSource.snapshot(self)
            value['depth']['stamp'] -= 1.0
            value['depth']['min_m'] = 0.4
            value['depth']['valid_grid'] = [[float('nan')] * 3 for _ in range(3)]
            return value
    session = create(BadDepth())
    session.start()
    try:
        wait_for(lambda: session.status()['depth_diagnostic_counts']['samples'] >= 2)
        counts = session.status()['depth_diagnostic_counts']
        assert counts['timestamp_stale_or_invalid'] == counts['samples']
        assert counts['minimum_at_or_below_0_5m'] == counts['samples']
        assert counts['coverage_unavailable'] == counts['samples']
        assert counts['coverage_insufficient'] == 0
        counts['samples'] = -1
        assert session.status()['depth_diagnostic_counts']['samples'] >= 2
        assert session.status()['would_command']['linear'] == 0
    finally:
        session.stop(timeout=0.8)


def test_stale_after_accepted_zero_decision_waits_for_new_model_without_latching():
    source = OccasionallyStaleSource()
    calls = []
    def stop_decision(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        return {'ok': True, 'choice': 'stop', 'captured_at': calls[-1]}
    session = create(source, model=stop_decision, decision_interval_sec=0.05)
    session.start()
    try:
        wait_for(lambda: session.status()['model_status'] == 'accepted')
        assert session.status()['would_command']['linear'] == 0
        with source.lock:
            source.stale = True
        wait_for(lambda: session.status()['reason'] == 'waiting_for_fresh_observation')
        time.sleep(0.3)
        assert not session.status()['latched']
        assert session.status()['would_command']['linear'] == 0
        with source.lock:
            source.stale = False
        wait_for(lambda: len(calls) >= 2 and session.status()['model_status'] == 'accepted')
        assert calls[-1] > calls[0]
        assert not session.status()['latched']
    finally:
        session.stop(timeout=0.8)


def test_prior_fault_discards_pending_job_without_later_false_timeout():
    source = FakeSource()
    entered, release = threading.Event(), threading.Event()
    def delayed(snapshot, cancelled):
        entered.set()
        release.wait(2.0)
        return forward(snapshot, cancelled)
    session = create(source, model=delayed, decision_timeout_sec=0.4)
    session.start()
    try:
        assert entered.wait(1.0)
        with source.lock:
            source.error = True
        wait_for(lambda: session.status()['latched'])
        assert session.status()['reason'] == 'observation_error'
        time.sleep(0.5)
        state = session.status()
        assert state['model_status'] != 'timeout'
        assert state['model_diagnostic']['discarded_on_fault'] == 1
        assert state['model_diagnostic']['session_timeout'] == 0
    finally:
        release.set()
        session.stop(timeout=0.8)


def test_callback_none_is_not_reported_as_proven_subprocess_timeout():
    session = create(FakeSource(), model=lambda snapshot, cancelled: None)
    session.start()
    try:
        wait_for(lambda: session.status()['latched'])
        status = session.status()
        assert status['reason'] == 'decision_error'
        assert status['model_diagnostic']['callback_returned_none'] == 1
        assert status['model_diagnostic']['session_timeout'] == 0
    finally:
        session.stop(timeout=0.8)


def test_explicit_subprocess_timeout_is_distinct_from_session_timeout_and_discard():
    def worker_timeout(snapshot, cancelled):
        return {'ok': False, 'failure': 'subprocess_timeout'}
    session = create(FakeSource(), model=worker_timeout)
    session.start()
    try:
        wait_for(lambda: session.status()['latched'])
        status = session.status()
        assert status['reason'] == 'decision_timeout'
        assert status['model_status'] == 'worker_timeout'
        assert status['model_diagnostic']['worker_reported_timeout'] == 1
        assert status['model_diagnostic']['session_timeout'] == 0
        assert status['model_diagnostic']['discarded_stale'] == 0
    finally:
        session.stop(timeout=0.8)


def test_first_candidate_expiring_during_policy_work_is_not_counted_as_emitted():
    entered_model, release_model = threading.Event(), threading.Event()
    entered_step, release_step, followup = threading.Event(), threading.Event(), threading.Event()
    reports, calls = [], []
    def model(snapshot, cancelled):
        calls.append(capture_time(snapshot))
        if len(calls) == 1:
            entered_model.set()
            release_model.wait(2.0)
        else:
            followup.wait(2.0)
        return forward(snapshot, cancelled)
    session = create(FakeSource(), model=model, report=reports.append, decision_interval_sec=0.05)
    session.start()
    try:
        assert entered_model.wait(1.0)
        original = session.control.step
        def delay_first_candidate(sample, decision, pose, now, epoch):
            if decision is not None:
                entered_step.set()
                release_step.wait(1.0)
            return original(sample, decision, pose, now, epoch)
        session.control.step = delay_first_candidate
        release_model.set()
        assert entered_step.wait(1.0)
        time.sleep(0.12)
        release_step.set()
        wait_for(lambda: session.status()['reason'] == 'waiting_for_fresh_observation')
        state = session.status()
        assert not state['latched'] and not session.ever_would_move
        assert state['would_command']['linear'] == 0
        assert all(item['would_command']['linear'] == 0 for item in reports)
    finally:
        release_model.set()
        release_step.set()
        followup.set()
        session.stop(timeout=0.8)


def test_missing_startup_samples_do_not_claim_a_model_result_was_discarded():
    source = FakeSource()
    source.snapshot_missing = True
    session = create(source)
    session.start()
    try:
        time.sleep(0.4)
        state = session.status()
        assert state['would_command']['linear'] == 0 and not state['latched']
        assert state['model_diagnostic']['callbacks_started'] == 0
        assert state['model_status'] == 'waiting'
    finally:
        session.stop(timeout=0.8)


def test_invalidated_inflight_job_is_counted_before_any_result_arrives():
    source = FakeSource()
    source.stationary_trace = None
    entered, release = threading.Event(), threading.Event()
    def blocked(snapshot, cancelled):
        entered.set()
        release.wait(2.0)
        return forward(snapshot, cancelled)
    session = create(source, model=blocked, decision_timeout_sec=2.0)
    session.start()
    try:
        assert entered.wait(1.0)
        with source.lock:
            source.snapshot_missing = True
        wait_for(lambda: session.status()['model_diagnostic']['jobs_invalidated_stale'] == 1)
        state = session.status()
        assert state['model_diagnostic']['callbacks_returned'] == 0
        assert state['model_diagnostic']['session_timeout'] == 0
        assert state['would_command']['linear'] == 0
    finally:
        release.set()
        session.stop(timeout=0.8)
