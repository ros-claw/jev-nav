"""Dependency-injected tests only; no ROS master, real subprocess, or device."""
import copy
import json
import os
import threading

import pytest
from test_forward_trial import Clock, Source


def module():
    try:
        import live_trial_adapter
        return live_trial_adapter
    except ImportError:
        pytest.fail("live adapter candidate is not implemented", pytrace=False)


class Obj(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Observation(object):
    def __init__(self, clock):
        self.clock = clock
        self.source = Source(clock)
        self.current = True
        self.mutate = None

    def snapshot(self):
        sample = self.source.snapshot()
        raw = sample.pop("raw_scan")
        sample["scan"] = Obj(header=Obj(frame_id=raw["frame_id"]), **{
            key: value for key, value in raw.items() if key not in ("stamp", "frame_id")})
        sample["color"] = b"source-color"
        sample.pop("image_jpeg")
        if self.mutate:
            self.mutate(sample)
        return sample

    def stationary_trace(self, start, end):
        return self.source.stationary_trace(start, end)

    def is_current_epoch(self, epoch):
        return self.source.is_current_epoch(epoch)

    def is_current_sample(self, sample):
        return self.current and self.source.is_current_sample(sample)


def source(clock):
    obs = Observation(clock)
    return module().ObservationAdapter(obs, lambda color: b"\xff\xd8offline\xff\xd9"), obs


def test_observation_adapter_owns_jpeg_and_scan_without_fabricating_fields():
    clock = Clock()
    adapter, obs = source(clock)
    sample = adapter.snapshot()
    assert adapter.simulation_only is False
    assert sample["image_jpeg"] == b"\xff\xd8offline\xff\xd9"
    assert sample["raw_scan"]["stamp"] == sample["scan_at"]
    assert sample["raw_scan"]["frame_id"] == "laser_link"
    assert sample["raw_scan"]["ranges"][16] == 2.0
    assert sample["depth_pose"] == (0.0, 0.0, 0.0)
    assert sample["state"]["vehicle_state"] == 0 and sample["state"]["control_mode"] == 1
    assert adapter.is_current_sample(sample)
    obs.current = False
    assert not adapter.is_current_sample(sample)
    assert adapter.snapshot() is None
    assert adapter.last_issue == 'current_after_copy'


def test_deferred_adapter_check_still_cannot_authorize_motion_with_stale_source():
    mod = module()
    clock = Clock()
    observation = Observation(clock)
    observation.current = False
    adapter = mod.ObservationAdapter(observation, lambda color: b'\xff\xd8offline\xff\xd9',
                                     recheck_snapshot=False)
    assert isinstance(adapter.snapshot(), dict)
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result['motion_requested'] is False
    assert all(json.loads(line)['linear'] == 0 for line in relay.transport.lines)


def test_serialized_processing_waits_for_snapshot_before_current_validation():
    import time
    mod = module()
    obs = Observation(Clock())
    entered, release, checked = threading.Event(), threading.Event(), threading.Event()
    original_snapshot = obs.snapshot
    def slow_snapshot():
        entered.set()
        assert release.wait(1.)
        return original_snapshot()
    def current(_sample):
        checked.set()
        return 123.0
    obs.snapshot = slow_snapshot
    obs.current_valid_until = current
    adapter = mod.ObservationAdapter(obs, lambda _: b'jpeg', serialize_processing=True)
    worker = threading.Thread(target=adapter.snapshot)
    worker.start()
    try:
        assert entered.wait(.5)
        result = []
        verifier = threading.Thread(target=lambda: result.append(adapter.current_valid_until({})))
        verifier.start()
        time.sleep(.02)
        assert not checked.is_set()
        release.set()
        verifier.join(.5)
        assert not verifier.is_alive()
        assert result == [123.0] and checked.is_set()
    finally:
        release.set()
        worker.join(.5)


def test_serialized_processing_timeout_cannot_return_a_validity_proof():
    from relay import _monotonic
    mod = module()
    obs = Observation(Clock())
    entered, release = threading.Event(), threading.Event()
    def blocked():
        entered.set()
        release.wait(1.)
        return None
    obs.snapshot = blocked
    obs.current_valid_until = lambda _sample: 123.0
    adapter = mod.ObservationAdapter(obs, lambda _: b'jpeg', serialize_processing=True)
    worker = threading.Thread(target=adapter.snapshot)
    worker.start()
    try:
        assert entered.wait(.5)
        started = _monotonic()
        assert adapter.current_valid_until({}) is None
        assert _monotonic() - started < .3
        assert adapter.current_issue() == 'processing_busy'
    finally:
        release.set()
        worker.join(.5)


def test_serialized_processing_rejects_invalid_or_regressing_clock(monkeypatch):
    mod = module()
    adapter = mod.ObservationAdapter(Observation(Clock()), lambda _: b'jpeg',
                                     serialize_processing=True)
    class Busy(object):
        def acquire(self, _blocking):
            return False
    adapter._processing_lock = Busy()
    for times in ([float('nan')], [10., 9.]):
        values = iter(times)
        monkeypatch.setattr(mod, '_monotonic', lambda: next(values))
        assert adapter.current_valid_until({}) is None
        assert adapter.current_issue() == 'processing_busy'


def test_failed_serialized_snapshot_blocks_further_current_proofs():
    mod = module()
    obs = Observation(Clock())
    obs.snapshot_last_issue = 'depth_transform_changed'
    obs.snapshot = lambda: None
    obs.current_valid_until = lambda _sample: 123.0
    adapter = mod.ObservationAdapter(obs, lambda _: b'jpeg', serialize_processing=True)
    assert adapter.snapshot() is None
    assert adapter.current_valid_until({}) is None
    assert adapter.current_issue() == 'source_depth_transform_changed'


def test_hard_snapshot_failure_remains_latched_after_a_later_good_snapshot():
    mod = module()
    obs = Observation(Clock())
    original = obs.snapshot
    obs.snapshot_last_issue = 'depth_transform_changed'
    obs.snapshot = lambda: None
    obs.current_valid_until = lambda _sample: 123.0
    adapter = mod.ObservationAdapter(obs, lambda _: b'jpeg', serialize_processing=True)
    assert adapter.snapshot() is None
    obs.snapshot = original
    assert isinstance(adapter.snapshot(), dict)
    assert adapter.current_valid_until({}) is None


def test_concurrent_busy_snapshot_cannot_hide_a_hard_snapshot_failure():
    mod = module()
    obs = Observation(Clock())
    original_observation = obs.snapshot
    obs.snapshot_last_issue = 'depth_transform_changed'
    obs.snapshot = lambda: None
    obs.current_valid_until = lambda _sample: 123.0
    adapter = mod.ObservationAdapter(obs, lambda _: b'jpeg', serialize_processing=True)
    original = adapter._snapshot_unlocked
    entered, release = threading.Event(), threading.Event()
    def paused():
        result = original()
        entered.set()
        assert release.wait(1.)
        return result
    adapter._snapshot_unlocked = paused
    first = threading.Thread(target=adapter.snapshot)
    first.start()
    try:
        assert entered.wait(.5)
        second = threading.Thread(target=adapter.snapshot)
        second.start()
        second.join(.5)
        assert not second.is_alive()
        release.set()
        first.join(.5)
        assert not first.is_alive()
    finally:
        release.set()
        first.join(.5)
    adapter._snapshot_unlocked = original
    obs.snapshot = original_observation
    assert isinstance(adapter.snapshot(), dict)
    assert adapter.current_valid_until({}) is None


def test_transient_snapshot_gap_blocks_until_a_fresh_snapshot_succeeds():
    mod = module()
    obs = Observation(Clock())
    original = obs.snapshot
    obs.snapshot_last_issue = 'sensor_alignment'
    obs.snapshot = lambda: None
    obs.current_valid_until = lambda _sample: 123.0
    adapter = mod.ObservationAdapter(obs, lambda _: b'jpeg', serialize_processing=True)
    assert adapter.snapshot() is None
    assert adapter.current_valid_until({}) is None
    obs.snapshot = original
    assert isinstance(adapter.snapshot(), dict)
    assert adapter.current_valid_until({}) == 123.0


def test_incomplete_coverage_requires_validated_recovery_not_just_a_snapshot():
    mod = module()
    clock = Clock()
    obs = Observation(clock)
    obs.current_valid_until = lambda _sample: 123.0
    adapter = mod.ObservationAdapter(obs, lambda _: b'jpeg', serialize_processing=True)
    assert adapter.validated_snapshot(lambda *_: (mod._INITIAL_COVERAGE_PENDING, None), clock.now)[0] is None
    assert isinstance(adapter.snapshot(), dict)
    assert adapter.current_valid_until({}) is None
    assert isinstance(adapter.validated_snapshot(lambda *_: (None, {}), clock.now)[0], dict)
    assert adapter.current_valid_until({}) == 123.0


def test_depth_gate_rejection_cannot_collide_with_internal_coverage_pending():
    mod = module()
    clock = Clock()
    obs = Observation(clock)
    obs.current_valid_until = lambda _sample: 123.0
    adapter = mod.ObservationAdapter(obs, lambda _: b'jpeg', serialize_processing=True)
    rejected = {'candidate': False, 'reason': 'initial_coverage_pending'}
    sample, reason, assessment = adapter.validated_snapshot(
        lambda *_: ('initial_coverage_pending', rejected), clock.now)
    assert isinstance(sample, dict) and reason == 'initial_coverage_pending' and assessment == rejected
    assert adapter.current_valid_until({}) is None
    adapter.validated_snapshot(lambda *_: (None, {}), clock.now)
    assert adapter.current_valid_until({}) is None


def test_background_assessment_failure_cannot_race_with_a_new_proof():
    import time
    from async_safety_sampler import AsyncSafetySampler
    mod = module()
    clock = Clock()
    obs = Observation(clock)
    obs.current_valid_until = lambda _sample: 123.0
    entered, release = threading.Event(), threading.Event()
    def validator(_sample, _now):
        entered.set()
        assert release.wait(1.)
        return 'front_blocked', None
    adapter = mod.ObservationAdapter(obs, lambda _: b'jpeg', serialize_processing=True)
    sampler = AsyncSafetySampler(adapter, validator, clock)
    sampler.start()
    try:
        assert entered.wait(.5)
        assert adapter.current_valid_until({}) is None
        release.set()
        for _ in range(100):
            if sampler.poll()['state'] == 'fault':
                break
            time.sleep(.005)
        assert sampler.poll()['state'] == 'fault'
        assert adapter.current_valid_until({}) is None
    finally:
        release.set()
        assert sampler.close(.5)


def test_command_guard_prevents_snapshot_failure_between_proof_and_submission():
    mod = module()
    obs = Observation(Clock())
    entered = threading.Event()
    def failed_snapshot():
        entered.set()
        return None
    obs.snapshot = failed_snapshot
    obs.snapshot_last_issue = 'depth_transform_changed'
    obs.current_valid_until = lambda _sample: 123.0
    adapter = mod.ObservationAdapter(obs, lambda _: b'jpeg', serialize_processing=True)
    assert adapter.begin_command_check()
    worker = threading.Thread(target=adapter.snapshot)
    try:
        assert adapter.current_valid_until({}) == 123.0
        worker.start()
        assert not entered.wait(.02)
        # The caller can submit only while holding this same guard.
    finally:
        adapter.end_command_check()
        worker.join(.5)
    assert entered.is_set()
    assert adapter.current_valid_until({}) is None


def test_motion_submission_occurs_inside_processing_guard():
    mod = module()
    clock = Clock()
    obs = Observation(clock)
    source = mod.ObservationAdapter(obs, lambda _: b'\xff\xd8offline\xff\xd9',
                                    serialize_processing=True)
    guarded = []
    begin = source.begin_command_check
    end = source.end_command_check
    active = [False]
    def begin_guard():
        ok = begin()
        active[0] = ok
        return ok
    def end_guard():
        active[0] = False
        end()
    source.begin_command_check, source.end_command_check = begin_guard, end_guard
    class GuardedRelay(mod.RelayAdapter):
        def send_frame(self, *args, **kwargs):
            guarded.append(active[0])
            return mod.RelayAdapter.send_frame(self, *args, **kwargs)
        def wait_for_frame_write(self, _sequence):
            guarded.append(active[0])
            return True
    relay = GuardedRelay(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(source, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result['motion_requested'] and guarded and all(guarded)
    assert active == [False]


def test_observation_adapter_owns_mutable_scan_and_scene_evidence():
    adapter, observation = source(Clock())
    raw_snapshot = observation.snapshot
    captured = []

    def track():
        item = raw_snapshot()
        captured.append(item)
        return item

    observation.snapshot = track
    result = adapter.snapshot()
    original = captured[0]
    ray_before = result['scene']['rays'][0]['stamp']
    hit_before = result['scene']['occupied_hits'][0]['stamp']
    coverage_before = result['scene']['coverage']['front_samples']
    range_before = result['raw_scan']['ranges'][0]
    original['scene']['rays'][0]['stamp'] = -1
    original['scene']['occupied_hits'][0]['stamp'] = -1
    original['scene']['coverage']['front_samples'] = 0
    original['scan'].ranges[0] = -1
    assert result['scene']['rays'][0]['stamp'] == ray_before
    assert result['scene']['occupied_hits'][0]['stamp'] == hit_before
    assert result['scene']['coverage']['front_samples'] == coverage_before
    assert result['raw_scan']['ranges'][0] == range_before


def test_safety_snapshots_skip_jpeg_after_model_evidence_is_captured():
    mod = module()
    clock = Clock()
    observation = Observation(clock)
    calls = []
    adapter = mod.ObservationAdapter(observation, lambda color: calls.append(color) or b'\xff\xd8offline\xff\xd9')
    model_sample = adapter.snapshot()
    assert model_sample['image_jpeg'] and len(calls) == 1
    adapter.enable_safety_snapshots()
    safety_sample = adapter.snapshot()
    assert 'image_jpeg' not in safety_sample and len(calls) == 1
    assert mod._snapshot_without_depth_gate(safety_sample, clock.now(), require_image=False) is None
    assert mod._snapshot_without_depth_gate(safety_sample, clock.now()) == 'sensor_unavailable'
    adapter.restore_model_snapshots()
    assert adapter.snapshot()['image_jpeg'] and len(calls) == 2


def test_projection_only_mode_requires_explicit_source_support():
    mod = module()
    observation = Observation(Clock())
    enabled = []
    observation.enable_projection_only = lambda: enabled.append(True)
    observation.disable_projection_only = lambda: enabled.append(False)
    adapter = mod.ObservationAdapter(observation, lambda color: b'\xff\xd8offline\xff\xd9',
                                     projection_only_after_model=True)
    adapter.enable_safety_snapshots()
    assert enabled == [True]
    adapter.restore_model_snapshots()
    assert enabled == [True, False]
    unsupported = mod.ObservationAdapter(Observation(Clock()), lambda color: b'',
                                          projection_only_after_model=True)
    with pytest.raises(ValueError):
        unsupported.enable_safety_snapshots()


def test_adapter_can_take_ownership_of_a_newly_allocated_scene():
    mod = module()
    clock = Clock()
    observation = Observation(clock)
    original = observation.snapshot
    snapshots = []
    def capture():
        sample = original()
        snapshots.append(sample)
        return sample
    observation.snapshot = capture
    adapter = mod.ObservationAdapter(observation, lambda color: b'\xff\xd8offline\xff\xd9',
                                     take_scene_ownership=True)
    result = adapter.snapshot()
    assert result['scene'] is snapshots[0]['scene']
    assert result['raw_scan']['ranges'] is not snapshots[0]['scan'].ranges


def test_live_timestamps_allow_processing_delay_but_reject_stale_sensors():
    mod = module()
    sample = source(Clock())[0].snapshot()
    sample['snapshot_at'] = sample['odom_at'] = 99.85
    assert mod._live_timestamps_fresh(sample, 100.0)
    sample['snapshot_at'] = 99.74
    assert not mod._live_timestamps_fresh(sample, 100.0)
    sample['snapshot_at'] = 99.77
    assert mod._live_timestamps_fresh(sample, 100.0)
    sample['snapshot_at'] = 99.85
    sample['scan_at'] = 99.49
    assert not mod._live_timestamps_fresh(sample, 100.0)


def test_old_snapshot_odometry_can_only_be_reused_with_a_fresh_live_recheck():
    mod = module()
    sample = source(Clock())[0].snapshot()
    sample['snapshot_at'] = 99.9
    sample['odom_at'] = 99.7
    assert not mod._live_timestamps_fresh(sample, 100.0)
    assert mod._live_timestamps_fresh(sample, 100.0, latest_odom_rechecked=True)


def test_accepted_snapshot_cannot_outlive_its_historical_coverage():
    from test_policy import fusion_scene
    mod = module()
    sample = source(Clock())[0].snapshot()
    sample['scan_at'] = sample['raw_scan']['stamp'] = sample['scene']['coverage']['scan_stamp'] = 99.8
    old = [2.0 if i % 4 == 0 else 0.0 for i in range(33)]
    latest = [2.0 if i % 4 == 2 else 0.0 for i in range(33)]
    sample['scene'] = fusion_scene(99.41, 99.8, old, latest)
    sample['scene']['coverage']['scan_pose'] = sample['pose']
    sample['raw_scan']['ranges'] = latest
    reason, assessment, _ = mod._assess_sample(sample, 100.0, mod.StrictDepthGate(), True)
    assert reason is None
    assert assessment['valid_until'] == pytest.approx(100.06)
    reason, _, _ = mod._assess_sample(sample, 100.07, mod.StrictDepthGate(), True)
    assert reason == 'sensor_unavailable'


def test_redundant_historical_rays_do_not_shorten_current_coverage():
    mod = module()
    sample = source(Clock())[0].snapshot()
    sample['scan_at'] = sample['raw_scan']['stamp'] = sample['scene']['coverage']['scan_stamp'] = 99.8
    for item in sample['scene']['rays'] + sample['scene']['occupied_hits']:
        item['stamp'] = 99.8
    old_ray = copy.deepcopy(sample['scene']['rays'][0])
    old_ray['stamp'] = 99.41
    sample['scene']['rays'].append(old_ray)
    sample['scene']['occupied_hits'].append({'point_base': old_ray['hit_base'], 'stamp': old_ray['stamp']})
    reason, assessment, _ = mod._assess_sample(sample, 100.0, mod.StrictDepthGate(), True)
    assert reason is None
    assert assessment['valid_until'] == pytest.approx(100.3)


def test_only_identified_snapshot_timing_gaps_are_retryable():
    mod = module()
    class FailedSource(object):
        last_issue = 'source_refreshed_odom'
        detail = None
        def current_issue(self):
            return self.detail
    failing = FailedSource()
    assert mod._transient_snapshot_gap(failing)
    failing.last_issue = 'current_after_copy'
    # Another thread can update the observer's diagnostic; never use that
    # shared value to soften a failed current-sample check.
    for problem in ('stale_snapshot', 'new_scan', 'new_depth_unchecked', 'epoch_or_guard', 'base_state'):
        failing.detail = problem
        assert not mod._transient_snapshot_gap(failing)
    failing.last_issue = 'source_guard_fault'
    assert not mod._transient_snapshot_gap(failing)


def test_previous_scan_can_be_checked_at_capture_time_when_latest_scan_is_rechecked():
    mod = module()
    sample = source(Clock())[0].snapshot()
    sample['snapshot_at'] = sample['odom_at'] = sample['color_at'] = sample['depth_at'] = 99.85
    sample['status_at'] = sample['state']['stamp'] = 99.8
    sample['scan_at'] = sample['raw_scan']['stamp'] = sample['scene']['coverage']['scan_stamp'] = 99.65
    for item in sample['scene']['rays'] + sample['scene']['occupied_hits']:
        item['stamp'] = 99.65
    sample['depth']['stamp'] = 99.85
    assert mod._snapshot_without_depth_gate(sample, 100.0) is None


def test_refreshed_odom_does_not_make_aligned_sensor_frames_appear_skewed():
    mod = module()
    sample = source(Clock())[0].snapshot()
    sample['snapshot_at'] = 100.05
    sample['odom_at'] = 100.1
    sample['status_at'] = sample['state']['stamp'] = 100.05
    sample['scan_at'] = sample['raw_scan']['stamp'] = sample['scene']['coverage']['scan_stamp'] = 99.82
    sample['color_at'] = sample['depth_at'] = sample['depth']['stamp'] = 99.82
    for item in sample['scene']['rays'] + sample['scene']['occupied_hits']:
        item['stamp'] = 99.82
    assert mod._snapshot_without_depth_gate(sample, 100.1) is None
    sample['color_at'] = 100.09
    assert mod._snapshot_without_depth_gate(sample, 100.1) == 'sensor_unavailable'


@pytest.mark.parametrize("field", ["depth_pose", "vehicle_state", "control_mode"])
def test_missing_observation_fields_cannot_be_filled_with_defaults(field):
    clock = Clock()
    adapter, obs = source(clock)
    def remove(sample):
        if field == "depth_pose":
            del sample[field]
        else:
            del sample["state"][field]
    obs.mutate = remove
    assert adapter.snapshot() is None


def test_decisions_adapter_binds_response_to_image_scan_and_epoch():
    mod = module()
    clock = Clock()
    sample = source(clock)[0].snapshot()
    bound = mod.make_evidence(sample)
    requests = []
    def invoke(payload, timeout):
        requests.append(copy.deepcopy(payload))
        clock.t += 2.0
        return {"ok": True, "choice": "forward", "captured_at": payload["captured_at"]}
    decision = mod.DecisionsAdapter(invoke).decide(bound)
    assert decision == {"ok": True, "choice": "forward", "observation_id": bound["observation_id"]}
    assert requests[0]["image_b64"] and requests[0]["laser"]["forward_view"]
    altered = copy.deepcopy(sample)
    altered["epoch"] += 1
    assert mod.make_evidence(altered)["observation_id"] != bound["observation_id"]


def test_model_exception_exposes_only_class_not_sensitive_message():
    mod = module()
    bound = mod.make_evidence(source(Clock())[0].snapshot())
    def fail(_payload, _timeout):
        raise RuntimeError('Authorization secret details')
    decision = mod.DecisionsAdapter(fail).decide(bound)
    assert decision['ok'] is False and decision['error_class'] == 'RuntimeError'
    assert 'secret' not in str(decision) and 'Authorization' not in str(decision)


def test_model_invalid_diagnostic_never_copies_untrusted_error_text():
    mod = module()
    class BadModel(object):
        def decide(self, _evidence):
            return {'ok': False, 'error_class': 'Authorization secret details'}
    clock = Clock()
    relay = mod.RelayAdapter(Transport(), clock)
    result = mod.LiveTrial(source(clock)[0], BadModel(), relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result['reason'] == 'model_invalid'
    assert 'secret' not in str(result) and 'Authorization' not in str(result)


def test_decisions_adapter_rejects_mismatched_or_failed_response():
    mod = module()
    bound = mod.make_evidence(source(Clock())[0].snapshot())
    for response in ({"ok": True, "choice": "forward", "captured_at": -1},
                     {"ok": False, "choice": "forward", "captured_at": bound["color_at"]}):
        assert mod.DecisionsAdapter(lambda payload, timeout, value=response: value).decide(bound)["ok"] is False


class Transport(object):
    def __init__(self, ready=b"READY motion_limit_sec=0.300000\n"):
        self.ready = ready
        self.started = []
        self.lines = []
        self.closed = False

    def start(self, limit):
        self.started.append(limit)

    def read_line(self, timeout):
        result, self.ready = self.ready, None
        return result

    def write_line(self, payload, timeout):
        self.lines.append(payload)
        return True

    def alive(self):
        return not self.closed

    def close(self):
        self.closed = True


def test_relay_adapter_uses_one_writer_and_requests_zero_on_shutdown():
    mod = module()
    clock = Clock()
    transport = Transport()
    relay = mod.RelayAdapter(transport, clock)
    assert relay.simulation_only is False and transport.started == []
    contract = relay.arm(0.3)
    assert contract["ready_line"] == b"READY motion_limit_sec=0.300000\n"
    assert relay.send_frame(0, 100.0, 0.02, 0.0) is True
    clock.t += 0.05
    assert relay.stop_and_latch() is True
    relay.close()
    frames = [json.loads(line) for line in transport.lines]
    assert [frame["linear"] for frame in frames] == [0.02, 0.0]
    assert all(frame["angular"] == 0.0 for frame in frames)
    assert relay.evidence["physical_stop_verified"] is False


def test_longer_forward_relay_requires_explicit_matching_deadline():
    mod = module()
    clock = Clock()
    ordinary = mod.RelayAdapter(Transport(), clock)
    assert ordinary.arm(2.0) is None
    transport = Transport(ready=b'READY motion_limit_sec=2.000000\n')
    relay = mod.RelayAdapter(transport, clock, motion_limit_sec=2.0)
    assert relay.arm(2.0)['relative_motion_limit'] is True
    assert transport.started == [2.0]
    assert relay.send_frame(0, clock.now(), 0.02, 0.0)
    assert relay.stop_and_latch() and relay.close()


def test_longer_forward_trial_streams_and_still_stops_on_new_obstacle():
    mod = module()
    class SynchronousLongTrial(mod.LongerForwardTrial):
        ASYNC_SAMPLING = False
    for obstacle in (False, True):
        clock = Clock()
        adapter, obs = source(clock)
        if obstacle:
            def mutate(sample, count):
                if count >= 12:
                    sample['raw_scan']['ranges'][16] = 0.195
                return sample
            obs.source.mutate = mutate
        transport = Transport(ready=b'READY motion_limit_sec=2.000000\n')
        relay = mod.RelayAdapter(transport, clock, motion_limit_sec=2.0)
        model = mod.DecisionsAdapter(lambda payload, timeout: {
            'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
        result = SynchronousLongTrial(adapter, model, relay, FakeArbiter(), clock,
                                      duration_sec=2.0, enable_motion=True,
                                      serial_guard=FakeSerialGuard()).run()
        frames = [json.loads(line) for line in transport.lines]
        moving = [frame for frame in frames if frame['linear'] > 0]
        assert len(moving) > 3 and frames[-1]['linear'] == 0.0
        if obstacle:
            assert result['reason'] == 'front_blocked'
        else:
            assert result['reason'] in ('time_limit', 'motion_limit')
            assert moving[-1]['stamp'] - moving[0]['stamp'] > 1.0


def test_longer_trial_streams_while_background_source_produces_samples():
    import time

    from relay import _monotonic
    mod = module()
    class RealClock(object):
        def now(self):
            return _monotonic()
        def sleep(self, seconds):
            time.sleep(seconds)
    clock = RealClock()
    adapter, _ = source(clock)
    transport = Transport(ready=b'READY motion_limit_sec=2.000000\n')
    relay = mod.RelayAdapter(transport, clock, motion_limit_sec=2.0)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LongerForwardTrial(adapter, model, relay, FakeArbiter(), clock,
                                   duration_sec=0.6, enable_motion=True,
                                   serial_guard=FakeSerialGuard()).run()
    frames = [json.loads(line) for line in transport.lines]
    moving = [frame for frame in frames if frame['linear'] > 0]
    assert result['sampler_closed'] is True and frames[-1]['linear'] == 0.0
    assert all(frame['stamp'] < frame['expires_at'] for frame in moving)
    if result['reason'] == 'time_limit':
        assert len(moving) >= 5 and moving[-1]['stamp'] - moving[0]['stamp'] > 0.3
    else:
        assert result['reason'] in ('sensor_unavailable', 'expiry_unavailable')


def test_background_sampler_starts_after_first_relay_pipe_write(monkeypatch):
    import time

    import async_safety_sampler
    from relay import _monotonic
    mod = module()
    class RealClock(object):
        def now(self):
            return _monotonic()
        def sleep(self, seconds):
            time.sleep(seconds)
    events = []
    class RecordingRelay(mod.RelayAdapter):
        def wait_for_frame_write(self, sequence):
            assert json.loads(self.transport.lines[-1])['linear'] > 0
            events.append('written')
            return True
    original = async_safety_sampler.AsyncSafetySampler
    class RecordingSampler(original):
        def start(self):
            events.append('sampler')
            return original.start(self)
    monkeypatch.setattr(async_safety_sampler, 'AsyncSafetySampler', RecordingSampler)
    clock = RealClock()
    relay = RecordingRelay(Transport(ready=b'READY motion_limit_sec=2.000000\n'), clock,
                           motion_limit_sec=2.0)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LongerForwardTrial(source(clock)[0], model, relay, FakeArbiter(), clock,
                                   duration_sec=.6, enable_motion=True,
                                   serial_guard=FakeSerialGuard()).run()
    assert result['motion_requested'] is True
    assert events == ['written', 'sampler']


def test_guarded_trial_prepares_a_validated_sample_before_first_motion(monkeypatch):
    import time
    import async_safety_sampler
    from relay import _monotonic
    mod = module()
    class RealClock(object):
        def now(self):
            return _monotonic()
        def sleep(self, seconds):
            time.sleep(seconds)
    events = []
    class RecordingRelay(mod.RelayAdapter):
        def wait_for_frame_write(self, sequence):
            events.append('written')
            return True
    original = async_safety_sampler.AsyncSafetySampler
    class RecordingSampler(original):
        def start(self):
            events.append('sampler')
            return original.start(self)
        def poll(self):
            result = original.poll(self)
            if result['state'] == 'ready' and 'ready' not in events:
                events.append('ready')
            return result
    monkeypatch.setattr(async_safety_sampler, 'AsyncSafetySampler', RecordingSampler)
    clock = RealClock()
    obs = Observation(clock)
    adapter = mod.ObservationAdapter(obs, lambda _: b'\xff\xd8offline\xff\xd9',
                                     serialize_processing=True)
    relay = RecordingRelay(Transport(ready=b'READY motion_limit_sec=2.000000\n'), clock,
                           motion_limit_sec=2.0)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LongerForwardTrial(adapter, model, relay, FakeArbiter(), clock,
                                   duration_sec=.6, enable_motion=True,
                                   serial_guard=FakeSerialGuard()).run()
    assert result['motion_requested'] is True
    assert events.index('sampler') < events.index('ready') < events.index('written')


def test_incomplete_background_coverage_can_recover_only_before_motion(monkeypatch):
    import time
    from relay import _monotonic
    mod = module()
    class RealClock(object):
        def now(self):
            return _monotonic()
        def sleep(self, seconds):
            time.sleep(seconds)
    clock = RealClock()
    adapter = mod.ObservationAdapter(Observation(clock), lambda _: b'\xff\xd8offline\xff\xd9',
                                     serialize_processing=True)
    relay = mod.RelayAdapter(Transport(ready=b'READY motion_limit_sec=2.000000\n'), clock,
                             motion_limit_sec=2.0)
    original = mod._assess_sample
    background = []
    def assess(sample, now, gate, require_image):
        if threading.current_thread().name == 'limo-safety-sampler':
            background.append(relay.had_motion)
            if len(background) == 1:
                return 'sensor_unavailable', None, {'check': 'front_forward_coverage_incomplete'}
        return original(sample, now, gate, require_image)
    monkeypatch.setattr(mod, '_assess_sample', assess)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LongerForwardTrial(adapter, model, relay, FakeArbiter(), clock,
                                   duration_sec=1., enable_motion=True,
                                   serial_guard=FakeSerialGuard()).run()
    assert len(background) >= 2 and background[0] is False
    assert result['motion_requested'] is True


def test_incomplete_background_coverage_after_motion_latches_stop(monkeypatch):
    import time
    from relay import _monotonic
    mod = module()
    class RealClock(object):
        def now(self):
            return _monotonic()
        def sleep(self, seconds):
            time.sleep(seconds)
    clock = RealClock()
    adapter = mod.ObservationAdapter(Observation(clock), lambda _: b'\xff\xd8offline\xff\xd9',
                                     serialize_processing=True)
    relay = mod.RelayAdapter(Transport(ready=b'READY motion_limit_sec=2.000000\n'), clock,
                             motion_limit_sec=2.0)
    original = mod._assess_sample
    failed_at = []
    def assess(sample, now, gate, require_image):
        if threading.current_thread().name == 'limo-safety-sampler' and relay.had_motion:
            failed_at.append(clock.now())
            return 'sensor_unavailable', None, {'check': 'front_forward_coverage_incomplete'}
        return original(sample, now, gate, require_image)
    monkeypatch.setattr(mod, '_assess_sample', assess)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LongerForwardTrial(adapter, model, relay, FakeArbiter(), clock,
                                   duration_sec=1., enable_motion=True,
                                   serial_guard=FakeSerialGuard()).run()
    moving = [json.loads(line) for line in relay.transport.lines if json.loads(line)['linear'] > 0]
    assert moving and failed_at and result['stop_requested'] is True
    assert all(frame['stamp'] <= failed_at[0] for frame in moving)


@pytest.mark.parametrize('state', ('waiting', 'fault', 'stale', 'moving'))
def test_guarded_trial_never_moves_on_unready_or_bad_initial_sample(monkeypatch, state):
    import async_safety_sampler
    mod = module()
    clock = Clock()
    obs = Observation(clock)
    adapter = mod.ObservationAdapter(obs, lambda _: b'\xff\xd8offline\xff\xd9',
                                     serialize_processing=True)
    class StubSampler(object):
        def __init__(self, *_args, **_kwargs):
            pass
        def start(self):
            pass
        def close(self, _timeout):
            return True
        def poll(self):
            if state == 'waiting':
                return {'state': 'waiting'}
            if state == 'fault':
                return {'state': 'fault', 'reason': 'sensor_unavailable'}
            sample = adapter.snapshot()
            if state == 'stale':
                sample['snapshot_at'] = clock.now() - .3
            else:
                sample['state']['linear_x_mps'] = .01
            return {'state': 'ready', 'sample': sample,
                    'assessment': {'valid_until': clock.now() + .4}}
    monkeypatch.setattr(async_safety_sampler, 'AsyncSafetySampler', StubSampler)
    relay = mod.RelayAdapter(Transport(ready=b'READY motion_limit_sec=2.000000\n'), clock,
                             motion_limit_sec=2.0)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LongerForwardTrial(adapter, model, relay, FakeArbiter(), clock,
                                   duration_sec=.6, enable_motion=True,
                                   serial_guard=FakeSerialGuard()).run()
    assert result['motion_requested'] is False
    assert all(json.loads(line)['linear'] == 0.0 for line in relay.transport.lines)


def test_guarded_pre_motion_retry_waits_for_sampler_without_foreground_snapshot(monkeypatch):
    import async_safety_sampler
    mod = module()
    clock = Clock()
    obs = Observation(clock)
    adapter = mod.ObservationAdapter(obs, lambda _: b'\xff\xd8offline\xff\xd9',
                                     serialize_processing=True)
    original_snapshot = adapter.snapshot
    active = [False]
    def foreground_snapshot():
        if active[0]:
            raise AssertionError('foreground snapshot competes with the background sampler')
        return original_snapshot()
    adapter.snapshot = foreground_snapshot
    calls = []
    def validity(_sample):
        calls.append(True)
        return clock.now() + (.05 if len(calls) == 1 else .12)
    adapter.current_valid_until = validity
    class StubSampler(object):
        def __init__(self, *_args, **_kwargs):
            self.samples = []
        def start(self):
            self.samples = [original_snapshot() for _ in range(4)]
            active[0] = True
        def poll(self):
            item = self.samples.pop(0) if len(self.samples) > 1 else self.samples[0]
            return {'state': 'ready', 'sample': item,
                    'assessment': {'valid_until': clock.now() + .4}}
        def close(self, _timeout):
            active[0] = False
            return True
    monkeypatch.setattr(async_safety_sampler, 'AsyncSafetySampler', StubSampler)
    relay = mod.RelayAdapter(Transport(ready=b'READY motion_limit_sec=2.000000\n'), clock,
                             motion_limit_sec=2.0)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LongerForwardTrial(adapter, model, relay, FakeArbiter(), clock,
                                   duration_sec=.6, enable_motion=True,
                                   serial_guard=FakeSerialGuard()).run()
    assert len(calls) >= 2 and result['motion_requested'] is True
    moving = [json.loads(line) for line in relay.transport.lines if json.loads(line)['linear'] > 0]
    assert moving and all(frame['expires_at'] <= frame['stamp'] + .120001 for frame in moving)


def test_guarded_pre_motion_retry_does_not_reuse_same_short_sample(monkeypatch):
    import async_safety_sampler
    mod = module()
    clock = Clock()
    adapter = mod.ObservationAdapter(Observation(clock), lambda _: b'\xff\xd8offline\xff\xd9',
                                     serialize_processing=True)
    calls = []
    def validity(_sample):
        calls.append(True)
        return clock.now() + .05
    adapter.current_valid_until = validity
    class StubSampler(object):
        def __init__(self, *_args, **_kwargs):
            self.sample = None
        def start(self):
            self.sample = adapter.snapshot()
        def poll(self):
            return {'state': 'ready', 'sample': self.sample,
                    'assessment': {'valid_until': clock.now() + .4}}
        def close(self, _timeout):
            return True
    monkeypatch.setattr(async_safety_sampler, 'AsyncSafetySampler', StubSampler)
    relay = mod.RelayAdapter(Transport(ready=b'READY motion_limit_sec=2.000000\n'), clock,
                             motion_limit_sec=2.0)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LongerForwardTrial(adapter, model, relay, FakeArbiter(), clock,
                                   duration_sec=.6, enable_motion=True,
                                   serial_guard=FakeSerialGuard()).run()
    assert len(calls) == 1 and result['motion_requested'] is False
    assert result['initial_preparation']['short_window'] == 1
    assert result['initial_preparation']['same_sample'] > 0
    assert result['initial_preparation']['largest_window_sec'] == pytest.approx(.05)
    assert all(json.loads(line)['linear'] == 0.0 for line in relay.transport.lines)


def test_longer_trial_requests_decision_after_relay_is_ready_at_zero():
    mod = module()
    clock = Clock()
    class SynchronousLongTrial(mod.LongerForwardTrial):
        ASYNC_SAMPLING = False
    relay = mod.RelayAdapter(Transport(ready=b'READY motion_limit_sec=2.000000\n'), clock,
                             motion_limit_sec=2.0)
    observed = []
    def decide(payload, timeout):
        observed.append((relay.armed, list(relay.transport.lines)))
        return {'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']}
    result = SynchronousLongTrial(source(clock)[0], mod.DecisionsAdapter(decide), relay,
                                  FakeArbiter(), clock, duration_sec=0.6, enable_motion=True,
                                  serial_guard=FakeSerialGuard()).run()
    assert result['motion_requested'] is True and len(observed) == 1
    assert observed[0][0] and observed[0][1]
    assert all(json.loads(line)['linear'] == 0.0 for line in observed[0][1])


def test_decision_warmup_refreshes_observation_only_while_model_is_pending():
    import threading
    import time

    from relay import _monotonic
    mod = module()
    class RealClock(object):
        def now(self):
            return _monotonic()
        def sleep(self, seconds):
            time.sleep(seconds)
    class SynchronousLongTrial(mod.LongerForwardTrial):
        ASYNC_SAMPLING = False
    clock = RealClock()
    adapter, _ = source(clock)
    adapter.predecision_warmup = True
    in_model, refreshed = threading.Event(), threading.Event()
    original_snapshot = adapter.snapshot
    def snapshot():
        if in_model.is_set():
            refreshed.set()
        return original_snapshot()
    adapter.snapshot = snapshot
    def decide(payload, _timeout):
        in_model.set()
        refreshed.wait(.8)
        in_model.clear()
        return {'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']}
    relay = mod.RelayAdapter(Transport(ready=b'READY motion_limit_sec=2.000000\n'), clock,
                             motion_limit_sec=2.0)
    result = SynchronousLongTrial(adapter, mod.DecisionsAdapter(decide), relay,
                                  FakeArbiter(), clock, duration_sec=.3, enable_motion=True,
                                  serial_guard=FakeSerialGuard()).run()
    assert refreshed.is_set()
    assert result['model_choice'] == 'forward'


def test_busy_decision_warmup_prevents_motion(monkeypatch):
    import async_safety_sampler
    mod = module()
    class BusyWarmup(object):
        def __init__(self, _source):
            pass
        def start(self):
            pass
        def close(self, _timeout):
            return False
    monkeypatch.setattr(async_safety_sampler, 'DecisionWarmup', BusyWarmup)
    clock = Clock()
    adapter, _ = source(clock)
    adapter.predecision_warmup = True
    transport = Transport(ready=b'READY motion_limit_sec=2.000000\n')
    relay = mod.RelayAdapter(transport, clock, motion_limit_sec=2.0)
    model = mod.DecisionsAdapter(lambda payload, _timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LongerForwardTrial(adapter, model, relay, FakeArbiter(), clock,
                                    duration_sec=.3, enable_motion=True,
                                    serial_guard=FakeSerialGuard()).run()
    assert result['reason'] == 'predecision_warmup_busy'
    assert not any(json.loads(line)['linear'] > 0 for line in transport.lines)


def test_longer_trial_stops_when_background_sensing_fails():
    import time

    from relay import _monotonic
    mod = module()
    class RealClock(object):
        def now(self):
            return _monotonic()
        def sleep(self, seconds):
            time.sleep(seconds)
    clock = RealClock()
    adapter, _ = source(clock)
    relay = mod.RelayAdapter(Transport(ready=b'READY motion_limit_sec=2.000000\n'), clock,
                             motion_limit_sec=2.0)
    class FailingSource(object):
        def snapshot(self):
            return None if relay.had_motion else adapter.snapshot()
        def __getattr__(self, name):
            return getattr(adapter, name)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LongerForwardTrial(FailingSource(), model, relay, FakeArbiter(), clock,
                                   duration_sec=0.6, enable_motion=True,
                                   serial_guard=FakeSerialGuard()).run()
    frames = [json.loads(line) for line in relay.transport.lines]
    assert result['reason'] == 'sensor_unavailable' and result['sampler_closed'] is True
    assert any(frame['linear'] > 0 for frame in frames) and frames[-1]['linear'] == 0.0
def test_transport_waits_for_relay_zero_burst_and_ros_unregister_after_term():
    mod = module()
    clock = Clock()
    class Stream(object):
        def close(self):
            pass
    class Process(object):
        def __init__(self):
            self.stdin, self.stdout = Stream(), Stream()
            self.terminated_at, self.killed = None, False
        def poll(self):
            if self.killed or (self.terminated_at is not None and clock.now() - self.terminated_at >= 0.6):
                return 0
            return None
        def terminate(self):
            self.terminated_at = clock.now()
        def kill(self):
            self.killed = True
        def wait(self):
            return self.poll()
    process = Process()
    transport = mod.RelayProcessTransport(clock)
    transport.process = process
    transport.close()
    assert process.terminated_at is not None
    assert not process.killed
    assert transport.exited is True


def test_relay_rejects_wrong_ready_and_multiple_writer_threads():
    mod = module()
    clock = Clock()
    wrong = mod.RelayAdapter(Transport(ready=b"READY\n"), clock)
    assert wrong.arm(0.3) is None
    relay = mod.RelayAdapter(Transport(), clock)
    assert relay.arm(0.3)
    result = []
    t = threading.Thread(target=lambda: result.append(relay.send_frame(0, 100.0, 0.02, 0.0)))
    t.start()
    t.join()
    assert result == [False]
    assert relay.transport.lines == []


def test_same_clock_tick_stop_still_attempts_a_valid_zero():
    mod = module()
    clock = Clock()
    transport = Transport()
    relay = mod.RelayAdapter(transport, clock)
    assert relay.arm(0.3)
    assert relay.send_frame(0, clock.now(), 0.02, 0.0)
    assert relay.stop_and_latch()
    frames = [json.loads(line) for line in transport.lines]
    assert frames[-1]["linear"] == 0 and frames[-1]["stamp"] > frames[0]["stamp"]


def test_relay_records_stop_status_as_process_evidence_only():
    mod = module()
    clock = Clock()
    class StopTransport(Transport):
        def write_line(self, payload, timeout):
            result = super(StopTransport, self).write_line(payload, timeout)
            if json.loads(payload)["linear"] == 0:
                self.ready = b"STOP requested_stop\n"
            return result
    relay = mod.RelayAdapter(StopTransport(), clock)
    assert relay.arm(0.3)
    assert relay.send_frame(0, clock.now(), 0.02, 0.0)
    assert relay.stop_and_latch()
    assert relay.evidence["zero_write_completed"] is True
    assert relay.evidence["stop_line"] == b"STOP requested_stop\n"
    assert relay.evidence["physical_stop_verified"] is False


def test_zero_heartbeat_before_first_motion_preserves_frame_sequence():
    mod = module()
    clock = Clock()
    relay = mod.RelayAdapter(Transport(), clock)
    assert relay.arm(0.3)
    assert relay.send_zero_heartbeat()
    clock.t += 0.05
    assert relay.send_frame(1, clock.now(), 0.02, 0.0)
    frames = [json.loads(line) for line in relay.transport.lines]
    assert [frame["sequence"] for frame in frames] == [0, 1]
    assert [frame["linear"] for frame in frames] == [0.0, 0.02]


def test_multiple_zero_heartbeats_are_allowed_only_before_motion():
    mod = module()
    clock = Clock()
    relay = mod.RelayAdapter(Transport(), clock)
    assert relay.arm(0.3)
    assert relay.send_zero_heartbeat()
    clock.t += 0.05
    assert relay.send_zero_heartbeat()
    clock.t += 0.05
    assert relay.send_frame(2, clock.now(), 0.02, 0.0)
    clock.t += 0.05
    assert not relay.send_zero_heartbeat()
    frames = [json.loads(line) for line in relay.transport.lines]
    assert [frame['sequence'] for frame in frames] == [0, 1, 2]
    assert [frame['linear'] for frame in frames] == [0.0, 0.0, 0.02]


def test_unreaped_relay_does_not_release_exclusive_guards():
    mod = module()
    clock = Clock()
    class StuckTransport(Transport):
        def close(self):
            pass
    relay = mod.RelayAdapter(StuckTransport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {"ok": True, "choice": "forward",
                                                           "captured_at": payload["captured_at"]})
    arbiter, serial = FakeArbiter(), FakeSerialGuard()
    result = mod.LiveTrial(source(clock)[0], model, relay, arbiter, clock,
                           enable_motion=True, serial_guard=serial).run()
    assert result["reason"] == "relay_unreaped"
    assert not arbiter.released and not serial.released
    assert result["physical_stop_verified"] is False


def test_default_dry_run_never_arms_or_sends_and_reports_depth_reason():
    mod = module()
    clock = Clock()
    adapter, obs = source(clock)
    obs.mutate = lambda sample: sample["depth"].update(valid_grid=[[0.312, 0, 0.185],
        [0.469, 0.312, 0.228], [0.168, 0.160, 0.404]])
    relay = mod.RelayAdapter(Transport(), clock)
    trial = mod.LiveTrial(adapter, None, relay, None, clock)
    result = trial.run()
    assert result["reason"] == "depth_grid_incomplete"
    assert relay.transport.started == [] and relay.transport.lines == []
    assert result["physical_stop_verified"] is False


def test_operator_report_is_recorded_as_unverified_site_conditions():
    mod = module()
    clock = Clock()
    report = {"clear_room": True, "supervised": True, "emergency_stop_available": True}
    result = mod.LiveTrial(source(clock)[0], None, mod.RelayAdapter(Transport(), clock), None,
                           clock, operator_site_report=report).run()
    assert result["site_conditions_operator_report"] == report
    assert result["site_conditions_verified"] is False
    report["supervised"] = False
    assert result["site_conditions_operator_report"]["supervised"] is True


class ProjectedDepthFixture(object):
    def __init__(self, candidate=True, mismatched=False):
        self.candidate, self.mismatched = candidate, mismatched

    def assess(self, sample, now):
        return {"candidate": self.candidate, "reason": "depth_obstacle" if not self.candidate else "sampled_depth_corridor",
                "depth_at": sample["depth_at"] - (1.0 if self.mismatched else 0),
                "geometric_clearance_verified": False, "unknown_space": True}


def test_explicit_depth_gate_can_classify_ground_without_claiming_clearance():
    mod = module()
    clock = Clock()
    adapter, obs = source(clock)
    def ground(sample):
        sample["depth"].update(min_m=0.475, valid_grid=[[0.312, 0, 0.185],
                                 [0.469, 0.312, 0.228], [0.168, 0.160, 0.404]])
    obs.mutate = ground
    relay = mod.RelayAdapter(Transport(), clock)
    result = mod.LiveTrial(adapter, None, relay, None, clock, depth_gate=ProjectedDepthFixture()).run()
    assert result["reason"] == "dry_run"
    assert result["depth_assessment"]["unknown_space"] is True
    assert result["depth_assessment"]["geometric_clearance_verified"] is False
    assert relay.transport.started == []


def test_depth_gate_rejection_and_wrong_frame_are_fail_closed():
    mod = module()
    for gate, reason in ((ProjectedDepthFixture(False), "depth_obstacle"),
                         (ProjectedDepthFixture(True, mismatched=True), "depth_gate_invalid")):
        clock = Clock()
        result = mod.LiveTrial(source(clock)[0], None, mod.RelayAdapter(Transport(), clock),
                               None, clock, depth_gate=gate).run()
        assert result["reason"] == reason


def test_untrusted_depth_gate_reason_does_not_enter_result():
    mod = module()
    class BadGate(object):
        def assess(self, sample, _now):
            return {'candidate': False, 'reason': 'AUTH_SECRET_EVIDENCE',
                    'depth_at': sample['depth_at'], 'geometric_clearance_verified': False}
    clock = Clock()
    result = mod.LiveTrial(source(clock)[0], None, mod.RelayAdapter(Transport(), clock),
                           None, clock, depth_gate=BadGate()).run()
    assert result['reason'] == 'depth_gate_invalid' and 'AUTH_SECRET' not in str(result)


def test_untrusted_depth_gate_extra_fields_are_not_returned():
    mod = module()
    class BadGate(object):
        def assess(self, sample, _now):
            return {'candidate': True, 'reason': 'sampled_depth_corridor', 'depth_at': sample['depth_at'],
                    'geometric_clearance_verified': False, 'header': 'AUTH_SECRET_EVIDENCE'}
    sample = source(Clock())[0].snapshot()
    reason, assessment, _ = mod._assess_sample(sample, 100., BadGate(), True)
    assert reason is None and 'AUTH_SECRET' not in str(assessment)


def test_rejected_depth_gate_cannot_smuggle_non_numeric_deadline():
    mod = module()
    class BadGate(object):
        def assess(self, sample, _now):
            return {'candidate': False, 'reason': 'depth_obstacle', 'depth_at': sample['depth_at'],
                    'geometric_clearance_verified': False, 'valid_until': 'AUTH_SECRET_EVIDENCE'}
    sample = source(Clock())[0].snapshot()
    reason, assessment, _ = mod._assess_sample(sample, 100., BadGate(), True)
    assert reason == 'depth_gate_invalid' and 'AUTH_SECRET' not in str(assessment)


def test_motion_opt_in_still_requires_atomic_exclusive_arbitration():
    mod = module()
    clock = Clock()
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {"ok": True, "choice": "forward",
                                                           "captured_at": payload["captured_at"]})
    trial = mod.LiveTrial(source(clock)[0], model, relay, mod.RosGraphGuard(), clock,
                          enable_motion=True)
    result = trial.run()
    assert result["reason"] == "exclusive_arbitration_unavailable"
    assert relay.transport.started == [] and relay.transport.lines == []


class FakeArbiter(object):
    """A fake monitor, not a real ROS or chassis lock."""
    monitoring_enabled = True

    def __init__(self, fail_at=None):
        self.fail_at, self.checks, self.released = fail_at, 0, False

    def acquire(self):
        return True

    def bind_relay(self, relay):
        return True

    def verify(self):
        self.checks += 1
        return self.checks != self.fail_at

    def evidence(self):
        return {"driver_caller_bound": True, "observed_competitors": [], "graph_query_atomic": False}

    def valid_until(self):
        return 1e100

    def release(self):
        self.released = True


class FakeSerialGuard(object):
    """Fixture only: TIOCEXCL alone is not proof of sole serial ownership."""
    def __init__(self, verified=True):
        self.monitoring_enabled = verified
        self.released = False

    def acquire(self):
        return True

    def verify(self):
        return True

    def evidence(self):
        return {"tioc_exclusive_reported": False, "initial_open_errno": "EBUSY",
                "probe_repeated": False, "observed_other_fds": [], "previous_fd_excluded": False}

    def valid_until(self):
        return 1e100

    def release(self):
        self.released = True


def test_zero_heartbeat_is_sent_before_monitor_binding_can_delay():
    mod = module()
    clock, observed = Clock(), []

    class MonitoredRos(FakeArbiter):
        def bind_relay(self, relay):
            observed.append(relay.sequence)
            return False

    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(source(clock)[0], model, relay, MonitoredRos(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert observed == [0]
    assert json.loads(relay.transport.lines[0])['linear'] == 0.0
    assert result['motion_requested'] is False


def test_explicit_guard_evidence_does_not_claim_atomic_exclusivity():
    mod = module()
    clock = Clock()
    class MonitoredRos(FakeArbiter):
        atomic_exclusive = False
        monitoring_enabled = True
        def evidence(self):
            return {"driver_caller_bound": True, "observed_competitors": [],
                    "graph_query_atomic": False}
    class MonitoredSerial(FakeSerialGuard):
        sole_writer_verified = False
        monitoring_enabled = True
        def evidence(self):
            return {"tioc_exclusive_reported": False, "initial_open_errno": "EBUSY",
                    "probe_repeated": False, "observed_other_fds": [], "previous_fd_excluded": False}
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {"ok": True, "choice": "forward",
                                                           "captured_at": payload["captured_at"]})
    result = mod.LiveTrial(source(clock)[0], model, relay, MonitoredRos(), clock,
                           enable_motion=True, serial_guard=MonitoredSerial()).run()
    assert result["reason"] == "motion_limit"
    assert result["atomic_exclusivity_verified"] is False
    assert result["serial_sole_writer_verified"] is False
    assert result["ros_guard_evidence"]["graph_query_atomic"] is False
    assert result["serial_guard_evidence"]["previous_fd_excluded"] is False
    assert result["serial_guard_evidence"]["tioc_exclusive_reported"] is False
    assert result["serial_guard_evidence"]["initial_open_errno"] == "EBUSY"


def test_live_trial_requires_independent_serial_sole_writer_evidence():
    mod = module()
    clock = Clock()
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {"ok": True, "choice": "forward",
                                                           "captured_at": payload["captured_at"]})
    result = mod.LiveTrial(source(clock)[0], model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard(False)).run()
    assert result["reason"] == "serial_exclusivity_unavailable"
    assert not relay.transport.started and not relay.transport.lines


def test_bad_relay_ready_always_closes_the_started_transport():
    mod = module()
    clock = Clock()
    relay = mod.RelayAdapter(Transport(ready=b"READY\n"), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {"ok": True, "choice": "forward",
                                                           "captured_at": payload["captured_at"]})
    result = mod.LiveTrial(source(clock)[0], model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result["reason"] == "relay_contract"
    assert relay.transport.started and relay.transport.closed


def test_relay_process_transport_uses_bounded_fake_pipes_and_exact_command():
    mod = module()
    incoming_r, incoming_w = os.pipe()
    outgoing_r, outgoing_w = os.pipe()
    class Process(object):
        def __init__(self):
            self.stdin = os.fdopen(outgoing_w, "wb", 0)
            self.stdout = os.fdopen(incoming_r, "rb", 0)
            self.done = False
        def poll(self):
            return 0 if self.done else None
        def terminate(self):
            self.done = True
        def kill(self):
            self.done = True
        def wait(self):
            return 0
    process = Process()
    arguments = []
    def fake_popen(command, **kwargs):
        arguments.append(command)
        return process
    transport = mod.RelayProcessTransport(Clock(), popen=fake_popen)
    try:
        transport.start(0.3)
        os.write(incoming_w, b"READY motion_limit_sec=0.300000\n")
        assert transport.read_line(0.1) == b"READY motion_limit_sec=0.300000\n"
        assert transport.write_line(b"synthetic\n", 0.1)
        assert os.read(outgoing_r, 20) == b"synthetic\n"
        assert arguments[0][-2:] == ["--max-motion-sec", "0.300000"]
    finally:
        transport.close()
        os.close(incoming_w)
        os.close(outgoing_r)


@pytest.mark.parametrize("delay", [2.0, 6.0])
def test_fake_live_trial_accepts_delayed_model_only_when_odom_continuous(delay):
    mod = module()
    clock = Clock()
    def invoke(payload, timeout):
        clock.t += delay
        return {"ok": True, "choice": "forward", "captured_at": payload["captured_at"]}
    relay = mod.RelayAdapter(Transport(), clock)
    arbiter = FakeArbiter()
    trial = mod.LiveTrial(source(clock)[0], mod.DecisionsAdapter(invoke), relay, arbiter, clock,
                          enable_motion=True, serial_guard=FakeSerialGuard())
    result = trial.run()
    assert result["reason"] == "motion_limit"
    frames = [json.loads(line) for line in relay.transport.lines]
    moving = [frame for frame in frames if frame["linear"] > 0]
    assert 1 <= len(moving) <= 61 and frames[0]["linear"] == 0.0 and frames[-1]["linear"] == 0.0
    assert all(frame["linear"] <= 0.02 and frame["angular"] == 0 for frame in moving)
    assert moving[-1]["stamp"] - moving[0]["stamp"] < 0.3
    assert arbiter.released and result["physical_stop_verified"] is False


def test_fake_live_trial_rejects_inference_motion_and_new_laser_obstacle():
    mod = module()
    for problem in ("move", "obstacle"):
        clock = Clock()
        adapter, obs = source(clock)
        if problem == "move":
            def trace_mutation(trace):
                trace[len(trace) // 2]["linear_y_mps"] = 0.02
                return trace
            obs.source.trace_mutate = trace_mutation
        else:
            def mutate(sample, count):
                if count >= 2:
                    sample["raw_scan"]["ranges"][16] = 0.195
                return sample
            obs.source.mutate = mutate
        def invoke(payload, timeout, current_clock=clock):
            current_clock.t += 2.0
            return {"ok": True, "choice": "forward", "captured_at": payload["captured_at"]}
        relay = mod.RelayAdapter(Transport(), clock)
        result = mod.LiveTrial(adapter, mod.DecisionsAdapter(invoke), relay, FakeArbiter(), clock,
                               enable_motion=True, serial_guard=FakeSerialGuard()).run()
        assert result["reason"] in ("sensor_unavailable", "front_blocked")
        assert relay.transport.lines == []


def test_model_evidence_expiring_after_initial_zero_cannot_start_motion():
    mod = module()
    clock = Clock()
    def invoke(payload, timeout):
        clock.t += 7.9995
        return {"ok": True, "choice": "forward", "captured_at": payload["captured_at"]}
    relay = mod.RelayAdapter(Transport(), clock)
    result = mod.LiveTrial(source(clock)[0], mod.DecisionsAdapter(invoke), relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result["reason"] == "model_stale"
    assert all(json.loads(line)["linear"] == 0 for line in relay.transport.lines)


def test_guard_loss_after_first_fake_command_requests_zero():
    mod = module()
    clock = Clock()
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {"ok": True, "choice": "forward",
                                                           "captured_at": payload["captured_at"]})
    result = mod.LiveTrial(source(clock)[0], model, relay, FakeArbiter(fail_at=3), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result["reason"] == "exclusive_arbitration_lost"
    frames = [json.loads(line) for line in relay.transport.lines]
    assert frames[0]["linear"] == 0 and any(frame["linear"] > 0 for frame in frames)
    assert frames[-1]["linear"] == 0.0


def test_motion_rechecks_latest_observation_after_guard_and_before_each_command():
    mod = module()
    clock = Clock()
    adapter, observation = source(clock)
    relay = mod.RelayAdapter(Transport(), clock)
    class LateChange(FakeSerialGuard):
        def verify(self):
            if relay.had_motion:
                observation.current = False
            return True
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=LateChange()).run()
    frames = [json.loads(line) for line in relay.transport.lines]
    assert result['reason'] == 'sensor_unavailable'
    assert len([frame for frame in frames if frame['linear'] > 0]) == 1
    assert frames[-1]['linear'] == 0.0


@pytest.mark.parametrize('latest_odom_valid', [False, True])
def test_first_motion_uses_live_odom_recheck_after_old_snapshot_odom_ages(latest_odom_valid):
    mod = module()
    clock = Clock()
    adapter, _ = source(clock)
    adapter.current_valid_until = lambda sample: clock.now() + .1 if latest_odom_valid else None
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    class DelayedOdomTrial(mod.LiveTrial):
        calls = 0
        def _aligned_stationary_sample(self, before):
            sample, now = mod.LiveTrial._aligned_stationary_sample(self, before)
            self.calls += 1
            if self.calls >= 2:
                # The full snapshot is younger than 250 ms, while the odom
                # it carried has aged out; the live check is still required.
                sample['odom_at'] = now - .1
                clock.sleep(.16)
            return sample, self._now()
    result = DelayedOdomTrial(adapter, model, relay, FakeArbiter(), clock,
                              enable_motion=True, serial_guard=FakeSerialGuard()).run()
    moving = [json.loads(line) for line in relay.transport.lines if json.loads(line)['linear'] > 0]
    if latest_odom_valid:
        assert result['motion_requested'] is True and moving
    else:
        assert result['motion_requested'] is False and not moving


def test_motion_loop_prepares_next_sample_during_the_frame_interval():
    mod = module()
    clock = Clock()
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        "ok": True, "choice": "forward", "captured_at": payload["captured_at"]})

    class TimedTrial(mod.LiveTrial):
        def __init__(self, *args, **kwargs):
            mod.LiveTrial.__init__(self, *args, **kwargs)
            self.motion_samples = self.motion_validations = 0

        def _sample(self, defer_current_check=False):
            if relay.had_motion:
                self.motion_samples += 1
                # Sampling takes 40 ms of the 50 ms command period.
                clock.sleep(0.04)
            return mod.LiveTrial._sample(self, defer_current_check=defer_current_check)

        def _validate(self, sample, now):
            if relay.had_motion:
                self.motion_validations += 1
            return mod.LiveTrial._validate(self, sample, now)

    trial = TimedTrial(source(clock)[0], model, relay, FakeArbiter(), clock,
                       enable_motion=True, serial_guard=FakeSerialGuard())
    result = trial.run()
    moving = [json.loads(line) for line in relay.transport.lines if json.loads(line)["linear"] > 0]
    assert result["reason"] == "motion_limit"
    assert len(moving) >= 3
    assert max(b["stamp"] - a["stamp"] for a, b in zip(moving, moving[1:])) <= 0.066
    assert trial.motion_samples == trial.motion_validations
    assert result['motion_timing']['max_sample_sec'] >= 0.039


def test_single_threaded_long_trial_samples_immediately_and_stops_on_obstacle():
    mod = module()
    clock = Clock()
    adapter, obs = source(clock)
    def mutate(sample, count):
        if count >= 12:
            sample['raw_scan']['ranges'][16] = 0.195
        return sample
    obs.source.mutate = mutate
    relay = mod.RelayAdapter(Transport(ready=b'READY motion_limit_sec=2.000000\n'),
                             clock, motion_limit_sec=2.0)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    delays = []
    class TimedTrial(mod.SynchronousLongerForwardTrial):
        def _sample(self, defer_current_check=False):
            if relay.had_motion:
                frames = [json.loads(line) for line in relay.transport.lines]
                delays.append(clock.now() - frames[-1]['stamp'])
            return mod.SynchronousLongerForwardTrial._sample(self, defer_current_check)
    trial = TimedTrial(adapter, model, relay, FakeArbiter(), clock,
                       duration_sec=2.0, enable_motion=True, serial_guard=FakeSerialGuard())
    result = trial.run()
    frames = [json.loads(line) for line in relay.transport.lines]
    assert result['reason'] == 'front_blocked'
    assert len(delays) > 3 and max(delays) < 0.002
    assert frames[-1]['linear'] == 0.0
    assert trial.ASYNC_SAMPLING is False


def test_shorter_refresh_period_does_not_reduce_distance_reserve():
    mod = module()
    clock = Clock()
    relay = mod.RelayAdapter(Transport(ready=b'READY motion_limit_sec=2.000000\n'),
                             clock, motion_limit_sec=2.0)
    trial = mod.SynchronousLongerForwardTrial(source(clock)[0], None, relay, FakeArbiter(), clock,
                                             duration_sec=2.0, period_sec=0.025,
                                             serial_guard=FakeSerialGuard())
    assert trial._distance_limit_reached(0.0392, 0.0392)
    assert trial._distance_limit_reached(0.037, 0.037)
    assert not trial._distance_limit_reached(0.036, 0.036)


def test_motion_intent_is_bounded_by_monitor_expiry():
    mod = module()
    clock = Clock()
    class ExpiringArbiter(FakeArbiter):
        def valid_until(self):
            return clock.now() + 0.03
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(source(clock)[0], model, relay, ExpiringArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    moving = [json.loads(line) for line in relay.transport.lines if json.loads(line)['linear'] > 0]
    assert result['reason'] == 'motion_limit' and moving
    assert all(frame['expires_at'] <= frame['stamp'] + 0.030001 for frame in moving)


def test_motion_refreshes_before_short_sensor_deadline():
    mod = module()
    clock = Clock()
    adapter, _ = source(clock)
    adapter.current_valid_until = lambda sample: clock.now() + 0.035
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    moving = [json.loads(line) for line in relay.transport.lines if json.loads(line)['linear'] > 0]
    assert result['reason'] in ('motion_limit', 'expiry_unavailable') and len(moving) >= 3
    assert max(b['stamp'] - a['stamp'] for a, b in zip(moving, moving[1:])) <= 0.021


def test_motion_uses_monitor_deadline_after_final_sensor_check():
    mod = module()
    clock = Clock()
    adapter, _ = source(clock)
    checks = []

    class Watchdog(FakeArbiter):
        def valid_until(self):
            return clock.now() + (.15 if checks else .015)

    def check(_sample):
        clock.sleep(.005)
        checks.append(clock.now())
        return clock.now() + .10

    adapter.current_valid_until = check
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, Watchdog(), clock, enable_motion=True,
                           serial_guard=FakeSerialGuard()).run()
    first = result['renewal_checks'][0]
    assert first['submitted']
    assert first['sensor_check_started_at'] < first['sensor_checked_at'] <= first['checked_at']
    assert first['limits']['ros_monitor'] >= first['sensor_checked_at'] + .14
    assert first['valid_until'] <= first['limits']['latest_sensors']


def test_motion_never_submits_an_intent_with_too_little_time_to_reach_the_relay():
    mod = module()
    clock = Clock()
    adapter, _ = source(clock)
    adapter.current_valid_until = lambda sample: clock.now() + 0.005
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result['reason'] == 'expiry_unavailable' and result['motion_requested'] is False
    assert all(json.loads(line)['linear'] == 0.0 for line in relay.transport.lines)


@pytest.mark.parametrize('recovery_delay,recovered', [(0.0, True), (0.125, False)])
def test_short_valid_refresh_rechecks_once_without_extending_previous_intent(recovery_delay, recovered):
    mod = module()
    clock = Clock()
    adapter, _ = source(clock)
    relay = mod.RelayAdapter(Transport(), clock)
    checked = []

    def validity(_sample):
        if relay.had_motion:
            checked.append(clock.now())
            if len(checked) == 1:
                return clock.now() + .016
            if len(checked) == 2:
                clock.sleep(recovery_delay)
            else:
                return None
        return clock.now() + .10

    adapter.current_valid_until = validity
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    frames = [json.loads(line) for line in relay.transport.lines]
    moving = [frame for frame in frames if frame['linear'] > 0]
    assert result['motion_timing']['short_window_rechecks'] == 1
    assert len(moving) == (2 if recovered else 1)
    assert frames[-1]['linear'] == 0.0
    if recovered:
        assert moving[1]['stamp'] < moving[0]['expires_at']


def test_unavailable_latest_sensor_preserves_its_failure_reason():
    mod = module()
    clock = Clock()
    adapter, _ = source(clock)
    adapter.current_valid_until = lambda _sample: None
    adapter.current_issue = lambda: 'depth_generation_changed'
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result['reason'] == 'expiry_unavailable' and not result['motion_requested']
    assert result['diagnostic']['detail'] == 'depth_generation_changed'


def test_trial_refreshes_once_before_motion_if_initial_sample_is_nearly_expired():
    mod = module()
    clock = Clock()
    adapter, _ = source(clock)
    calls = []
    def validity(sample):
        calls.append(True)
        return clock.now() + (0.005 if len(calls) == 1 else 0.1)
    adapter.current_valid_until = validity
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert len(calls) > 1 and result['motion_requested'] is True


def test_first_motion_waits_for_enough_time_to_prepare_the_next_observation():
    mod = module()
    clock = Clock()
    adapter = mod.ObservationAdapter(Observation(clock), lambda _: b'\xff\xd8offline\xff\xd9',
                                     serialize_processing=True)
    relay = mod.RelayAdapter(Transport(), clock)
    calls = []
    def validity(_sample):
        calls.append(True)
        if len(calls) <= 2:
            assert not relay.had_motion
            return clock.now() + .05
        return clock.now() + .12
    adapter.current_valid_until = validity
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert len(calls) >= 3 and result['motion_requested'] is True


def test_stationary_retry_can_outlast_two_seconds_without_sending_motion():
    mod = module()
    clock = Clock()
    adapter, _ = source(clock)
    relay = mod.RelayAdapter(Transport(), clock)
    started = clock.now()
    def validity(_sample):
        if clock.now() - started < 2.4:
            assert not relay.had_motion
            clock.sleep(.4)
            return None
        return clock.now() + .1
    adapter.current_valid_until = validity
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result['motion_requested'] is True


def test_trial_refreshes_stale_initial_snapshot_while_still_stopped():
    mod = module()
    clock = Clock()
    class InitiallyStale(mod.LiveTrial):
        calls = 0
        def _aligned_stationary_sample(self, before):
            sample, now = mod.LiveTrial._aligned_stationary_sample(self, before)
            self.calls += 1
            if self.calls == 2:
                sample['snapshot_at'] = now - 0.26
            return sample, now
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    trial = InitiallyStale(source(clock)[0], model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard())
    result = trial.run()
    assert trial.calls >= 3 and result['motion_requested'] is True


@pytest.mark.parametrize('initially_enabled', [True, False])
def test_trial_defers_gc_only_during_armed_window_and_restores_prior_state(monkeypatch, initially_enabled):
    mod = module()
    class Gc(object):
        enabled = initially_enabled
        calls = []
        def isenabled(self):
            return self.enabled
        def collect(self):
            self.calls.append('collect')
        def disable(self):
            self.calls.append('disable')
            self.enabled = False
        def enable(self):
            self.calls.append('enable')
            self.enabled = True
    recorder = Gc()
    monkeypatch.setattr(mod, 'gc', recorder)
    clock = Clock()
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(source(clock)[0], model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result['reason'] == 'motion_limit'
    assert recorder.enabled is initially_enabled
    assert recorder.calls == (['collect', 'disable', 'enable'] if initially_enabled else [])


@pytest.mark.parametrize("problem", ["new_hit", "sensor_lost"])
def test_fake_live_trial_stops_on_new_hit_or_missing_sensor(problem):
    mod = module()
    clock = Clock()
    adapter, obs = source(clock)
    def mutate(sample, count):
        if count >= 5:
            if problem == "sensor_lost":
                return None
            sample["raw_scan"]["ranges"][16] = 0.195
        return sample
    obs.source.mutate = mutate
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {"ok": True, "choice": "forward",
                                                           "captured_at": payload["captured_at"]})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result["reason"] == ("front_blocked" if problem == "new_hit" else "sensor_unavailable")
    frames = [json.loads(line) for line in relay.transport.lines]
    assert any(frame["linear"] > 0 for frame in frames) and frames[-1]["linear"] == 0


def test_missing_guard_evidence_prevents_nonzero_motion():
    mod = module()
    clock = Clock()
    class MissingEvidence(FakeSerialGuard):
        def evidence(self):
            return {}
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {"ok": True, "choice": "forward",
                                                           "captured_at": payload["captured_at"]})
    result = mod.LiveTrial(source(clock)[0], model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=MissingEvidence()).run()
    assert result["reason"] == "serial_exclusivity_lost"
    assert all(json.loads(line)["linear"] == 0 for line in relay.transport.lines)


def test_live_trial_waits_for_new_image_and_scan_after_session_begins():
    mod = module()
    clock = Clock()
    adapter, obs = source(clock)
    def mutate(sample, count):
        if count == 1:
            old = clock.now() - 0.02
            sample["scan_at"] = old
            sample["color_at"] = old
            sample["raw_scan"]["stamp"] = old
            sample["scene"]["coverage"]["scan_stamp"] = old
            for item in sample["scene"]["rays"] + sample["scene"]["occupied_hits"]:
                item["stamp"] = old
        return sample
    obs.source.mutate = mutate
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {"ok": True, "choice": "forward",
                                                           "captured_at": payload["captured_at"]})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result["reason"] == "motion_limit"
    assert obs.source.calls >= 2 and relay.transport.lines


def test_preflight_can_wait_over_one_second_without_sending_motion():
    mod = module()
    clock = Clock()
    adapter, obs = source(clock)

    def delayed_status(sample, count):
        if count <= 110:
            sample['status_at'] = 99.9
            sample['state']['stamp'] = 99.9
        return sample

    obs.source.mutate = delayed_status
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert obs.source.calls > 110
    assert result['reason'] == 'motion_limit'


def test_temporary_missing_observation_before_arming_is_retried_without_motion():
    mod = module()
    clock = Clock()
    adapter, observation = source(clock)

    def gap(sample, count):
        return None if count == 2 else sample

    observation.source.mutate = gap
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert observation.source.calls > 2
    assert result['reason'] == 'motion_limit'
    assert json.loads(relay.transport.lines[0])['linear'] == 0.0


def test_temporary_missing_stationary_history_is_retried_before_motion():
    mod = module()
    clock = Clock()
    adapter, observation = source(clock)
    calls = []
    def trace_gap(trace):
        calls.append(True)
        return [] if len(calls) == 1 else trace
    observation.source.trace_mutate = trace_gap
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert len(calls) > 1 and result['motion_requested'] is True
    assert json.loads(relay.transport.lines[0])['linear'] == 0.0


@pytest.mark.parametrize('moving', [False, True])
def test_only_a_stationary_history_prefix_can_wait_for_a_delayed_tail(moving):
    mod = module()
    import forward_trial
    clock = Clock()
    adapter, observation = source(clock)
    before = adapter.snapshot()
    trace = observation.source.stationary_trace(99.8, 99.82)
    if moving:
        trace[-1]['linear_x_mps'] = .01
    trial = mod.LiveTrial(adapter, None, None, None, clock)
    with pytest.raises(forward_trial._StopError):
        trial._trace_failed(trace, 99.8, 99.9733, 'stationary_trace_head', before)
    assert trial._last_diagnostic['retryable'] is (not moving)


def test_malformed_stationary_history_still_fails_without_motion():
    mod = module()
    clock = Clock()
    adapter, observation = source(clock)
    observation.source.trace_mutate = lambda trace: [None]
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result['reason'] == 'sensor_unavailable'
    assert result['motion_requested'] is False


@pytest.mark.parametrize('tail_moves', [False, True])
def test_stationary_history_processing_delay_rechecks_the_missing_tail(tail_moves):
    mod = module()
    clock = Clock()
    adapter, observation = source(clock)
    trace_calls = []

    def delayed_trace(trace):
        trace_calls.append(len(trace_calls))
        if len(trace_calls) == 1:
            clock.t += 0.12
        elif len(trace_calls) == 2 and tail_moves:
            trace[len(trace) // 2]['linear_y_mps'] = 0.02
        return trace

    observation.source.trace_mutate = delayed_trace
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert len(trace_calls) >= 2
    assert result['reason'] == ('sensor_unavailable' if tail_moves else 'motion_limit')
    if tail_moves:
        assert result['motion_requested'] is False


def test_new_sample_is_captured_after_the_long_stationary_history_is_checked():
    mod = module()
    clock = Clock()
    adapter, observation = source(clock)
    trace_calls = []

    def delayed_trace(trace):
        trace_calls.append(1)
        if len(trace) > 10:
            clock.t += 0.25
        return trace

    observation.source.trace_mutate = delayed_trace
    relay = mod.RelayAdapter(Transport(), clock)
    def invoke(payload, timeout):
        clock.t += 2.0
        return {'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']}
    model = mod.DecisionsAdapter(invoke)
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result['reason'] == 'motion_limit'
    assert observation.source.calls >= 3


def test_stationary_alignment_checks_current_sensors_once_after_history():
    mod = module()
    clock = Clock()
    adapter, _ = source(clock)
    adapter.recheck_snapshot = False
    before = adapter.snapshot()
    relay = mod.RelayAdapter(Transport(), clock)
    trial = mod.LiveTrial(adapter, None, relay, None, clock)
    trial._stationary_with_retry = lambda _before: clock.now()
    trial._model_fresh = lambda _sample, _now: None
    checks = []
    adapter.is_current_sample = lambda _sample: checks.append(True) or True
    sample, _ = trial._aligned_stationary_sample(before)
    assert sample is not None and len(checks) == 1


def test_stationary_alignment_cannot_accept_a_failed_final_sensor_recheck():
    import forward_trial
    mod = module()
    clock = Clock()
    adapter, _ = source(clock)
    adapter.recheck_snapshot = False
    before = adapter.snapshot()
    relay = mod.RelayAdapter(Transport(), clock)
    trial = mod.LiveTrial(adapter, None, relay, None, clock)
    trial._stationary_with_retry = lambda _before: clock.now()
    trial._model_fresh = lambda _sample, _now: None
    adapter.is_current_sample = lambda _sample: False
    with pytest.raises(forward_trial._StopError):
        trial._aligned_stationary_sample(before)


def test_first_command_reuses_fresh_post_relay_sample_then_handles_later_sensor_loss():
    mod = module()
    clock = Clock()
    adapter, observation = source(clock)
    observation.source.mutate = lambda sample, count: None if count >= 4 else sample
    relay = mod.RelayAdapter(Transport(), clock)
    model = mod.DecisionsAdapter(lambda payload, timeout: {
        'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']})
    result = mod.LiveTrial(adapter, model, relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result['motion_requested'] is True
    assert result['reason'] == 'sensor_unavailable'
    assert any(json.loads(line)['linear'] > 0 for line in relay.transport.lines)
    assert json.loads(relay.transport.lines[-1])['linear'] == 0


def test_model_evidence_timestamp_cannot_precede_session_start():
    mod = module()
    clock = Clock()
    adapter, obs = source(clock)
    def mutate(sample, count):
        if count == 1:
            sample["status_at"] -= 0.1
            sample["state"]["stamp"] = sample["status_at"]
        return sample
    obs.source.mutate = mutate
    relay = mod.RelayAdapter(Transport(), clock)
    captured = []
    def invoke(payload, timeout):
        captured.append(payload["captured_at"])
        return {"ok": True, "choice": "forward", "captured_at": payload["captured_at"]}
    result = mod.LiveTrial(adapter, mod.DecisionsAdapter(invoke), relay, FakeArbiter(), clock,
                           enable_motion=True, serial_guard=FakeSerialGuard()).run()
    assert result["reason"] == "motion_limit"
    assert captured and captured[0] >= 100.0
