"""Offline control-session and relay-protocol regression checks."""
from __future__ import division

import json
import threading

from control_session import ControlSession
from relay import FrameReader, RelayGate
from test_policy import healthy_depth, healthy_state, sample


def evidence(now, pose=(0.0, 0.0, 0.0), epoch=7):
    return {'epoch': epoch, 'snapshot_at': now, 'odom_at': now,
            'pose': pose, 'scene': sample(now),
            'state': healthy_state(now), 'depth': healthy_depth(now)}


def decision(stamp, choice='forward', pose=(0.0, 0.0, 0.0), epoch=7):
    return {'ok': True, 'choice': choice, 'captured_at': stamp,
            'image_pose': pose, 'epoch': epoch}


def test_forward_decision_generates_bounded_relay_heartbeats():
    session = ControlSession(10.0, 20.0, epoch=7)
    gate = RelayGate(10.0)
    reader = FrameReader()
    first = session.step(evidence(10.05), None, (0, 0, 0), 10.05, 7)
    assert first['frame']['linear'] == 0 and not first['latched']
    assert gate.accept(reader.feed((json.dumps(first['frame']) + '\n').encode('ascii'))[0], 10.05)
    approved = decision(10.05)
    outputs = [session.step(evidence(t), approved, (0, 0, 0), t, 7)
               for t in (10.1, 10.15, 10.2, 10.25)]
    assert all(0 < output['frame']['linear'] <= 0.03 for output in outputs)
    assert [output['frame']['sequence'] for output in outputs] == [1, 2, 3, 4]
    for output in outputs:
        wire = (json.dumps(output['frame'], allow_nan=False) + '\n').encode('ascii')
        frame = reader.feed(wire)[0]
        assert gate.accept(frame, frame['stamp'])
        assert gate.command(frame['stamp']) == (frame['linear'], 0.0)
    assert not outputs[-1]['latched']


def test_plain_choice_and_stamp_cannot_authorize_motion():
    session = ControlSession(10.0, 20.0, epoch=7)
    result = session.step(evidence(10.1), {'choice': 'forward', 'stamp': 10.1},
                          (0, 0, 0), 10.1, 7)
    assert result['frame']['linear'] == 0 and result['latched']


def test_stale_pose_or_sensor_loss_latches_after_motion():
    session = ControlSession(10.0, 20.0, epoch=7)
    approved = decision(10.1)
    assert session.step(evidence(10.1), approved, (0, 0, 0), 10.1, 7)['frame']['linear'] > 0
    stale = session.step(evidence(10.1), approved, (0, 0, 0), 10.25, 7)
    assert stale['frame']['linear'] == 0 and stale['latched']
    restored = session.step(evidence(10.3), approved, (0, 0, 0), 10.3, 7)
    assert restored['frame']['linear'] == 0 and restored['latched']


def test_opt_in_shadow_age_budget_keeps_current_pose_bound():
    session = ControlSession(10.0, 20.0, epoch=7, snapshot_max_age_sec=0.15)
    approved = decision(10.05)
    first = session.step(evidence(10.05), approved, (0, 0, 0), 10.05, 7)
    assert first['frame']['linear'] > 0
    recent = session.step(evidence(10.05), approved, (0.001, 0, 0), 10.18, 7)
    assert recent['frame']['linear'] > 0 and not recent['latched']
    assert session.step(evidence(10.05), approved, (0.001, 0, 0), 10.21, 7)['latched']


def test_opt_in_shadow_age_budget_cannot_exceed_150ms():
    try:
        ControlSession(10.0, 20.0, epoch=7, snapshot_max_age_sec=0.16)
    except ValueError:
        return
    raise AssertionError('unbounded snapshot age was accepted')


def test_live_epoch_change_invalidates_even_a_fresh_old_epoch_sample():
    session = ControlSession(10.0, 20.0, epoch=7)
    approved = decision(10.05)
    assert session.step(evidence(10.05), approved, (0, 0, 0), 10.05, 7)['frame']['linear'] > 0
    result = session.step(evidence(10.1), approved, (0, 0, 0), 10.1, 8)
    assert result['frame']['linear'] == 0 and result['latched']
    assert session.step(evidence(10.15), approved, (0, 0, 0), 10.15, 7)['frame']['linear'] == 0


def test_decision_epoch_and_capture_pose_are_mandatory():
    for bad in (decision(10.1, epoch=8), decision(10.1, pose=(0.2, 0, 0)),
                decision(10.1, pose=None)):
        session = ControlSession(10.0, 20.0, epoch=7)
        result = session.step(evidence(10.1), bad, (0, 0, 0), 10.1, 7)
        assert result['frame']['linear'] == 0 and result['latched']


def test_decision_cannot_remain_valid_after_pose_moves_past_capture_limit():
    session = ControlSession(10.0, 20.0, epoch=7)
    approved = decision(10.05)
    result = None
    for i in range(1, 105):
        now = 10.0 + i * 0.05
        pose = (max(0.0, (now - 10.05) * 0.03), 0.0, 0.0)
        result = session.step(evidence(now, pose), approved, pose, now, 7)
        if result['latched']:
            break
    assert pose[0] > 0.15
    assert result['frame']['linear'] == 0 and result['latched']
    now += 0.05
    assert session.step(evidence(now, pose), approved, pose, now, 7)['frame']['linear'] == 0


def test_decision_expires_even_with_fresh_sensors_and_pose():
    session = ControlSession(10.0, 25.0, epoch=7)
    approved = decision(10.125)
    result = None
    for i in range(1, 66):
        now = 10.0 + i * 0.125
        result = session.step(evidence(now), approved, (0, 0, 0), now, 7)
        if result['latched']:
            break
    assert now - approved['captured_at'] >= 8.0
    assert result['frame']['linear'] == 0 and result['latched']


def test_same_stamp_cannot_change_choice_or_capture_pose():
    for changed in (decision(10.05, choice='stop'), decision(10.05, pose=(0.01, 0, 0))):
        session = ControlSession(10.0, 20.0, epoch=7)
        assert session.step(evidence(10.05), decision(10.05), (0, 0, 0), 10.05, 7)['frame']['linear'] > 0
        result = session.step(evidence(10.1), changed, (0, 0, 0), 10.1, 7)
        assert result['frame']['linear'] == 0 and result['latched']


def test_explicit_stop_cannot_resume_motion():
    session = ControlSession(10.0, 20.0, epoch=7)
    approved = decision(10.05)
    assert session.step(evidence(10.05), approved, (0, 0, 0), 10.05, 7)['frame']['linear'] > 0
    result = session.step(evidence(10.1), decision(10.1, choice='stop'), (0, 0, 0), 10.1, 7)
    assert result['frame']['linear'] == 0 and result['latched']
    assert session.step(evidence(10.15), approved, (0, 0, 0), 10.15, 7)['frame']['linear'] == 0


def test_invalid_clock_returns_no_sendable_frame_and_permanently_latches():
    for now in (float('nan'), float('inf'), None, '10.1', 10.05, 10.0, 10.3):
        session = ControlSession(10.0, 20.0, epoch=7)
        approved = decision(10.05)
        assert session.step(evidence(10.05), approved, (0, 0, 0), 10.05, 7)['frame']['linear'] > 0
        bad = session.step(None, None, None, now, 7)
        assert bad['frame'] is None and bad['latched']
        assert bad['reason'] == 'heartbeat_discontinuity'
        later = session.step(evidence(10.1), approved, (0, 0, 0), 10.1, 7)
        assert later['frame'] is None and later['latched']


def test_unexpected_exception_latches_zero_and_cannot_resume():
    class Fault(object):
        started_at = 10.0
        called = False
        def step(self, *args):
            self.called = True
            raise RuntimeError('injected unexpected policy failure')
    session = ControlSession(10.0, 20.0, epoch=7)
    approved = decision(10.05)
    assert session.step(evidence(10.05), approved, (0, 0, 0), 10.05, 7)['frame']['linear'] > 0
    real_policy = session.policy
    fault = Fault()
    session.policy = fault
    failed = session.step(evidence(10.1), approved, (0, 0, 0), 10.1, 7)
    assert failed['frame']['linear'] == 0 and failed['latched']
    assert failed['reason'] == 'control_error' and fault.called
    session.policy = real_policy
    assert session.step(evidence(10.15), approved, (0, 0, 0), 10.15, 7)['frame']['linear'] == 0


def test_concurrent_ticks_are_serialized_before_fault_and_sequence_changes():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    class SlowPolicy(object):
        started_at = 10.0
        def step(self, *args):
            entered.set()
            if not release.wait(2.0):
                raise RuntimeError('test synchronization timed out')
            return {'linear': 0.01, 'angular': 0.0, 'reason': 'stub'}
    session = ControlSession(10.0, 20.0, epoch=7)
    session.policy = SlowPolicy()
    results = {}
    def first():
        results['first'] = session.step(evidence(10.1), decision(10.1), (0, 0, 0), 10.1, 7)
    def second():
        results['second'] = session.step(evidence(10.15), decision(10.1), (0, 0, 0), 10.15, 8)
        finished.set()
    thread_a = threading.Thread(target=first)
    thread_b = threading.Thread(target=second)
    thread_a.daemon = True
    thread_b.daemon = True
    thread_a.start()
    try:
        assert entered.wait(1.0)
        thread_b.start()
        finished_early = finished.wait(0.1)
    finally:
        release.set()
        thread_a.join(2.0)
        if thread_b.ident is not None:
            thread_b.join(2.0)
    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert not finished_early
    assert results['first']['frame']['sequence'] == 0
    assert results['first']['frame']['linear'] > 0 and not results['first']['latched']
    assert results['second']['frame']['sequence'] == 1
    assert results['second']['frame']['linear'] == 0 and results['second']['latched']



def test_invalid_policy_outputs_latch_zero():
    class BadPolicy(object):
        started_at = 10.0
        def __init__(self, linear, angular):
            self.linear, self.angular = linear, angular
        def step(self, *args):
            return {'linear': self.linear, 'angular': self.angular, 'reason': 'invalid'}
    for linear, angular in ((-0.001, 0), (0.050001, 0), (float('nan'), 0),
                            (False, 0), (0.03, 0.000001), (0.03, False)):
        session = ControlSession(10.0, 20.0, epoch=7)
        session.policy = BadPolicy(linear, angular)
        result = session.step(evidence(10.1), decision(10.1), (0, 0, 0), 10.1, 7)
        assert result['frame']['linear'] == 0 and result['latched']
        assert result['reason'] == 'invalid_policy_output'


def test_boolean_epoch_does_not_match_numeric_epoch():
    for sample_epoch, live_epoch, decision_epoch in ((True, 1, 1), (1, True, 1), (1, 1, True)):
        session = ControlSession(10.0, 20.0, epoch=1)
        result = session.step(evidence(10.1, epoch=sample_epoch), decision(10.1, epoch=decision_epoch),
                              (0, 0, 0), 10.1, live_epoch)
        assert result['frame']['linear'] == 0 and result['latched']


def test_observation_loss_invalidates_an_earlier_unused_decision():
    session = ControlSession(10.0, 20.0, epoch=7)
    approved = decision(10.05)
    blocked = evidence(10.05)
    blocked['depth']['min_m'] = 0.1
    first = session.step(blocked, approved, (0, 0, 0), 10.05, 7)
    assert first['frame']['linear'] == 0 and not first['latched']
    missing = session.step(None, None, None, 10.1, 7)
    assert missing['frame']['linear'] == 0 and not missing['latched']
    resumed = session.step(evidence(10.15), approved, (0, 0, 0), 10.15, 7)
    assert resumed['frame']['linear'] == 0 and resumed['latched']


def test_stop_frame_latches_the_relay_and_sequence_stays_ordered():
    session = ControlSession(10.0, 20.0, epoch=7)
    gate = RelayGate(10.0)
    first = session.step(evidence(10.05), decision(10.05), (0, 0, 0), 10.05, 7)['frame']
    second = session.step(evidence(10.1), decision(10.1, choice='stop'), (0, 0, 0), 10.1, 7)['frame']
    assert gate.accept(first, first['stamp']) and gate.command(10.05)[0] > 0
    assert gate.accept(second, second['stamp']) and gate.stopped
    assert gate.command(10.1) == (0.0, 0.0)
    assert second['sequence'] == first['sequence'] + 1


def test_decision_cannot_change_during_gate_validation():
    session = ControlSession(10.0, 20.0, epoch=7)
    supplied = decision(10.1, choice='stop')
    real_accept = session.decision_gate.accept
    def change_external_value(value, captured, image_pose, now, live_pose):
        supplied['choice'] = 'forward'
        return real_accept(value, captured, image_pose, now, live_pose)
    session.decision_gate.accept = change_external_value
    result = session.step(evidence(10.1), supplied, (0, 0, 0), 10.1, 7)
    assert result['frame']['linear'] == 0
