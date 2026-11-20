import threading
import time

from relay import _monotonic
from split_sensor_validation import SensorWakeup, SplitSensorValidation, pending_sensor_deadline


def test_pending_successors_use_the_oldest_recorded_receipt():
    checked = {'generation': 8, 'stamp': 10.}
    latest = {'generation': 9, 'stamp': 10.04, 'received_at': 10.06}
    assert pending_sensor_deadline(latest, checked, 10.07, .3) == 10.13
    assert pending_sensor_deadline(latest, checked, 10.07, .05) == 10.09
    assert pending_sensor_deadline(latest, checked, 10.14, .3) is None
    assert pending_sensor_deadline(dict(latest, generation=10), checked, 10.07, .3) is None
    assert pending_sensor_deadline(dict(latest, received_at=10.1), checked, 10.07, .3) is None
    second = {'generation': 10, 'stamp': 10.07, 'received_at': 10.08}
    receipts = [(9, 10.04, 10.06), (10, 10.07, 10.08)]
    assert pending_sensor_deadline(second, checked, 10.09, .3, receipts) == 10.13
    assert pending_sensor_deadline(second, checked, 10.14, .3, receipts) is None
    assert pending_sensor_deadline(dict(second, generation=11), checked, 10.09, .3, receipts) is None


def test_sensor_wakeup_clears_notifications_and_wakes_a_waiter():
    event = SensorWakeup()
    try:
        event.set()
        event.set()
        assert event.wait(0.)
        event.clear()
        assert not event.wait(0.)
        result = []
        worker = threading.Thread(target=lambda: result.append(event.wait(.2)))
        worker.start()
        event.set()
        worker.join(.25)
        assert result == [True]
    finally:
        event.close()
    event.set()


def test_depth_advance_rechecks_depth_without_recomputing_scan():
    generation = {'scan': 1, 'depth': 1}
    calls = {'scan': [], 'depth': []}
    advance = threading.Event()

    def capture(stream, _sample):
        now = generation[stream]
        return now, now

    def check(stream, _sample, value):
        calls[stream].append(value)
        if stream == 'depth' and value == 1:
            generation['depth'] = 2
            advance.set()
        return value, None

    def combine(_sample, scan, depth, nonblocking=False):
        if depth != generation['depth']:
            return None, 'prepared_depth_advanced'
        return (scan, depth), None

    workers = SplitSensorValidation(capture, check, combine)
    try:
        first = {'snapshot_at': _monotonic()}
        evidence, issue = workers.prepare(first, budget=.15)
        assert advance.is_set() and issue is None and evidence == (1, 2)
        assert calls == {'scan': [1], 'depth': [1, 2]}
        workers.commit(first)
        sample = {'snapshot_at': _monotonic()}
        assert workers.prepare(sample, budget=.15) == ((1, 2), None)
        generation['depth'] = 3
        deadline = _monotonic() + .2
        while workers.read(sample) != ((1, 3), None) and _monotonic() < deadline:
            time.sleep(.005)
        assert workers.read(sample) == ((1, 3), None)
        assert calls['scan'] == [1, 1]
    finally:
        assert workers.close()


def test_confirmed_stop_rejects_even_when_the_other_sensor_is_ready():
    workers = SplitSensorValidation(
        lambda stream, _: (1, stream),
        lambda stream, _sample, _: (None, 'front_blocked') if stream == 'scan' else (1, None),
        lambda *_: (True, None))
    try:
        for _ in range(2):
            assert workers.prepare({'snapshot_at': _monotonic()}, budget=.15) == (None, 'front_blocked')
    finally:
        assert workers.close()


def test_replacement_keeps_active_proof_until_submission():
    workers = SplitSensorValidation(
        lambda stream, _sample: (stream, stream),
        lambda _stream, sample, _value: (sample['name'], None),
        lambda _sample, scan, depth, **_: ((scan, depth), None))
    try:
        old = {'snapshot_at': _monotonic(), 'name': 'old'}
        new = {'snapshot_at': _monotonic(), 'name': 'new'}
        workers.seed(old, 'old', 'old')
        assert workers.prepare(new, budget=.15) == (('new', 'new'), None)
        assert workers.read(old) == (('old', 'old'), None)
        with workers._cv:
            writer = threading.Thread(target=workers.commit, args=(new,))
            writer.start()
            writer.join(.25)
            assert not writer.is_alive()
            assert workers.read(new) == (('new', 'new'), None)
            assert workers.read(old) == (None, 'prepared_pending')
        final = {'snapshot_at': _monotonic(), 'name': 'final'}
        assert workers.prepare(final, budget=.15) == (('final', 'final'), None)
    finally:
        assert workers.close()


def test_staged_proof_validation_does_not_block_active_proof():
    entered = threading.Event()
    release = threading.Event()

    def combine(sample, scan, depth, nonblocking=False):
        if sample['name'] == 'staged':
            entered.set()
            assert release.wait(.5)
        return (scan, depth), None

    workers = SplitSensorValidation(
        lambda stream, _: (1, stream),
        lambda _stream, sample, _: (sample['name'], None), combine)
    old = {'snapshot_at': _monotonic(), 'name': 'active'}
    staged = {'snapshot_at': _monotonic(), 'name': 'staged'}
    outcome = []
    preparation = threading.Thread(target=lambda: outcome.append(workers.prepare(staged, budget=.20)))
    try:
        workers.seed(old, 'active', 'active')
        preparation.start()
        assert entered.wait(.20)
        assert workers.read(old) == (('active', 'active'), None)
    finally:
        release.set()
        preparation.join(.5)
        assert workers.close()
    assert outcome == [(('staged', 'staged'), None)]


def test_projected_obstacle_latches_before_a_newer_completed_frame_can_be_used():
    workers = SplitSensorValidation(
        lambda stream, _: (1, stream),
        lambda _stream, _sample, value: (value, None),
        lambda _sample, scan, depth, **_kwargs: ((scan, depth), None))
    try:
        sample = {'snapshot_at': _monotonic()}
        workers.note_depth_obstacle()
        assert workers._fault is None
        workers.seed(sample, 'scan', 'depth')
        workers.note_depth_obstacle()
        with workers._cv:
            assert workers.read(sample) == (None, 'depth_obstacle')
        assert workers.counts()['depth:producer_obstacle'] == 1
    finally:
        assert workers.close()


def test_staged_scan_proof_can_be_paired_while_a_newer_scan_is_checked():
    entered = threading.Event()
    release = threading.Event()

    def check(stream, _sample, value):
        if stream == 'scan':
            entered.set()
            assert release.wait(.5)
        return value, None

    workers = SplitSensorValidation(
        lambda stream, _: (1, stream), check,
        lambda _sample, scan, depth, **_kwargs: ((scan, depth), None))
    try:
        sample = {'snapshot_at': _monotonic()}
        assert workers.prepare(sample, budget=.15, scan_seed='initial scan') == (('initial scan', 'depth'), None)
        assert entered.wait(.15)
    finally:
        release.set()
        assert workers.close()
