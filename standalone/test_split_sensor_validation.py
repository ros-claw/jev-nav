import threading
import time

from relay import _monotonic
from split_sensor_validation import SplitSensorValidation


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

    def combine(_sample, scan, depth):
        if depth != generation['depth']:
            return None, 'prepared_depth_advanced'
        return (scan, depth), None

    workers = SplitSensorValidation(capture, check, combine)
    try:
        evidence, issue = workers.prepare({'snapshot_at': _monotonic()}, budget=.15)
        assert advance.is_set() and issue is None and evidence == (1, 2)
        assert calls == {'scan': [1], 'depth': [1, 2]}
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
