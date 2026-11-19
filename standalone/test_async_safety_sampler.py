"""Synthetic tests; no ROS, network, serial or robot."""
import threading
import time

from relay import _monotonic


class Clock(object):
    def now(self):
        return _monotonic()


def wait_for(predicate):
    deadline = _monotonic() + 1.5
    while _monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.005)
    raise AssertionError('sampler did not produce the expected state')


def test_sampler_delivers_new_validated_snapshots_without_blocking_consumer():
    from async_safety_sampler import AsyncSafetySampler
    class Source(object):
        count = 0
        def snapshot(self):
            self.count += 1
            return {'index': self.count}
    sampler = AsyncSafetySampler(Source(), lambda sample, now: (None, {'accepted': sample['index']}), Clock())
    try:
        sampler.start()
        first = wait_for(lambda: sampler.poll() if sampler.poll()['state'] == 'ready' else None)
        first_index = first['sample']['index']
        later = wait_for(lambda: sampler.poll() if sampler.poll()['sample']['index'] > first_index else None)
        assert later['assessment']['accepted'] == later['sample']['index']
    finally:
        assert sampler.close(.5)


def test_sampler_latches_sensor_failure_and_discards_earlier_success():
    from async_safety_sampler import AsyncSafetySampler
    class Source(object):
        count = 0
        def snapshot(self):
            self.count += 1
            return {'index': self.count} if self.count == 1 else None
    sampler = AsyncSafetySampler(Source(), lambda sample, now: (None, {}), Clock())
    try:
        sampler.start()
        failed = wait_for(lambda: sampler.poll() if sampler.poll()['state'] == 'fault' else None)
        assert failed['reason'] == 'sensor_unavailable' and failed['sample'] is None
    finally:
        assert sampler.close(.5)


def test_explicitly_transient_source_gaps_can_recover_without_fabricating_a_sample():
    from async_safety_sampler import AsyncSafetySampler
    class Source(object):
        count = 0
        last_issue = 'temporary_alignment'
        def snapshot(self):
            self.count += 1
            if 2 <= self.count <= 4:
                return None
            return {'index': self.count}
    sampler = AsyncSafetySampler(Source(), lambda sample, now: (None, {}), Clock(),
                                  transient_unavailable=lambda source: source.last_issue == 'temporary_alignment')
    try:
        sampler.start()
        recovered = wait_for(lambda: sampler.poll() if sampler.poll()['state'] == 'ready'
                             and sampler.poll()['sample']['index'] >= 5 else None)
        assert recovered['sample']['index'] >= 5
    finally:
        assert sampler.close(.5)


def test_blocked_sampler_does_not_block_poll_or_bounded_close():
    from async_safety_sampler import AsyncSafetySampler
    entered, release = threading.Event(), threading.Event()
    class Source(object):
        def snapshot(self):
            entered.set()
            release.wait(1.0)
            return {'index': 1}
    sampler = AsyncSafetySampler(Source(), lambda sample, now: (None, {}), Clock())
    sampler.start()
    assert entered.wait(.5)
    started = _monotonic()
    assert sampler.poll()['state'] == 'waiting'
    assert sampler.close(.02) is False
    assert _monotonic() - started < .2
    release.set()
    assert sampler.close(.5)
