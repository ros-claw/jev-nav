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


def test_unfinished_replacement_preserves_published_sample_without_changing_it():
    from async_safety_sampler import AsyncSafetySampler, WAITING_REPLACEMENT
    entered = threading.Event()
    candidate = {'index': 2}

    class Source(object):
        def snapshot(self):
            return candidate

    def validate(sample, _now):
        assert sample is candidate
        entered.set()
        return WAITING_REPLACEMENT, None

    seed, assessment = {'index': 1}, {'valid_until': _monotonic() + .1}
    sampler = AsyncSafetySampler(Source(), validate, Clock())
    try:
        sampler.start(seed, assessment)
        assert entered.wait(.5)
        current = sampler.poll()
        assert current['sample'] is seed and current['assessment'] is assessment
    finally:
        assert sampler.close(.5)


def test_sampler_waits_for_staging_capacity_before_building_a_scene():
    from async_safety_sampler import AsyncSafetySampler
    waiting, available, captured = threading.Event(), threading.Event(), threading.Event()

    class Source(object):
        def wait_for_stage_slot(self, timeout):
            waiting.set()
            return available.wait(timeout)

        def snapshot(self):
            captured.set()
            return {'index': 1}

    sampler = AsyncSafetySampler(Source(), lambda sample, now: (None, {}), Clock())
    try:
        sampler.start()
        assert waiting.wait(.5)
        assert not captured.wait(.04)
        available.set()
        assert captured.wait(.5)
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


def test_decision_warmup_repeatedly_refreshes_only_until_closed():
    from async_safety_sampler import DecisionWarmup
    class Source(object):
        count = 0
        def snapshot(self):
            self.count += 1
            return None
    source = Source()
    warmup = DecisionWarmup(source)
    warmup.start()
    wait_for(lambda: source.count >= 2)
    assert warmup.close(.5)
    stopped = source.count
    time.sleep(.04)
    assert source.count == stopped


def test_decision_warmup_refuses_to_close_while_snapshot_is_blocked():
    from async_safety_sampler import DecisionWarmup
    entered, release = threading.Event(), threading.Event()
    class Source(object):
        def snapshot(self):
            entered.set()
            release.wait(1.)
    warmup = DecisionWarmup(Source())
    warmup.start()
    assert entered.wait(.5)
    assert warmup.close(.01) is False
    release.set()
    assert warmup.close(.5)


def test_decision_warmup_reports_snapshot_exceptions():
    from async_safety_sampler import DecisionWarmup
    failed = threading.Event()
    class Source(object):
        def snapshot(self):
            failed.set()
            raise RuntimeError('observation failed')
    warmup = DecisionWarmup(Source())
    warmup.start()
    assert failed.wait(.5)
    assert warmup.close(.5) is False
