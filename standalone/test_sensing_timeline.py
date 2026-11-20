"""Read-only perception timing fixture; no ROS or motion transport."""
from sensing_timeline import measure_once


class Clock(object):
    def __init__(self):
        self.t = 10.0

    def now(self):
        return self.t


class Source(object):
    def __init__(self, clock, sample):
        self.clock, self.sample, self.last_issue = clock, sample, None

    def snapshot(self):
        self.clock.t += .096
        return self.sample


class Recorder(object):
    def __init__(self):
        self.events = []

    def record(self, event, at, fields):
        self.events.append((event, at, fields))
        return True


def test_measures_once_with_original_acquisition_times_and_assessment():
    clock = Clock()
    sample = {'epoch': 1, 'scan_at': 9.9, 'depth_at': 9.95, 'odom_at': 10.0,
              'status_at': 9.7, 'color_at': 9.95, 'snapshot_at': 10.05,
              'scene': {'rays': [1, 2], 'coverage': {'front_samples': 40}}}
    source = Source(clock, sample)
    log = Recorder()
    calls = []
    def assess(value, now):
        calls.append((value, now))
        clock.t += .005
        return None, {'candidate': True, 'valid_until': 10.15}, {}
    result = measure_once(source, assess, clock, log)
    assert len(calls) == 1 and result['reason'] == 'ready'
    fields = log.events[0][2]
    assert fields['scan_at'] == 9.9 and fields['depth_at'] == 9.95
    assert fields['perception_start'] == 10.0 and fields['perception_end'] == clock.t
    assert fields['valid_until'] == 10.15 and fields['ray_count'] == 2
    assert abs(fields['processing_sec'] - .101) < 1e-9


def test_missing_and_exception_are_recorded_without_message_or_payload():
    clock, log = Clock(), Recorder()
    source = Source(clock, None)
    source.last_issue = 'source_missing_scan'
    result = measure_once(source, lambda *_: None, clock, log)
    assert result['reason'] == 'source_missing_scan'
    class Fault(Source):
        def snapshot(self):
            raise ValueError('sensitive exception contents')
    result = measure_once(Fault(clock, None), lambda *_: None, clock, log)
    assert result['reason'] == 'exception' and result['error_class'] == 'ValueError'
    assert 'sensitive' not in str(log.events)
