"""Bounded timeline tests; no ROS, network, subprocess or device."""
import json
import os
import threading

import pytest

from nav_timeline import Timeline


def test_timeline_records_acquisition_and_processing_times_without_payload(tmpdir):
    path = str(tmpdir.join('events.jsonl'))
    log = Timeline(path, 'run-1', capacity=8)
    assert log.record('perception', 100.25, {'sensor_acquired_at': 100.0,
                                              'perception_start': 100.1,
                                              'perception_end': 100.25,
                                              'reason': 'ready', 'ray_count': 42})
    assert log.close(2.0)
    rows = [json.loads(line) for line in open(path)]
    assert rows == [{'run_id': 'run-1', 'event': 'perception', 'at': 100.25,
                     'sensor_acquired_at': 100.0, 'perception_start': 100.1,
                     'perception_end': 100.25, 'reason': 'ready', 'ray_count': 42}]
    assert os.stat(path).st_mode & 0o777 == 0o600


@pytest.mark.parametrize('fields', [
    {'authorization': 'secret'}, {'image': 'payload'}, {'reason': 'x' * 129},
    {'sensor_acquired_at': float('nan')}, {'reason': {'raw': 'payload'}},
])
def test_timeline_rejects_unapproved_or_malformed_fields(tmpdir, fields):
    log = Timeline(str(tmpdir.join('events.jsonl')), 'run-1')
    assert not log.record('perception', 1.0, fields)
    assert log.dropped == 1
    assert not log.close(2.0)


def test_timeline_refuses_to_overwrite_and_reports_queue_loss(tmpdir):
    path = str(tmpdir.join('events.jsonl'))
    log = Timeline(path, 'run-1', capacity=1, autostart=False)
    assert log.record('control', 1.0, {'reason': 'test'})
    assert not log.record('control', 1.1, {'reason': 'test'})
    assert log.dropped == 1
    log.start()
    assert not log.close(2.0)
    with pytest.raises(OSError):
        Timeline(path, 'another')


def test_blocked_disk_does_not_hold_record_lock(tmpdir):
    entered, release = threading.Event(), threading.Event()
    class BlockedTimeline(Timeline):
        def _write(self, value):
            entered.set()
            assert release.wait(2.0)
            return Timeline._write(self, value)
    log = BlockedTimeline(str(tmpdir.join('events.jsonl')), 'run-1', capacity=2)
    try:
        assert log.record('control', 1.0, {'reason': 'test'})
        assert entered.wait(1.0)
        assert log.record('control', 1.1, {'reason': 'test'})
    finally:
        release.set()
        assert log.close(2.0)


def test_closing_before_start_is_idempotent_and_invalid_event_is_rejected(tmpdir):
    log = Timeline(str(tmpdir.join('events.jsonl')), 'run-1', autostart=False)
    assert not log.record([], 1.0, {})
    assert not log.close(1.0)
    assert not log.close(1.0)


def test_timeline_can_record_explicit_unknown_and_clock_domain(tmpdir):
    log = Timeline(str(tmpdir.join('events.jsonl')), 'run-1')
    assert log.record('sensor', 10., {'clock_domain': 'ros_bag_time', 'invalid_count': 3,
                                      'unobserved_space': True, 'sampled_candidate': True})
    assert log.close(1.)
