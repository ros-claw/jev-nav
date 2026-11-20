"""Threaded per-frame processing with bounded queues, no ROS or motion."""
import threading
import time
import os

from independent_worker import IndependentWorkers


class Clock(object):
    def __init__(self):
        self.t = 10.0

    def now(self):
        return self.t


class Timeline(object):
    def __init__(self):
        self.events = []

    def record(self, event, at, fields):
        self.events.append((event, at, fields))
        return True


def test_processes_each_owned_frame_once_and_never_renews_stamp():
    clock, timeline = Clock(), Timeline()
    processed = []
    def process(payload, _now):
        processed.append(payload)
        return {'clear': True, 'valid_until': payload + .15, 'reason': 'sampled'}
    workers = IndependentWorkers({'depth': .15}, .1, {'depth': process}, clock, timeline)
    workers.start()
    try:
        assert workers.submit('depth', 10., 10., 1, 10.)
        for _ in range(100):
            if workers.read(10., 1)['state'] == 'ready':
                break
            time.sleep(.002)
        assert processed == [10.]
        assert workers.read(10., 1)['valid_until'] <= 10.1
        assert len([e for e in timeline.events if e[2].get('stage') == 'depth_result']) == 1
    finally:
        assert workers.close(1.)


def test_ingress_preserves_callback_time_separately_from_clock_pair_receipt():
    clock, timeline = Clock(), Timeline()
    workers = IndependentWorkers({'depth': .15}, .1,
                                 {'depth': lambda _p, _n: {'clear': None, 'valid_until': None,
                                                           'reason': 'unavailable'}}, clock, timeline)
    workers.start()
    try:
        assert workers.submit('depth', 10., 10., 1, {'callback_at': 9.99})
        assert timeline.events[0][2]['callback_at'] == 9.99
        assert timeline.events[0][2]['sensor_received_at'] == 10.
    finally:
        workers.close(1.)


def test_queue_overflow_and_processing_exception_latch_fault():
    clock, timeline = Clock(), Timeline()
    entered, release = threading.Event(), threading.Event()
    def block(_payload, _now):
        entered.set()
        release.wait(1.)
        return {'clear': None, 'valid_until': None, 'reason': 'unavailable'}
    workers = IndependentWorkers({'depth': .15}, .1, {'depth': block}, clock, timeline,
                                 queue_size=1)
    workers.start()
    try:
        assert workers.submit('depth', 10., 10., 1, None)
        assert entered.wait(1.)
        clock.t = 10.01
        assert workers.submit('depth', 10.01, 10.01, 1, None)
        clock.t = 10.02
        assert not workers.submit('depth', 10.02, 10.02, 1, None)
        assert workers.read(10.02, 1)['state'] == 'fault'
    finally:
        release.set()
        workers.close(1.)
    def crash(_payload, _now):
        raise ValueError('secret details should not enter timeline')
    clock = Clock()
    timeline = Timeline()
    workers = IndependentWorkers({'depth': .15}, .1, {'depth': crash}, clock, timeline)
    workers.start()
    try:
        assert workers.submit('depth', 10., 10., 1, None)
        for _ in range(100):
            if workers.read(10., 1)['state'] == 'fault':
                break
            time.sleep(.002)
        assert workers.read(10., 1)['reason'] == 'processor_exception'
        assert 'secret' not in str(timeline.events)
    finally:
        workers.close(1.)


def test_same_stream_concurrent_callbacks_preserve_ingest_and_queue_order():
    clock = Clock()
    clock.t = 10.02
    entered, release = threading.Event(), threading.Event()
    class DelayedTimeline(Timeline):
        def record(self, event, at, fields):
            if fields.get('stage') == 'depth_ingress' and fields.get('sequence') == 0:
                entered.set()
                assert release.wait(2.)
            return Timeline.record(self, event, at, fields)
    log, processed = DelayedTimeline(), []
    def process(payload, _now):
        processed.append(payload)
        return {'clear': True, 'valid_until': payload + .15, 'reason': 'sampled'}
    workers = IndependentWorkers({'depth': .15}, .1, {'depth': process}, clock, log)
    workers.start()
    first = threading.Thread(target=workers.submit, args=('depth', 10., 10., 1, 10.))
    second = threading.Thread(target=workers.submit, args=('depth', 10.01, 10.01, 1, 10.01))
    try:
        first.start()
        assert entered.wait(1.)
        second.start()
        time.sleep(.03)
        release.set()
        first.join(1.)
        second.join(1.)
        for _ in range(100):
            if len(processed) == 2:
                break
            time.sleep(.002)
        assert processed == [10., 10.01]
        assert workers.read(10.02, 1)['state'] == 'ready'
    finally:
        release.set()
        workers.close(1.)


def test_close_drains_accepted_frames_before_reporting_complete():
    clock, timeline = Clock(), Timeline()
    clock.t = 10.02
    entered, release = threading.Event(), threading.Event()
    processed = []
    def process(payload, _now):
        if not processed:
            entered.set()
            assert release.wait(1.)
        processed.append(payload)
        return {'clear': True, 'valid_until': payload + .15, 'reason': 'sampled'}
    workers = IndependentWorkers({'depth': .15}, .1, {'depth': process}, clock, timeline)
    workers.start()
    assert workers.submit('depth', 10., 10., 1, 10.)
    assert entered.wait(1.)
    assert workers.submit('depth', 10.01, 10.01, 1, 10.01)
    result = []
    closer = threading.Thread(target=lambda: result.append(workers.close(1.)))
    closer.start()
    release.set()
    closer.join(2.)
    assert result == [True] and processed == [10., 10.01]
    assert workers.unprocessed_at_close == 0


def test_backlog_fault_closes_private_channel_without_waiting_for_poll():
    from frame_pipe import FrameReader, FrameWriter, PipeTimeline
    clock = Clock()
    entered, release = threading.Event(), threading.Event()
    def block(_payload, _now):
        entered.set()
        release.wait(1.)
        return {'clear': None, 'valid_until': None, 'reason': 'unavailable'}
    read_fd, write_fd = os.pipe()
    reader = FrameReader(read_fd, 'depth')
    timeline = PipeTimeline(Timeline(), FrameWriter(write_fd, 'depth'), 'depth')
    workers = IndependentWorkers({'depth': .15}, .1, {'depth': block}, clock, timeline, queue_size=1)
    workers.start()
    try:
        assert workers.submit('depth', 10., 10., 1, None)
        assert entered.wait(1.)
        clock.t = 10.01
        assert workers.submit('depth', 10.01, 10.01, 1, None)
        clock.t = 10.02
        assert not workers.submit('depth', 10.02, 10.02, 1, None)
        assert reader.poll()[1] == 'producer_eof'
    finally:
        release.set()
        workers.close(1.)
        reader.close()
