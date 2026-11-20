"""Offline-only verdict pipe tests; no ROS, devices, network, or velocity output."""
from __future__ import division

import json
import os
import time

import pytest

try:
    import sensor_ipc
except ImportError:
    sensor_ipc = None


class Clock(object):
    def __init__(self, now=100.0):
        self.value = now
    def __call__(self):
        return self.value


def implementation():
    assert sensor_ipc is not None, 'sensor_ipc must provide the verdict pipe'
    return sensor_ipc


def make_channel(clock=None, epoch=7):
    mod = implementation()
    clock = clock if clock is not None else Clock()
    read_fd, write_fd = os.pipe()
    reader = mod.VerdictReader(read_fd, epoch=epoch, clock=clock, startup_timeout_sec=1.0)
    writer = mod.VerdictWriter(write_fd, epoch=epoch, clock=clock)
    return reader, writer, clock


def frame(sequence=0, stamp=100.0, epoch=7, result='pass', valid_until=100.1):
    return {'sequence': sequence, 'stamp': stamp, 'epoch': epoch,
            'result': result, 'valid_until': valid_until}


def raw_write(fd, item):
    encoded = (json.dumps(item, separators=(',', ':')) + '\n').encode('ascii')
    os.write(fd, encoded)


def test_fresh_pass_then_new_reject_latches_and_cannot_be_overwritten():
    reader, writer, clock = make_channel()
    try:
        assert writer.send(0, 100.0, 7, 'pass', 100.1)
        assert reader.poll()['status'] == 'pass'
        clock.value += 0.01
        assert writer.send(1, clock.value, 7, 'reject', clock.value + 0.1)
        verdict = reader.poll()
        assert verdict['status'] == 'reject' and verdict['sequence'] == 1
        assert verdict['latched'] and reader.poll()['status'] == 'reject'
        assert not writer.send(2, clock.value + 0.01, 7, 'pass', clock.value + 0.1)
    finally:
        reader.close()
        writer.close()


def test_buffered_pass_cannot_hide_a_later_reject():
    reader, writer, clock = make_channel()
    try:
        assert writer.send(0, 100.0, 7, 'pass', 100.1)
        clock.value += 0.01
        assert writer.send(1, clock.value, 7, 'reject', clock.value + 0.1)
        state = reader.poll()
        assert state['status'] == 'reject' and state['sequence'] == 1
    finally:
        reader.close()
        writer.close()


@pytest.mark.parametrize('frames', [
    [frame(sequence=1)],
    [frame(), frame(sequence=0, stamp=100.01, valid_until=100.1)],
    [frame(), frame(sequence=2, stamp=100.01, valid_until=100.1)],
    [frame(), frame(sequence=1, stamp=100.0, valid_until=100.1)],
    [frame(epoch=8)],
    [frame(stamp=100.01, valid_until=100.1)],
    [frame(stamp=99.9, valid_until=99.99)],
    [frame(valid_until=100.151)],
    [frame(stamp=float('nan'))],
    [frame(valid_until=float('inf'))],
    [frame(sequence=True)],
])
def test_invalid_or_replayed_frames_fail_closed(frames):
    reader, writer, clock = make_channel()
    try:
        clock.value = 100.02
        # Future and expired cases are relative to this test clock.
        if len(frames) == 1 and frames[0].get('stamp') == 100.01:
            clock.value = 100.0
        for item in frames:
            raw_write(writer.fd, item)
        state = reader.poll()
        assert state['status'] == 'fault' and state['latched']
        assert reader.poll()['status'] == 'fault'
    finally:
        reader.close()
        writer.close()


def test_duplicate_keys_and_oversized_frames_fail_closed():
    for wire in (b'{"sequence":0,"sequence":0}\n', b'x' * 513 + b'\n'):
        reader, writer, _ = make_channel()
        try:
            os.write(writer.fd, wire)
            assert reader.poll()['status'] == 'fault'
        finally:
            reader.close()
            writer.close()


def test_partial_frame_never_leaves_a_previous_pass_usable_and_eof_faults():
    reader, writer, clock = make_channel()
    try:
        assert writer.send(0, 100.0, 7, 'pass', 100.1)
        assert reader.poll()['status'] == 'pass'
        os.write(writer.fd, b'{"sequence":1')
        assert reader.poll()['status'] == 'waiting'
        writer.close()
        state = reader.poll()
        assert state['status'] == 'fault' and state['reason'] == 'truncated'
    finally:
        reader.close()
        writer.close()


def test_no_frame_and_stalled_producer_expire_without_busy_wait():
    reader, writer, clock = make_channel()
    try:
        assert reader.poll()['status'] == 'waiting'
        assert writer.send(0, 100.0, 7, 'pass', 100.1)
        assert reader.poll()['status'] == 'pass'
        clock.value = 100.101
        state = reader.poll()
        assert state['status'] == 'fault' and state['reason'] == 'expired'
    finally:
        reader.close()
        writer.close()
    reader, writer, clock = make_channel()
    try:
        clock.value += 1.01
        assert reader.poll()['status'] == 'fault'
    finally:
        reader.close()
        writer.close()


def test_producer_exit_with_buffered_pass_is_not_treated_as_live():
    mod = implementation()
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        raw_write(write_fd, frame())
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    os.waitpid(pid, 0)
    reader = mod.VerdictReader(read_fd, epoch=7, clock=Clock(), startup_timeout_sec=1.0)
    try:
        state = reader.poll()
        assert state['status'] == 'fault' and state['reason'] == 'producer_closed'
    finally:
        reader.close()


def test_consumer_close_or_backpressure_stops_writer_without_blocking():
    reader, writer, clock = make_channel()
    reader.close()
    assert not writer.send(0, 100.0, 7, 'pass', 100.1)
    assert writer.closed
    writer.close()

    reader, writer, clock = make_channel()
    try:
        started = time.time()
        for sequence in range(10000):
            clock.value = 100.0 + sequence * 0.00001
            if not writer.send(sequence, clock.value, 7, 'pass', clock.value + 0.1):
                break
        else:
            raise AssertionError('writer never detected full pipe')
        assert writer.closed and time.time() - started < 2.0
        assert reader.poll()['status'] == 'fault'
    finally:
        reader.close()
        writer.close()


def test_wrong_pipe_direction_is_rejected():
    mod = implementation()
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(ValueError):
            mod.VerdictReader(write_fd, epoch=7, clock=Clock())
        with pytest.raises(ValueError):
            mod.VerdictWriter(read_fd, epoch=7, clock=Clock())
    finally:
        os.close(read_fd)
        os.close(write_fd)
