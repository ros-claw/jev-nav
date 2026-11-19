"""Offline tests for the single-writer threaded relay facade; no ROS or devices."""
from __future__ import division

import json
import threading
import time

import pytest
from relay import _monotonic


class Clock(object):
    def now(self):
        return _monotonic()
    def sleep(self, seconds):
        time.sleep(seconds)


class Process(object):
    def __init__(self):
        self.running = True
    def poll(self):
        return None if self.running else 0


class FakeTransport(object):
    def __init__(self, ready=b'READY motion_limit_sec=0.300000\n'):
        self.process = None
        self.ready = ready
        self.lock = threading.Lock()
        self.writes = []
        self.fail_next_write = False
        self.started_limit = None

    def start(self, limit):
        self.started_limit = limit
        self.process = Process()

    def read_line(self, timeout):
        with self.lock:
            result, self.ready = self.ready, None
            return result

    def write_line(self, data, timeout):
        with self.lock:
            if self.fail_next_write:
                self.fail_next_write = False
                return False
            if self.process is None or self.process.poll() is not None:
                return False
            self.writes.append((_monotonic(), threading.current_thread().ident, json.loads(data)))
            return True

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def close(self):
        if self.process is not None:
            self.process.running = False

    def snapshot(self):
        with self.lock:
            return list(self.writes)


def make(transport=None):
    try:
        from threaded_relay_adapter import ThreadedRelayAdapter
    except ImportError:
        pytest.fail('ThreadedRelayAdapter is required', pytrace=False)
    clock = Clock()
    transport = transport if transport is not None else FakeTransport()
    return ThreadedRelayAdapter(transport, clock), transport, clock


def wait_for(predicate, timeout=2.0):
    deadline = _monotonic() + timeout
    while _monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError('timed out waiting for relay condition')


def test_idle_heartbeats_continue_during_control_pause_and_have_one_writer():
    relay, transport, clock = make()
    try:
        assert relay.arm(0.3)['ready_line'] == b'READY motion_limit_sec=0.300000\n'
        assert relay.armed and relay.transport is transport and relay.transport.process is not None
        assert relay.send_zero_heartbeat()
        caller = threading.current_thread().ident
        time.sleep(0.32)
        writes = transport.snapshot()
        assert len(writes) >= 6
        assert all(frame['linear'] == 0.0 and frame['angular'] == 0.0 for _, _, frame in writes)
        assert [frame['sequence'] for _, _, frame in writes] == list(range(len(writes)))
        assert len(set(writer for _, writer, _ in writes)) == 1
        assert writes[0][1] != caller
        assert max(writes[i][0] - writes[i - 1][0] for i in range(1, len(writes))) < 0.12
        assert relay.healthy()
    finally:
        relay.stop_and_latch()
        relay.close()


def test_explicit_longer_session_keeps_the_same_expiring_intent_contract():
    from threaded_relay_adapter import ThreadedRelayAdapter
    clock = Clock()
    transport = FakeTransport(ready=b'READY motion_limit_sec=2.000000\n')
    relay = ThreadedRelayAdapter(transport, clock, motion_limit_sec=2.0)
    try:
        assert relay.arm(2.0)['ready_line'] == b'READY motion_limit_sec=2.000000\n'
        for sequence in range(11):
            assert relay.send_frame(sequence, clock.now(), 0.02, 0.0)
            time.sleep(.05)
        writes = [frame for _, _, frame in transport.snapshot() if frame['linear'] > 0]
        assert transport.started_limit == 2.0 and len(writes) >= 8
    finally:
        relay.stop_and_latch()
        relay.close()


def test_sensor_deadline_shortens_intent_and_reaches_wire_atomically():
    relay, transport, clock = make()
    try:
        assert relay.arm(0.3)
        now = clock.now()
        deadline = now + 0.09
        assert relay.send_frame(0, now, 0.02, 0.0, valid_until=deadline)
        wait_for(lambda: any(frame['linear'] > 0 for _, _, frame in transport.snapshot()))
        time.sleep(0.14)
        moving = [(at, frame) for at, _, frame in transport.snapshot() if frame['linear'] > 0]
        assert all(frame['expires_at'] <= deadline for _, frame in moving)
        assert all(at < deadline for at, _ in moving)
        assert relay.evidence['worker_fault'] == 'setpoint_expired'
    finally:
        relay.stop_and_latch()
        relay.close()


def test_first_motion_write_can_be_awaited_without_extending_its_deadline():
    relay, transport, clock = make()
    try:
        assert relay.arm(0.3)
        now = clock.now()
        assert relay.send_frame(0, now, 0.02, 0.0, valid_until=now + .12)
        assert relay.wait_for_frame_write(0)
        moving = [frame for _, _, frame in transport.snapshot() if frame['linear'] > 0]
        assert moving and moving[0]['expires_at'] <= now + .12
    finally:
        relay.close()


def test_first_motion_write_wait_fails_closed_when_the_writer_stalls():
    entered, release = threading.Event(), threading.Event()
    class StalledMotion(FakeTransport):
        def write_line(self, data, timeout):
            if json.loads(data)['linear'] > 0:
                entered.set()
                release.wait(1.0)
            return FakeTransport.write_line(self, data, timeout)
    relay, _, clock = make(StalledMotion())
    try:
        assert relay.arm(0.3)
        now = clock.now()
        assert relay.send_frame(0, now, 0.02, 0.0, valid_until=now + .08)
        assert entered.wait(.1)
        assert not relay.wait_for_frame_write(0)
        assert not relay.healthy()
    finally:
        release.set()
        relay.close()


def test_control_clock_is_read_under_the_same_lock_as_worker_progress():
    from threaded_relay_adapter import ThreadedRelayAdapter
    class CheckedClock(Clock):
        condition = None
        owner = None
        violations = 0
        def now(self):
            if (self.condition is not None and threading.current_thread().ident == self.owner
                    and self.condition.acquire(False)):
                self.condition.release()
                self.violations += 1
            return _monotonic()
    clock, transport = CheckedClock(), FakeTransport()
    relay = ThreadedRelayAdapter(transport, clock)
    try:
        assert relay.arm(0.3)
        clock.condition, clock.owner = relay._cv, threading.current_thread().ident
        assert relay.healthy()
        assert relay.send_zero_heartbeat()
        assert relay.send_frame(relay.sequence + 1, _monotonic(), 0.02, 0.0)
        assert clock.violations == 0
    finally:
        relay.stop_and_latch()
        relay.close()


def test_recent_zero_heartbeat_does_not_wait_for_another_wire_period():
    entered, release = threading.Event(), threading.Event()
    class BlockAfterPrime(FakeTransport):
        block = False
        def write_line(self, data, timeout):
            if self.block:
                entered.set()
                release.wait(1.0)
            return FakeTransport.write_line(self, data, timeout)
    transport = BlockAfterPrime()
    relay, _, _ = make(transport)
    try:
        assert relay.arm(0.3)
        transport.block = True
        assert relay.send_zero_heartbeat()
        assert relay.evidence['wire_heartbeats'] >= 1
    finally:
        release.set()
        relay.close()


def test_stale_nonzero_intent_requests_zero_and_latches_without_refresh():
    relay, transport, clock = make()
    try:
        assert relay.arm(0.3) and relay.send_zero_heartbeat()
        stamp = clock.now()
        assert relay.send_frame(relay.sequence + 1, stamp, 0.02, 0.0)
        wait_for(lambda: any(frame['linear'] > 0 for _, _, frame in transport.snapshot()))
        wait_for(lambda: not relay.healthy())
        writes = transport.snapshot()
        nonzero = [(at, frame) for at, _, frame in writes if frame['linear'] > 0]
        assert nonzero and all(frame['stamp'] < stamp + 0.18 for _, frame in nonzero)
        first_zero_after = next(i for i, (_, _, frame) in enumerate(writes)
                                if frame['linear'] == 0 and i > 0 and
                                any(prior[2]['linear'] > 0 for prior in writes[:i]))
        assert all(item[2]['linear'] == 0 for item in writes[first_zero_after:])
        assert relay.evidence['zero_write_completed'] is True
        assert relay.evidence['physical_stop_verified'] is False
        assert not relay.send_frame(relay.sequence + 1, clock.now(), 0.02, 0.0)
        assert relay.close()
    finally:
        relay.close()


def test_dead_process_fault_latches_and_does_not_claim_a_zero_write():
    relay, transport, clock = make()
    try:
        assert relay.arm(0.3) and relay.send_zero_heartbeat()
        transport.process.running = False
        wait_for(lambda: not relay.healthy())
        assert not relay.send_frame(relay.sequence + 1, clock.now(), 0.02, 0.0)
        assert relay.close()
        assert relay.evidence['zero_write_completed'] is False
        assert relay.evidence['process_exited'] is True
    finally:
        relay.close()


def test_pipe_failure_requests_zero_and_stays_latched():
    relay, transport, clock = make()
    try:
        assert relay.arm(0.3) and relay.send_zero_heartbeat()
        with transport.lock:
            transport.fail_next_write = True
        wait_for(lambda: not relay.healthy())
        assert relay.close()
        assert relay.evidence['zero_write_completed'] is True
        assert not relay.send_frame(relay.sequence + 1, clock.now(), 0.02, 0.0)
    finally:
        relay.close()


def test_invalid_or_replayed_control_frame_fails_closed():
    relay, transport, clock = make()
    try:
        assert relay.arm(0.3) and relay.send_zero_heartbeat()
        assert not relay.send_frame(relay.sequence + 1, clock.now() - 1.0, 0.02, 0.0)
        wait_for(lambda: not relay.healthy())
        assert not relay.send_frame(relay.sequence + 1, clock.now(), 0.02, 0.0)
        assert not any(frame['linear'] > 0 for _, _, frame in transport.snapshot())
    finally:
        relay.close()


def test_close_before_arm_cannot_start_a_process_later():
    relay, transport, _ = make()
    assert relay.close()
    assert relay.arm(0.3) is None
    assert transport.process is None


def test_first_nonzero_intent_keeps_the_periodic_wire_schedule():
    relay, transport, clock = make()
    try:
        assert relay.arm(0.3) and relay.send_zero_heartbeat()
        assert relay.send_frame(relay.sequence + 1, clock.now(), 0.02, 0.0)
        wait_for(lambda: len([frame for _, _, frame in transport.snapshot() if frame['linear'] > 0]) >= 2)
        times = [at for at, _, frame in transport.snapshot() if frame['linear'] > 0]
        assert times[1] - times[0] < 0.075
    finally:
        relay.close()


def test_bad_ready_closes_the_started_transport():
    relay, transport, _ = make(FakeTransport(ready=b'READY wrong\n'))
    assert relay.arm(0.3) is None
    assert relay.close()
    assert transport.process.poll() is not None
    assert relay.evidence['physical_stop_verified'] is False


def test_health_check_cannot_hang_behind_a_stalled_pipe_writer():
    entered, release = threading.Event(), threading.Event()
    class StalledTransport(FakeTransport):
        block = False
        def write_line(self, data, timeout):
            if self.block:
                entered.set()
                release.wait(2.0)
            return FakeTransport.write_line(self, data, timeout)
    transport = StalledTransport()
    relay, _, _ = make(transport)
    done, result = threading.Event(), []
    def check():
        result.append(relay.healthy())
        done.set()
    checker = threading.Thread(target=check)
    checker.daemon = True
    try:
        assert relay.arm(0.3) and relay.send_zero_heartbeat()
        transport.block = True
        assert entered.wait(1.0)
        time.sleep(0.2)
        checker.start()
        assert done.wait(0.3), 'health check waited on blocked pipe I/O'
        assert result == [False]
    finally:
        release.set()
        if checker.ident is not None:
            checker.join(1.0)
        relay.close()


def test_close_waits_for_bounded_graceful_process_cleanup():
    class SlowClose(FakeTransport):
        def close(self):
            time.sleep(1.85)
            FakeTransport.close(self)
    relay, transport, _ = make(SlowClose())
    try:
        assert relay.arm(0.3) and relay.send_zero_heartbeat()
        assert relay.stop_and_latch()
        assert relay.close()
        assert transport.process.poll() is not None
    finally:
        relay.close()


def test_frozen_clock_causes_worker_exit_without_external_health_poll():
    class FrozenClock(Clock):
        frozen = None
        def now(self):
            return self.frozen if self.frozen is not None else Clock.now(self)
    from threaded_relay_adapter import ThreadedRelayAdapter
    clock, transport = FrozenClock(), FakeTransport()
    relay = ThreadedRelayAdapter(transport, clock)
    try:
        assert relay.arm(0.3) and relay.send_zero_heartbeat()
        clock.frozen = clock.now()
        time.sleep(0.32)
        assert transport.process.poll() is not None
        assert relay.close()
    finally:
        relay.close()


def test_stop_waits_for_inflight_nonzero_then_orders_zero_and_closes():
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    class BlockingNonzero(FakeTransport):
        def write_line(self, data, timeout):
            if json.loads(data)['linear'] > 0:
                entered.set()
                release.wait(2.0)
            return FakeTransport.write_line(self, data, timeout)
    relay, transport, clock = make(BlockingNonzero())
    results = []
    def stop():
        results.append(relay.stop_and_latch())
        stopped.set()
    stopper = threading.Thread(target=stop)
    stopper.daemon = True
    try:
        assert relay.arm(0.3) and relay.send_zero_heartbeat()
        assert relay.send_frame(relay.sequence + 1, clock.now(), 0.02, 0.0)
        assert entered.wait(1.0)
        stopper.start()
        wait_for(lambda: not relay.healthy())
        assert not stopped.is_set()
        release.set()
        assert stopped.wait(1.0) and results == [True]
        assert relay.close()
        writes = transport.snapshot()
        motion = [index for index, item in enumerate(writes) if item[2]['linear'] > 0]
        assert len(motion) == 1
        assert all(item[2]['linear'] == 0 for item in writes[motion[-1] + 1:])
        assert writes[-1][2]['linear'] == 0
        assert len(set(item[1] for item in writes)) == 1
        count = len(writes)
        time.sleep(0.1)
        assert len(transport.snapshot()) == count
    finally:
        release.set()
        if stopper.ident is not None:
            stopper.join(1.0)
        relay.close()
