#!/usr/bin/env python2
"""Single-writer relay facade with an independently refreshed velocity intent.

This module starts no process until ``arm``. A single worker thread owns
the existing RelayAdapter and every pipe operation. Before motion it sends zero
heartbeats; an accepted nonzero intent expires after 0.16 s without refresh.
Expiry or any fault requests zero, closes the relay and permanently ends this
facade. The child relay retains its separately selected first-motion limit and parent-loss
handling. A successful submission or pipe write is not driver acknowledgment,
clearance, braking or proof of a physical stop.

``sequence`` and ``last_stamp`` describe control-thread submissions, not wire
heartbeats: the worker separately allocates the monotonically increasing wire
sequence. ``transport.process`` remains available to a caller monitor. The
clock must use the same local CLOCK_MONOTONIC domain as the child relay.
"""
from __future__ import division

import threading

import forward_trial as checks
from live_trial_adapter import RelayAdapter, relay_ready_line
from relay import _monotonic
from split_sensor_validation import SensorWakeup

_PERIOD_SEC = 0.05
_INTENT_TTL_SEC = 0.16
_START_TIMEOUT_SEC = 4.0
_STOP_TIMEOUT_SEC = 3.0
_PROGRESS_TIMEOUT_SEC = 0.18
_MOTION_WRITE_MARGIN_SEC = 0.01


class ThreadedRelayAdapter(object):  # noqa: UP004 - ROS Melodic Python 2
    """Expose the LiveTrial relay interface with a bounded worker lifetime."""
    simulation_only = False

    def __init__(self, transport, clock, motion_limit_sec=0.3):
        if not callable(getattr(clock, 'now', None)) or not callable(getattr(clock, 'sleep', None)):
            raise ValueError('invalid monotonic clock')
        relay_ready_line(motion_limit_sec)
        self.transport, self.clock = transport, clock
        self.motion_limit_sec = motion_limit_sec
        self._cv = threading.Condition(threading.Lock())
        self._writer_wakeup = SensorWakeup()
        self._thread = None
        self._owner = None
        self._started = False
        self._startup_done = False
        self._contract = None
        self._armed = False
        self._stopping = False
        self._stop_complete = False
        self._exited = False
        self._fault = None
        self._zero_count = 0
        self._last_zero_stamp = None
        self._last_progress = None
        self._last_wall_progress = None
        self._sequence = -1
        self._last_stamp = None
        self._intent = None
        self._intent_version = 0
        self._written_version = 0
        self._had_intent = False
        self._handoff_requested = False
        self._handed_adapter = None
        self._evidence = {'zero_write_completed': False, 'stop_line': None,
                          'process_exited': False, 'physical_stop_verified': False,
                          'worker_fault': None, 'wire_heartbeats': 0, 'wire_nonzero': 0}

    @property
    def armed(self):
        with self._cv:
            return self._armed and not self._stopping

    @property
    def sequence(self):
        with self._cv:
            return self._sequence

    @property
    def last_stamp(self):
        with self._cv:
            return self._last_stamp

    @property
    def evidence(self):
        with self._cv:
            return dict(self._evidence)

    def _stop_locked(self, reason):
        if not self._stopping:
            self._stopping = True
            self._intent = None
            self._fault = reason
            self._evidence['worker_fault'] = reason
        self._cv.notifyAll()
        self._writer_wakeup.set()

    def _usable_locked(self, now):
        try:
            wall_now = _monotonic()
        except Exception:  # noqa: BLE001
            wall_now = None
        if (not self._armed or self._stopping or self._thread is None or not self._thread.is_alive()
                or not checks._finite(now) or not checks._finite(self._last_progress)
                or not 0 <= now - self._last_progress < _PROGRESS_TIMEOUT_SEC
                or not checks._finite(wall_now) or not checks._finite(self._last_wall_progress)
                or not 0 <= wall_now - self._last_wall_progress < _PROGRESS_TIMEOUT_SEC):
            self._stop_locked('worker_unavailable')
            return False
        if self._intent is not None and now >= self._intent['expires_at']:
            self._stop_locked('setpoint_expired')
            return False
        return True

    def _wait_locked(self, predicate, seconds):
        deadline = _monotonic() + seconds
        while not predicate():
            remaining = deadline - _monotonic()
            if remaining <= 0:
                return False
            self._cv.wait(remaining)
        return True

    def arm(self, max_motion_sec):
        with self._cv:
            if self._started or self._stopping or max_motion_sec != self.motion_limit_sec:
                return None
            self._started = True
            self._owner = threading.current_thread().ident
            self._thread = threading.Thread(name='threaded-relay-writer', target=self._run)
            self._thread.daemon = True
            self._thread.start()
            if not self._wait_locked(lambda: self._startup_done, _START_TIMEOUT_SEC):
                self._stop_locked('startup_timeout')
                return None
            return dict(self._contract) if self._contract is not None and not self._stopping else None

    def healthy(self):
        with self._cv:
            try:
                now = self.clock.now()
            except Exception:  # noqa: BLE001 - an invalid clock must fail closed
                now = None
            return self._usable_locked(now)

    def send_zero_heartbeat(self):
        with self._cv:
            try:
                now = self.clock.now()
            except Exception:  # noqa: BLE001
                now = None
            if (threading.current_thread().ident != self._owner or self._had_intent
                    or not self._usable_locked(now)):
                self._stop_locked('invalid_zero_request')
                return False
            # The worker continuously sends zero. Reuse a recently completed
            # zero instead of waiting another 50 ms for each caller check.
            recent = (checks._finite(self._last_zero_stamp)
                      and 0 <= now - self._last_zero_stamp <= 0.075)
            if not recent:
                ticket = self._zero_count + 1
                if not self._wait_locked(lambda: self._zero_count >= ticket or self._stopping, 0.18):
                    self._stop_locked('zero_heartbeat_timeout')
                    return False
                if self._stopping or self._zero_count < ticket:
                    return False
            self._sequence += 1
            self._last_stamp = self._last_zero_stamp
            return True

    def send_frame(self, sequence, stamp, linear, angular, valid_until=None):
        if valid_until is None and checks._finite(stamp):
            valid_until = stamp + _INTENT_TTL_SEC
        with self._cv:
            try:
                now = self.clock.now()
            except Exception:  # noqa: BLE001
                now = None
            valid = (threading.current_thread().ident == self._owner and self._usable_locked(now)
                     and not isinstance(sequence, bool) and isinstance(sequence, checks.INTEGER_TYPES)
                     and sequence == self._sequence + 1 and checks._finite(stamp)
                     and (self._last_stamp is None or stamp > self._last_stamp)
                     and 0 <= now - stamp < _INTENT_TTL_SEC
                     and checks._finite(valid_until) and now < valid_until
                     and checks._finite(linear) and 0 < linear <= checks.MAX_SPEED_MPS
                     and checks._finite(angular) and angular == 0.0)
            if not valid:
                self._stop_locked('invalid_setpoint')
                return False
            self._sequence, self._last_stamp = sequence, stamp
            self._had_intent = True
            self._intent = {'linear': linear, 'angular': angular,
                            'expires_at': min(stamp + _INTENT_TTL_SEC, valid_until)}
            self._intent_version += 1
            self._cv.notifyAll()
            self._writer_wakeup.set()
            return True

    def wait_for_frame_write(self, sequence):
        """Wait for this submission's pipe write, never beyond its original expiry."""
        with self._cv:
            if (threading.current_thread().ident != self._owner or sequence != self._sequence
                    or self._intent is None or self._stopping):
                return False
            version, expires = self._intent_version, self._intent['expires_at']
            remaining = expires - self.clock.now()
            if (remaining <= 0 or not self._wait_locked(
                    lambda: self._written_version >= version or self._stopping, min(remaining, .1))
                    or self._stopping or self.clock.now() >= expires):
                self._stop_locked('wire_write_timeout')
                return False
            return True

    def handoff_before_motion(self):
        """Transfer the live pipe after the heartbeat writer has stopped using it."""
        with self._cv:
            caller = threading.current_thread().ident
            if (caller != self._owner or self._had_intent or self._handoff_requested
                    or not self._usable_locked(self.clock.now())):
                return None
            self._handoff_requested = True
            self._writer_wakeup.set()
            if not self._wait_locked(lambda: self._handed_adapter is not None or self._stopping, .08):
                self._stop_locked('writer_handoff_timeout')
            adapter = self._handed_adapter
            if adapter is not None:
                adapter.owner = caller
            return adapter

    def stop_and_latch(self):
        with self._cv:
            if self._handed_adapter is not None:
                return False
            self._stop_locked('requested_stop')
            if self._thread is None:
                return False
            self._wait_locked(lambda: self._stop_complete or self._exited, _STOP_TIMEOUT_SEC)
            return self._evidence['zero_write_completed'] is True

    def close(self):
        with self._cv:
            if self._handed_adapter is not None:
                return False
            if self._thread is None:
                self._stop_locked('closed')
                self._writer_wakeup.close()
                return True
            self._stop_locked('closed')
            thread = self._thread
        if thread is not threading.current_thread():
            thread.join(_STOP_TIMEOUT_SEC)
        with self._cv:
            if not thread.is_alive():
                self._writer_wakeup.close()
            return not thread.is_alive() and self._evidence['process_exited'] is True

    def _run(self):
        adapter = RelayAdapter(self.transport, self.clock, motion_limit_sec=self.motion_limit_sec)
        try:
            contract = adapter.arm(self.motion_limit_sec)
            with self._cv:
                can_prime = (isinstance(contract, dict)
                             and contract.get('ready_line') == relay_ready_line(self.motion_limit_sec)
                             and contract.get('relative_motion_limit') is True and not self._stopping)
                if not can_prime:
                    self._stop_locked('relay_startup_failed')
            # A broken pipe implementation must not hold up health checks or
            # stop requests by owning the facade's condition lock.
            primed = adapter.send_zero_heartbeat() if can_prime else False
            now = self.clock.now()
            with self._cv:
                if can_prime and not self._stopping:
                    if not primed:
                        self._stop_locked('zero_write_failed')
                    elif not checks._finite(now) or now < 0:
                        self._stop_locked('clock_invalid')
                    else:
                        self._contract = contract
                        self._armed = True
                        self._zero_count = 1
                        self._last_zero_stamp = adapter.last_stamp
                        self._last_progress = now
                        self._last_wall_progress = _monotonic()
                        self._evidence['wire_heartbeats'] = 1
                self._startup_done = True
                self._cv.notifyAll()
            next_deadline = self.clock.now() + _PERIOD_SEC
            last_clock = self.clock.now()
            sent_version = 0
            while True:
                self._writer_wakeup.clear()
                delay = None
                with self._cv:
                    now = self.clock.now()
                    if not checks._finite(now) or now < last_clock:
                        self._stop_locked('clock_invalid')
                    else:
                        last_clock = now
                    wall_now = _monotonic()
                    if (not checks._finite(wall_now) or not checks._finite(self._last_wall_progress)
                            or not 0 <= wall_now - self._last_wall_progress < _PROGRESS_TIMEOUT_SEC):
                        self._stop_locked('worker_stalled')
                    if self._stopping:
                        break
                    if self._handoff_requested:
                        self._writer_wakeup.close()
                        self._armed = False
                        self._handed_adapter = adapter
                        self._cv.notifyAll()
                        break
                    if self._intent is not None and now >= self._intent['expires_at']:
                        self._stop_locked('setpoint_expired')
                        break
                    changed = self._intent is not None and sent_version != self._intent_version
                    if not changed and now < next_deadline:
                        delay = next_deadline - now
                        if self._intent is not None:
                            delay = min(delay, self._intent['expires_at'] - now)
                    elif changed:
                        next_deadline = min(next_deadline, now)
                    # Motion writes are capped at 10 ms; the child relay also
                    # independently enforces the same absolute deadline.
                    if delay is None and self._intent is not None and self._intent['expires_at'] - now <= _MOTION_WRITE_MARGIN_SEC:
                        next_deadline = self._intent['expires_at']
                        delay = next_deadline - now
                    version = self._intent_version
                    intent = dict(self._intent) if self._intent is not None else None
                if delay is not None:
                    self._writer_wakeup.wait(delay)
                    continue
                if not adapter.healthy():
                    with self._cv:
                        self._stop_locked('relay_unhealthy')
                    break
                with self._cv:
                    now = self.clock.now()
                    if not checks._finite(now) or now < last_clock:
                        self._stop_locked('clock_invalid')
                    if self._stopping:
                        break
                    if version != self._intent_version:
                        continue
                    if intent is not None and intent['expires_at'] - now <= _MOTION_WRITE_MARGIN_SEC:
                        next_deadline = intent['expires_at']
                        continue
                if intent is None:
                    written = adapter.send_zero_heartbeat()
                else:
                    written = adapter.send_frame(adapter.sequence + 1, now, intent['linear'], intent['angular'],
                                                 intent['expires_at'])
                with self._cv:
                    if not written:
                        self._stop_locked('relay_write_failed')
                        break
                    self._last_progress = self.clock.now()
                    self._last_wall_progress = _monotonic()
                    if not checks._finite(self._last_progress) or self._last_progress < now:
                        self._stop_locked('clock_invalid')
                        break
                    self._evidence['wire_heartbeats'] += 1
                    if intent is None:
                        self._zero_count += 1
                        self._last_zero_stamp = adapter.last_stamp
                    else:
                        self._evidence['wire_nonzero'] += 1
                        sent_version = version
                        self._written_version = version
                    self._cv.notifyAll()
                    next_deadline = max(next_deadline + _PERIOD_SEC, self._last_progress)
        except BaseException:  # worker faults must discard the command and request zero
            with self._cv:
                self._stop_locked('worker_exception')
                self._startup_done = True
                self._cv.notifyAll()
        finally:
            if self._handed_adapter is None:
                self._finish_worker(adapter)

    def _finish_worker(self, adapter):
        prior_stop = adapter.evidence.get('stop_line')
        try:
            if adapter.armed:
                adapter.stop_and_latch()
        except BaseException:
            pass
        with self._cv:
            self._armed = False
            self._evidence.update(adapter.evidence)
            if prior_stop is not None and self._evidence.get('stop_line') is None:
                self._evidence['stop_line'] = prior_stop
            self._stop_complete = True
            self._cv.notifyAll()
        try:
            closed = adapter.close()
        except BaseException:
            closed = False
        with self._cv:
            self._evidence['process_exited'] = closed is True
            self._exited = True
            self._startup_done = True
            self._cv.notifyAll()
