"""Experimental processed-frame bookkeeping; it does not authorize motion.

One ordered producer per sensor records every received generation and completes
each exactly once. New arrivals do not invalidate completed work merely because
the generation changed. A consumer receives an explicit deadline no later than
the oldest unprocessed frame's processing budget and the original acquisition
expiry of every completed result. Any confirmed obstacle or hard fault latches.

This component contains no geometry, pose compensation, independent hazard
detector, driver watchdog or ROS I/O. Its 'ready' state is only a bookkeeping
result, never proof of clearance or permission to command a robot.
"""
from __future__ import division

import collections
import math
import re
import threading

try:
    INTEGER_TYPES = (int, long)  # noqa: F821
    STRING_TYPES = (str, unicode)  # noqa: F821
except NameError:
    INTEGER_TYPES = (int,)
    STRING_TYPES = (str,)

_NAME = re.compile(r'^[a-z][a-z0-9_]{0,63}$')


def _number(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, INTEGER_TYPES + (float,))
                and not math.isnan(value) and not math.isinf(value))
    except (OverflowError, TypeError, ValueError):
        return False


class ProcessedEvidence(object):  # noqa: UP004 - ROS Melodic Python 2
    def __init__(self, max_ages, max_processing_lag, max_pending=8):
        if (not isinstance(max_ages, dict) or not max_ages or len(max_ages) > 8
                or any(not isinstance(name, STRING_TYPES) or not _NAME.match(name)
                       or not _number(age) or not 0 < age <= 1.0 for name, age in max_ages.items())
                or not _number(max_processing_lag) or not 0 < max_processing_lag <= .2
                or type(max_pending) is not int or not 0 < max_pending <= 32):
            raise ValueError('invalid sensor budgets')
        self._ages = dict(max_ages)
        self._lag, self._limit = max_processing_lag, max_pending
        self._lock = threading.Lock()
        self._pending = {name: collections.deque() for name in max_ages}
        self._completed = {name: None for name in max_ages}
        self._last_acquired = {name: None for name in max_ages}
        self._last_received = {name: None for name in max_ages}
        self._sequence = {name: 0 for name in max_ages}
        self._epoch, self._last_read, self._fault = None, None, None

    def _fail(self, reason):
        if self._fault is None:
            self._fault = reason

    def ingest(self, stream, acquired_at, received_at, epoch):
        with self._lock:
            if self._fault is not None:
                return None
            if (not isinstance(stream, STRING_TYPES) or stream not in self._ages
                    or not _number(acquired_at) or not _number(received_at)
                    or not 0 <= acquired_at <= received_at or isinstance(epoch, bool)
                    or not isinstance(epoch, INTEGER_TYPES) or epoch < 0):
                self._fail('invalid_ingress')
                return None
            if self._epoch is None:
                self._epoch = epoch
            if epoch != self._epoch:
                self._fail('epoch_changed')
                return None
            if ((self._last_acquired[stream] is not None and acquired_at <= self._last_acquired[stream])
                    or (self._last_received[stream] is not None and received_at < self._last_received[stream])):
                self._fail('sensor_time_regression')
                return None
            if len(self._pending[stream]) >= self._limit:
                self._fail('backlog_overflow')
                return None
            sequence = self._sequence[stream]
            self._sequence[stream] += 1
            self._last_acquired[stream], self._last_received[stream] = acquired_at, received_at
            self._pending[stream].append((sequence, acquired_at, received_at))
            return sequence

    def complete(self, stream, sequence, clear, valid_until, completed_at, reason='obstacle'):
        with self._lock:
            if self._fault is not None:
                return False
            if (not isinstance(stream, STRING_TYPES) or stream not in self._ages or not self._pending[stream]
                    or isinstance(sequence, bool) or not isinstance(sequence, INTEGER_TYPES)
                    or sequence != self._pending[stream][0][0] or clear is not None and type(clear) is not bool
                    or not _number(completed_at)
                    or clear is True and not _number(valid_until)):
                self._fail('invalid_completion')
                return False
            _, acquired, received = self._pending[stream][0]
            if (completed_at < received or clear is True and
                    (valid_until <= acquired or valid_until > acquired + self._ages[stream])):
                self._fail('invalid_completion')
                return False
            self._pending[stream].popleft()
            if clear is None:
                self._completed[stream] = None
                return True
            if not clear:
                self._fail(reason if isinstance(reason, STRING_TYPES) and _NAME.match(reason) else 'obstacle')
                return True
            # Replacing a previous result with an already expired one must not
            # accidentally preserve the older, superficially valid verdict.
            self._completed[stream] = ((sequence, acquired, received, completed_at, valid_until)
                                       if completed_at < valid_until else None)
            return True

    def hazard(self, reason):
        with self._lock:
            self._fail(reason if isinstance(reason, STRING_TYPES) and _NAME.match(reason) else 'hazard')

    def producer_failed(self, reason):
        self.hazard(reason)

    def pending_count(self):
        with self._lock:
            return sum(len(queue) for queue in self._pending.values())

    def read(self, now, epoch):
        with self._lock:
            return self._read_locked(now, epoch)

    def read_current(self, clock, epoch):
        """Capture the consumer time under the same lock as sensor updates."""
        with self._lock:
            try:
                now = clock()
            except Exception:
                now = None
            result = self._read_locked(now, epoch)
            result['checked_at'] = now if _number(now) else None
            return result

    def _read_locked(self, now, epoch):
        if not _number(now) or now < 0 or (self._last_read is not None and now < self._last_read):
            self._fail('clock_regression')
        else:
            self._last_read = now
        if (isinstance(epoch, bool) or not isinstance(epoch, INTEGER_TYPES) or epoch < 0
                or self._epoch is not None and epoch != self._epoch):
            self._fail('epoch_changed')
        if (_number(now) and any(value is not None and value[3] > now
                                 for value in self._completed.values())):
            self._fail('clock_regression')
        if (_number(now) and any(queue and queue[-1][2] > now for queue in self._pending.values())):
            self._fail('clock_regression')
        if self._fault is not None:
            return {'state': 'fault', 'reason': self._fault, 'valid_until': None, 'pending': 0}
        if self._epoch is None or any(value is None for value in self._completed.values()):
            return {'state': 'waiting', 'reason': 'awaiting_processed_frame',
                    'valid_until': None, 'pending': sum(len(q) for q in self._pending.values())}
        # Account for a frame received before read but registered afterwards.
        limits = [value[4] for value in self._completed.values()]
        limits.extend(value[2] + self._lag for value in self._completed.values())
        for name, queue in self._pending.items():
            if queue:
                _, acquired, received = queue[0]
                limits.append(min(received + self._lag, acquired + self._ages[name]))
        deadline = min(limits)
        count = sum(len(q) for q in self._pending.values())
        if not _number(now) or now >= deadline:
            return {'state': 'waiting', 'reason': 'evidence_expired',
                    'valid_until': None, 'pending': count}
        return {'state': 'ready', 'reason': 'processed', 'valid_until': deadline, 'pending': count}
