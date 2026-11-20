#!/usr/bin/env python2
"""Bounded, fail-closed perception-verdict pipe; no ROS or motion I/O.

Create an anonymous pipe and give its read and write ends exclusively to a
VerdictReader and VerdictWriter in the same CLOCK_MONOTONIC domain. Close every
other copy of the write end, including inherited child descriptors, so EOF can
indicate producer loss. Each object owns and closes its descriptor.

The producer supplies ``stamp`` as the earliest acquisition time of all sensor
evidence used for the verdict, not the processing-completion time. Its deadline
must be after that stamp, no more than 0.15 s later, and still in the future.
Sequence begins at zero and increases by exactly one; the expected epoch is
fixed for the session. A reject, expired frame, clock error, malformed input,
backpressure or peer loss ends the session; recovering requires a new pipe.

``poll()`` is nonblocking and returns a copy of a status dictionary. Only
``status == 'pass'`` authorizes use of this verdict, and it must be checked
again at the point of use. ``waiting`` never authorizes use, including when an
incomplete later frame exists. Pipe receipt cannot prove sensing, freshness of
fabricated timestamps, physical clearance, scheduling bounds or motor stopping.
The producer may fail immediately after a poll; an independently enforced
watchdog and close/EOF discipline are required for any future integration.
"""
from __future__ import division

import ctypes
import errno
import fcntl
import json
import math
import os
import stat
import threading
import time

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - Python 2.7
except NameError:
    INTEGER_TYPES = (int,)
NUMBER_TYPES = INTEGER_TYPES + (float,)
_MAX_INTEGER = 9223372036854775807
MAX_FRAME_BYTES = 512  # including newline; no larger than POSIX PIPE_BUF
MAX_BACKLOG_BYTES = 4096
MAX_VALIDITY_SEC = 0.15
_ACCESS_MODE_MASK = getattr(os, 'O_ACCMODE', os.O_WRONLY | os.O_RDWR)
_FIELDS = frozenset(('sequence', 'stamp', 'epoch', 'result', 'valid_until'))


class _Timespec(ctypes.Structure):
    _fields_ = [('seconds', ctypes.c_long), ('nanoseconds', ctypes.c_long)]


def monotonic():
    """Read Linux CLOCK_MONOTONIC on Python 2.7 or Python 3."""
    if hasattr(time, 'monotonic'):
        return time.monotonic()
    timestamp = _Timespec()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.clock_gettime(1, ctypes.byref(timestamp)) != 0:
        raise OSError(ctypes.get_errno(), 'clock_gettime failed')
    return timestamp.seconds + timestamp.nanoseconds / 1000000000.0


def _integer(value):
    return (not isinstance(value, bool) and isinstance(value, INTEGER_TYPES)
            and 0 <= value <= _MAX_INTEGER)


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _pipe(fd, direction):
    if isinstance(fd, bool) or not isinstance(fd, INTEGER_TYPES) or fd < 0:
        raise ValueError('invalid pipe descriptor')
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        if not stat.S_ISFIFO(os.fstat(fd).st_mode) or flags & _ACCESS_MODE_MASK != direction:
            raise ValueError('descriptor is not the requested pipe end')
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    except (OSError, IOError) as error:
        raise ValueError('pipe descriptor unavailable: %s' % type(error).__name__)


def _close(fd):
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError('nonfinite JSON number')


def _frame_valid(item, epoch, sequence, previous_stamp, now):
    if not isinstance(item, dict) or set(item) != _FIELDS:
        return False
    stamp, deadline = item['stamp'], item['valid_until']
    return (_integer(item['sequence']) and item['sequence'] == sequence
            and _integer(item['epoch']) and item['epoch'] == epoch
            and item['result'] in ('pass', 'reject')
            and _finite(stamp) and _finite(deadline) and _finite(now)
            and 0 <= stamp <= now < deadline
            and 0 < deadline - stamp <= MAX_VALIDITY_SEC
            and (previous_stamp is None or stamp > previous_stamp))


class VerdictWriter(object):  # noqa: UP004 - Python 2.7
    def __init__(self, fd, epoch, clock=monotonic):
        if not _integer(epoch) or not callable(clock):
            raise ValueError('invalid writer configuration')
        _pipe(fd, os.O_WRONLY)
        self.fd, self.epoch, self.clock = fd, epoch, clock
        self.lock = threading.Lock()
        self.sequence = 0
        self.last_stamp = None
        self.last_now = None
        self.closed = False
        self.rejected = False

    def _close_locked(self):
        if not self.closed:
            self.closed = True
            _close(self.fd)
            self.fd = None

    def close(self):
        with self.lock:
            self._close_locked()

    def send(self, sequence, stamp, epoch, result, valid_until):
        """Write one atomic verdict; any invalid input or pipe failure closes."""
        with self.lock:
            if self.closed:
                return False
            try:
                now = self.clock()
                item = {'sequence': sequence, 'stamp': stamp, 'epoch': epoch,
                        'result': result, 'valid_until': valid_until}
                if (self.rejected or not _finite(now) or now < 0
                        or (self.last_now is not None and now < self.last_now)
                        or not _frame_valid(item, self.epoch, self.sequence, self.last_stamp, now)):
                    raise ValueError('invalid verdict')
                wire = (json.dumps(item, allow_nan=False, separators=(',', ':')) + '\n').encode('ascii')
                if len(wire) > MAX_FRAME_BYTES or os.write(self.fd, wire) != len(wire):
                    raise ValueError('short or oversized write')
                self.sequence += 1
                self.last_stamp, self.last_now = stamp, now
                self.rejected = result == 'reject'
                return True
            except (ArithmeticError, OSError, IOError, TypeError, ValueError, OverflowError):
                self._close_locked()
                return False


class VerdictReader(object):  # noqa: UP004 - Python 2.7
    def __init__(self, fd, epoch, clock=monotonic, startup_timeout_sec=2.0):
        if (not _integer(epoch) or not callable(clock) or not _finite(startup_timeout_sec)
                or not 0 < startup_timeout_sec <= 10):
            raise ValueError('invalid reader configuration')
        now = clock()
        if not _finite(now) or now < 0:
            raise ValueError('invalid reader clock')
        _pipe(fd, os.O_RDONLY)
        self.fd, self.epoch, self.clock = fd, epoch, clock
        self.lock = threading.Lock()
        self.started_at = self.last_now = now
        self.startup_timeout = startup_timeout_sec
        self.buffer = b''
        self.sequence = 0
        self.last_stamp = None
        self.closed = False
        self.current = {'status': 'waiting', 'latched': False, 'epoch': epoch,
                        'sequence': None, 'stamp': None, 'valid_until': None, 'reason': 'startup'}

    def _terminal(self, status, reason, item=None):
        if not self.current['latched']:
            self.current = {'status': status, 'latched': True, 'epoch': self.epoch,
                            'sequence': item['sequence'] if item is not None else None,
                            'stamp': item['stamp'] if item is not None else None,
                            'valid_until': item['valid_until'] if item is not None else None,
                            'reason': reason}
        if not self.closed:
            self.closed = True
            _close(self.fd)
            self.fd = None

    def close(self):
        with self.lock:
            self._terminal('fault', 'consumer_closed')

    def poll(self):
        """Drain a bounded backlog; never return an older pass over new data."""
        with self.lock:
            if self.current['latched']:
                return dict(self.current)
            try:
                now = self.clock()
                if not _finite(now) or now < self.last_now:
                    self._terminal('fault', 'clock_invalid')
                    return dict(self.current)
                self.last_now = now
                if self.sequence == 0 and now - self.started_at >= self.startup_timeout:
                    self._terminal('fault', 'startup_timeout')
                    return dict(self.current)
                eof = False
                while True:
                    try:
                        chunk = os.read(self.fd, min(1024, MAX_BACKLOG_BYTES + 1 - len(self.buffer)))
                    except (OSError, IOError) as error:
                        if error.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                            break
                        raise
                    if not chunk:
                        eof = True
                        break
                    self.buffer += chunk
                    if len(self.buffer) > MAX_BACKLOG_BYTES:
                        self._terminal('fault', 'backlog')
                        return dict(self.current)
                rejected = None
                while b'\n' in self.buffer:
                    line, self.buffer = self.buffer.split(b'\n', 1)
                    if len(line) + 1 > MAX_FRAME_BYTES:
                        raise ValueError('frame too large')
                    item = json.loads(line.decode('ascii'), object_pairs_hook=_unique_pairs,
                                      parse_constant=_invalid_constant)
                    if not _frame_valid(item, self.epoch, self.sequence, self.last_stamp, now):
                        raise ValueError('invalid verdict')
                    self.sequence += 1
                    self.last_stamp = item['stamp']
                    if item['result'] == 'reject':
                        rejected = item
                    elif rejected is None:
                        self.current = {'status': 'pass', 'latched': False, 'epoch': self.epoch,
                                        'sequence': item['sequence'], 'stamp': item['stamp'],
                                        'valid_until': item['valid_until'], 'reason': None}
                if eof and self.buffer:
                    self._terminal('fault', 'truncated')
                elif len(self.buffer) >= MAX_FRAME_BYTES:
                    self._terminal('fault', 'frame_too_large')
                elif rejected is not None:
                    self._terminal('reject', 'perception_rejected', rejected)
                elif eof:
                    self._terminal('fault', 'producer_closed')
                elif self.current['status'] == 'pass' and now >= self.current['valid_until']:
                    self._terminal('fault', 'expired')
                elif self.buffer:
                    return {'status': 'waiting', 'latched': False, 'epoch': self.epoch,
                            'sequence': None, 'stamp': None, 'valid_until': None, 'reason': 'partial'}
            except (ArithmeticError, OSError, IOError, TypeError, ValueError, OverflowError):
                self._terminal('fault', 'protocol_or_io_error')
            return dict(self.current)
