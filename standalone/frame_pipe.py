"""Bounded private pipe for processed sensor metadata; no payload or motion I/O."""
from __future__ import division

import errno
import fcntl
import json
import math
import os
import re
import stat
import threading

try:
    INTEGER_TYPES = (int, long)  # noqa: F821
    STRING_TYPES = (str, unicode)  # noqa: F821
except NameError:
    INTEGER_TYPES = (int,)
    STRING_TYPES = (str,)
_LABEL = re.compile(r'^[a-z][a-z0-9_]{0,127}$')
_INGRESS = frozenset(('kind', 'stream', 'sequence', 'epoch', 'acquired_at', 'received_at'))
_RESULT = _INGRESS | frozenset(('completed_at', 'valid_until', 'status', 'reason'))
_MAX_FRAME = 1024
_MAX_BUFFER = 16384
_ACCESS_MODE_MASK = getattr(os, 'O_ACCMODE', os.O_WRONLY | os.O_RDWR)


def _number(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, INTEGER_TYPES + (float,))
                and not math.isnan(value) and not math.isinf(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _valid(item, stream):
    if not isinstance(item, dict) or item.get('stream') != stream:
        return False
    kind = item.get('kind')
    if kind not in ('ingress', 'result') or set(item) != (_INGRESS if kind == 'ingress' else _RESULT):
        return False
    if any(isinstance(item[name], bool) or not isinstance(item[name], INTEGER_TYPES) or item[name] < 0
           for name in ('sequence', 'epoch')):
        return False
    a, r = item['acquired_at'], item['received_at']
    if not _number(a) or not _number(r) or not 0 <= a <= r:
        return False
    if kind == 'ingress':
        return True
    completed, deadline, status, reason = (item['completed_at'], item['valid_until'],
                                           item['status'], item['reason'])
    if (not _number(completed) or completed < r or status not in ('candidate', 'unknown', 'obstacle')
            or not isinstance(reason, STRING_TYPES) or not _LABEL.match(reason)):
        return False
    obstacle_reasons = ('front_blocked', 'forward_obstacle') if stream == 'scan' else ('depth_obstacle',)
    candidate_reason = 'sampled_forward_corridor' if stream == 'scan' else 'sampled_depth_corridor'
    if ((status == 'candidate' and reason != candidate_reason)
            or (status == 'obstacle' and reason not in obstacle_reasons)
            or (status == 'unknown' and reason in obstacle_reasons)):
        return False
    if status == 'candidate':
        age = .3 if stream == 'scan' else .15
        return _number(deadline) and completed < deadline <= a + age
    return deadline is None


def _prepare(fd, direction):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    if not stat.S_ISFIFO(os.fstat(fd).st_mode) or flags & _ACCESS_MODE_MASK != direction:
        raise ValueError('expected a pipe end')
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


class FrameWriter(object):  # noqa: UP004 - ROS Melodic Python 2
    def __init__(self, fd, stream):
        if stream not in ('scan', 'depth'):
            raise ValueError('invalid stream')
        _prepare(fd, os.O_WRONLY)
        self.fd, self.stream = fd, stream
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            self._close_locked()

    def _close_locked(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def send(self, item):
        with self._lock:
            if self.fd is None:
                return False
            try:
                if not _valid(item, self.stream):
                    raise ValueError('invalid event')
                wire = (json.dumps(item, allow_nan=False, separators=(',', ':')) + '\n').encode('ascii')
                if len(wire) > _MAX_FRAME or os.write(self.fd, wire) != len(wire):
                    raise ValueError('oversize or partial event')
                return True
            except (OSError, IOError, TypeError, ValueError, OverflowError):
                self._close_locked()
                return False


class FrameReader(object):  # noqa: UP004 - ROS Melodic Python 2
    def __init__(self, fd, stream):
        if stream not in ('scan', 'depth'):
            raise ValueError('invalid stream')
        _prepare(fd, os.O_RDONLY)
        self.fd, self.stream, self.buffer, self.fault = fd, stream, b'', None

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def poll(self):
        if self.fault is not None:
            return [], self.fault
        if self.fd is None:
            return [], 'reader_closed'
        rows = []
        try:
            eof = False
            while True:
                try:
                    data = os.read(self.fd, 4096)
                except (OSError, IOError) as exc:
                    if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        break
                    raise
                if not data:
                    eof = True
                    break
                self.buffer += data
                if len(self.buffer) > _MAX_BUFFER:
                    raise ValueError('backlog')
            while b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                if len(line) + 1 > _MAX_FRAME:
                    raise ValueError('oversize')
                item = json.loads(line.decode('ascii'), object_pairs_hook=_pairs)
                if not _valid(item, self.stream):
                    raise ValueError('invalid')
                rows.append(item)
            if len(self.buffer) >= _MAX_FRAME:
                raise ValueError('oversize')
            if eof:
                if self.buffer:
                    raise ValueError('incomplete event at EOF')
                self.fault = 'producer_eof'
                return rows, self.fault
            return rows, None
        except (OSError, IOError, TypeError, ValueError, UnicodeError, OverflowError):
            self.fault = 'invalid_frame'
            return rows, self.fault


class PipeTimeline(object):  # noqa: UP004 - ROS Melodic Python 2
    """Tee only a worker's bounded sensor metadata to a private pipe."""
    def __init__(self, timeline, writer, stream):
        self.timeline, self.writer, self.stream = timeline, writer, stream

    @property
    def dropped(self):
        return self.timeline.dropped

    def record(self, event, at, fields):
        if not self.timeline.record(event, at, fields):
            self.writer.close()
            return False
        stage = fields.get('stage') if isinstance(fields, dict) else None
        if event != 'sensor' or stage not in (self.stream + '_ingress', self.stream + '_result'):
            return True
        try:
            item = {'kind': 'ingress' if stage.endswith('_ingress') else 'result',
                    'stream': self.stream, 'sequence': fields['sequence'], 'epoch': fields['epoch'],
                    'acquired_at': fields['sensor_acquired_at'], 'received_at': fields['sensor_received_at']}
            if item['kind'] == 'result':
                status, deadline = fields['status'], fields['valid_until']
                reason = fields['reason']
                if status == 'candidate' and _number(deadline) and deadline <= at:
                    status, deadline, reason = 'unknown', None, 'processed_expired'
                if status != 'candidate':
                    deadline = None
                item.update({'completed_at': fields['perception_end'], 'valid_until': deadline,
                             'status': status, 'reason': reason})
            return self.writer.send(item)
        except (KeyError, TypeError, ValueError):
            self.writer.close()
            return False

    def close(self, timeout=2.0):
        self.writer.close()
        return self.timeline.close(timeout)

    def notify_fault(self):
        # EOF is observed by the parent independently of the control poll.
        self.writer.close()
