#!/usr/bin/env python2
"""Bounded asynchronous JSONL timeline; never accepts raw payloads or images."""
from __future__ import division

import collections
import json
import math
import os
import re
import threading

try:
    STRING_TYPES = (str, unicode)  # noqa: F821
    INTEGER_TYPES = (int, long)  # noqa: F821
except NameError:
    STRING_TYPES = (str,)
    INTEGER_TYPES = (int,)

_LABEL = re.compile(r'^[A-Za-z0-9_.:-]{1,128}$')
_EVENTS = frozenset(('perception', 'sensor', 'control', 'decision', 'odom', 'stop', 'run'))
_FIELDS = frozenset(('sensor_acquired_at', 'sensor_received_at', 'perception_start',
                     'perception_end', 'scan_at', 'depth_at', 'odom_at', 'status_at',
                     'color_at', 'snapshot_at', 'valid_until', 'command_sent_at',
                     'command_expires_at', 'decision_id', 'scene_id', 'candidate_id',
                     'scan_generation', 'depth_generation', 'generation_retries',
                     'ray_count', 'retained_scan_count', 'retained_front_ray_count',
                     'accel_used', 'fallback_reason', 'reason', 'error_class', 'stage',
                     'selected_action', 'final_action', 'veto_reason', 'stop_reason',
                     'linear', 'angular', 'x', 'y', 'yaw', 'linear_velocity',
                     'relay_at', 'relay_linear', 'relay_angular', 'relay_reason',
                     'angular_velocity', 'sequence', 'epoch', 'latency_sec',
                     'processing_sec', 'queue_delay_sec', 'remaining_sec', 'dropped',
                     'depth_count', 'front_count', 'status', 'clock_domain',
                     'invalid_count', 'unknown_count', 'unobserved_space', 'sampled_candidate', 'pending',
                     'callback_at', 'sensor_age_sec', 'rear_memory_cells', 'rear_memory_age_sec',
                     'raw_front_count', 'raw_valid_count',
                     'fused_front_count', 'fused_valid_count'))


def _value(value):
    if value is None or type(value) is bool:
        return True
    if isinstance(value, INTEGER_TYPES + (float,)):
        try:
            return not math.isnan(value) and not math.isinf(value)
        except (OverflowError, TypeError, ValueError):
            return False
    if isinstance(value, STRING_TYPES):
        return bool(_LABEL.match(value)) and not value.startswith(('sk-', 'Bearer'))
    return False


class Timeline(object):  # noqa: UP004 - ROS Melodic Python 2
    """Write fixed-schema events on a background thread; report any lost event."""
    def __init__(self, path, run_id, capacity=4096, autostart=True):
        if (not isinstance(run_id, STRING_TYPES) or not _LABEL.match(run_id)
                or type(capacity) is not int or not 0 < capacity <= 100000
                or type(autostart) is not bool):
            raise ValueError('invalid timeline configuration')
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.run_id, self.capacity = run_id, capacity
        self._queue = collections.deque()
        self._cv = threading.Condition(threading.Lock())
        self._closing, self._thread, self._failure = False, None, False
        self.dropped = 0
        if autostart:
            self.start()

    def start(self):
        with self._cv:
            if self._thread is not None or self._closing:
                raise ValueError('timeline already started or closed')
            self._thread = threading.Thread(target=self._run, name='nav-timeline')
            self._thread.daemon = True
            self._thread.start()

    def record(self, event, at, fields):
        if (not isinstance(event, STRING_TYPES) or event not in _EVENTS
                or not _value(at) or isinstance(at, bool)
                or not isinstance(at, INTEGER_TYPES + (float,)) or at < 0
                or not isinstance(fields, dict) or not set(fields).issubset(_FIELDS)
                or not all(_value(value) for value in fields.values())):
            with self._cv:
                self.dropped += 1
            return False
        item = dict(fields)
        item.update({'run_id': self.run_id, 'event': event, 'at': at})
        with self._cv:
            if self._closing or self._failure or len(self._queue) >= self.capacity:
                self.dropped += 1
                return False
            self._queue.append(item)
            self._cv.notify()
            return True

    def _write(self, value):
        wire = (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
        while wire:
            count = os.write(self.fd, wire)
            if count <= 0:
                raise IOError('timeline write failed')
            wire = wire[count:]

    def _run(self):
        try:
            while True:
                with self._cv:
                    while not self._queue and not self._closing:
                        self._cv.wait()
                    if not self._queue and self._closing:
                        return
                    item = self._queue.popleft()
                self._write(item)
        except BaseException:
            with self._cv:
                self._failure = True
                self.dropped += len(self._queue) + 1
                self._queue.clear()
        finally:
            os.close(self.fd)

    def close(self, timeout=2.0):
        with self._cv:
            self._closing = True
            self._cv.notifyAll()
            thread = self._thread
        if thread is None:
            with self._cv:
                if self.fd is not None:
                    os.close(self.fd)
                    self.fd = None
            return False
        thread.join(timeout)
        return not thread.is_alive() and not self._failure and self.dropped == 0
