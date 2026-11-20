#!/usr/bin/env python2
"""Bounded background snapshot producer; it never publishes motor commands.

The source must return newly owned snapshots, and the validator must reject
missing or unsafe evidence. An explicitly classified transient source gap may
be retried, but never refreshes the prior sample's expiry. Other failures
latch and discard the prior snapshot. Consumers must independently recheck
current sensors and enforce absolute expiry before every command.
"""
from __future__ import division

import threading


class DecisionWarmup(object):  # noqa: UP004 - Python 2.7 ROS
    """Refresh observation history while a decision is pending; never command motion."""
    def __init__(self, source):
        if not callable(getattr(source, 'snapshot', None)):
            raise ValueError('invalid observation source')
        self.source = source
        self._stop = threading.Event()
        self._thread = None
        self._fault = False

    def start(self):
        if self._thread is not None or self._stop.is_set():
            raise ValueError('warmup is single-use')
        self._thread = threading.Thread(target=self._run, name='limo-decision-warmup')
        self._thread.daemon = True
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                self.source.snapshot()
                self._stop.wait(.02)
        except BaseException:
            self._fault = True

    def close(self, timeout=.2):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        return (self._thread is None or not self._thread.is_alive()) and not self._fault


class AsyncSafetySampler(object):  # noqa: UP004 - Python 2.7 ROS
    def __init__(self, source, validate, clock, transient_unavailable=None):
        if not callable(getattr(source, 'snapshot', None)) or not callable(validate) or not callable(clock.now):
            raise ValueError('invalid safety sampler')
        if transient_unavailable is not None and not callable(transient_unavailable):
            raise ValueError('invalid transient classifier')
        self.source, self.validate, self.clock = source, validate, clock
        self.transient_unavailable = transient_unavailable
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self._assessment = None
        self._fault = None
        self._closed = False
        self._changed = threading.Event()

    def start(self):
        with self._lock:
            if self._thread is not None or self._closed:
                raise ValueError('sampler is single-use')
            self._thread = threading.Thread(target=self._run, name='limo-safety-sampler')
            self._thread.daemon = True
            self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                validated = (getattr(self.source, 'serialize_processing', False) is True
                             and callable(getattr(self.source, 'validated_snapshot', None)))
                if validated:
                    sample, reason, assessment = self.source.validated_snapshot(
                        self.validate, self.clock.now)
                else:
                    sample = self.source.snapshot()
                if self._stop.is_set():
                    return
                if not isinstance(sample, dict):
                    if (self.transient_unavailable is not None
                            and self.transient_unavailable(self.source) is True):
                        self._stop.wait(.01)
                        continue
                    raise ValueError('sensor_unavailable')
                if not validated:
                    reason, assessment = self.validate(sample, self.clock.now())
                if reason is not None:
                    with self._lock:
                        self._fault = reason if isinstance(reason, str) and reason else 'sensor_unavailable'
                        self._latest = self._assessment = None
                        self._changed.set()
                    return
                if not isinstance(assessment, dict):
                    raise ValueError('sensor_unavailable')
                with self._lock:
                    if self._stop.is_set():
                        return
                    self._latest, self._assessment = sample, assessment
                    self._changed.set()
                self._stop.wait(.01)
        except BaseException:
            with self._lock:
                self._fault = 'sensor_unavailable'
                self._latest = self._assessment = None
                self._changed.set()

    def wait_next(self, previous, timeout):
        """Wake when a new result or fault is available."""
        with self._lock:
            if self._fault is not None or self._closed or self._latest is not previous:
                return
            self._changed.clear()
        self._changed.wait(timeout)

    def poll(self):
        with self._lock:
            if self._fault is not None:
                return {'state': 'fault', 'reason': self._fault, 'sample': None, 'assessment': None}
            if self._closed or self._latest is None:
                return {'state': 'waiting', 'reason': None, 'sample': None, 'assessment': None}
            return {'state': 'ready', 'reason': None, 'sample': self._latest, 'assessment': self._assessment}

    def close(self, timeout=0.2):
        self._stop.set()
        with self._lock:
            self._closed = True
            self._changed.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        return thread is None or not thread.is_alive()
