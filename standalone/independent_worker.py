"""Bounded per-sensor workers for diagnostics; no publisher or motion permit.

Every received frame is registered before queueing. The producer owns payloads
until processing returns and must not mutate them. Confirmed obstacles and hard
errors latch; unknown observations revoke their channel until a new result.
Controller/relay fault propagation, pose safety and physical braking remain
separate requirements and are not supplied by this module.
"""
from __future__ import division

import threading
from relay import _monotonic
try:
    import Queue as queue
except ImportError:
    import queue

from processed_evidence import ProcessedEvidence, STRING_TYPES


class IndependentWorkers(object):  # noqa: UP004 - ROS Melodic Python 2
    def __init__(self, ages, max_processing_lag, processors, clock, timeline, queue_size=6):
        if (not isinstance(processors, dict) or set(processors) != set(ages)
                or any(not callable(value) for value in processors.values())
                or not callable(getattr(clock, 'now', None))
                or not callable(getattr(timeline, 'record', None))
                or type(queue_size) is not int or not 1 <= queue_size <= 30):
            raise ValueError('invalid independent workers')
        self.store = ProcessedEvidence(ages, max_processing_lag, max_pending=queue_size + 1)
        self.processors, self.clock, self.timeline = dict(processors), clock, timeline
        self.queues = {name: queue.Queue(maxsize=queue_size) for name in ages}
        self._submit_locks = {name: threading.Lock() for name in ages}
        self._stop = threading.Event()
        self._threads = []
        self._lock = threading.Lock()
        self._started = False
        self.unprocessed_at_close = None

    def start(self):
        with self._lock:
            if self._started or self._stop.is_set():
                raise ValueError('workers are single use')
            self._started = True
            for name in sorted(self.queues):
                thread = threading.Thread(target=self._run, args=(name,), name='nav-' + name)
                thread.daemon = True
                self._threads.append(thread)
                thread.start()

    def submit(self, name, stamp, received, epoch, payload):
        if not self._started or self._stop.is_set() or not isinstance(name, STRING_TYPES) or name not in self.queues:
            self.fail('invalid_submission')
            return False
        with self._submit_locks[name]:
            if self._stop.is_set():
                return False
            sequence = self.store.ingest(name, stamp, received, epoch)
            if sequence is None:
                self._notify_fault()
                return False
            fields = {'clock_domain': 'monotonic', 'stage': name + '_ingress', 'sequence': sequence,
                      'epoch': epoch, 'sensor_acquired_at': stamp, 'sensor_received_at': received}
            if isinstance(payload, dict) and 'callback_at' in payload:
                fields['callback_at'] = payload['callback_at']
            if not self.timeline.record('sensor', received, fields):
                self.fail('timeline_incomplete')
                return False
            try:
                self.queues[name].put_nowait((sequence, stamp, received, epoch, payload))
                return True
            except queue.Full:
                self.fail('backlog_overflow')
                return False

    def _notify_fault(self):
        notify = getattr(self.timeline, 'notify_fault', None)
        if callable(notify):
            notify()

    def fail(self, reason):
        self.store.producer_failed(reason)
        self._notify_fault()

    def _run(self, name):
        try:
            while not self._stop.is_set() or not self.queues[name].empty():
                try:
                    sequence, stamp, received, epoch, payload = self.queues[name].get(timeout=.05)
                except queue.Empty:
                    continue
                started = self.clock.now()
                try:
                    result = self.processors[name](payload, started)
                    ended = self.clock.now()
                    if not isinstance(result, dict):
                        raise ValueError('invalid processor result')
                    clear = result.get('clear')
                    reason = result.get('reason', 'sensor_unavailable')
                    fields = {'clock_domain': 'monotonic', 'stage': name + '_result',
                              'sequence': sequence, 'epoch': epoch, 'sensor_acquired_at': stamp,
                              'sensor_received_at': received, 'perception_start': started,
                              'perception_end': ended, 'processing_sec': ended - started,
                              'queue_delay_sec': started - received, 'valid_until': result.get('valid_until'),
                              'status': 'candidate' if clear is True else 'obstacle' if clear is False else 'unknown',
                              'reason': reason}
                    for key in ('ray_count', 'retained_scan_count', 'invalid_count', 'unknown_count',
                                'sampled_candidate', 'unobserved_space', 'raw_front_count',
                                'raw_valid_count', 'fused_front_count', 'fused_valid_count'):
                        if key in result:
                            fields[key] = result[key]
                    if not self.timeline.record('sensor', ended, fields):
                        self.fail('timeline_incomplete')
                        return
                    if not self.store.complete(name, sequence, clear, result.get('valid_until'), ended,
                                               reason=reason):
                        self._notify_fault()
                        return
                    if clear is False:
                        self._notify_fault()
                        return
                except Exception as exc:
                    self.timeline.record('sensor', self.clock.now(), {'clock_domain': 'monotonic',
                                         'stage': name + '_result', 'reason': 'processor_exception',
                                         'error_class': type(exc).__name__, 'sequence': sequence})
                    self.fail('processor_exception')
                    return
        except BaseException:
            self.fail('worker_failed')

    def read(self, now, epoch):
        result = self.store.read(now, epoch)
        if result['state'] == 'fault':
            self._notify_fault()
        return result

    def read_current(self, epoch):
        result = self.store.read_current(self.clock.now, epoch)
        if result['state'] == 'fault':
            self._notify_fault()
        return result

    def close(self, timeout=1.0):
        self._stop.set()
        deadline = _monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0., deadline - _monotonic()))
        self.unprocessed_at_close = self.store.pending_count()
        self.store.producer_failed('producer_closed')
        return (all(not thread.is_alive() for thread in self._threads)
                and self.unprocessed_at_close == 0)
