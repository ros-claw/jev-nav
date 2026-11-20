"""Check lidar and depth independently against one owned forward sample."""

from __future__ import division

import collections
import threading

from relay import _monotonic
from processed_evidence import _number


def pending_sensor_deadline(latest, checked, now, max_age, history=()):
    """Shorten a live proof to the oldest of at most two pending arrivals."""
    if not latest or not checked:
        return None
    generation, previous = latest.get('generation'), checked.get('generation')
    if (generation is None or previous is None or not 1 <= generation - previous <= 2
            or not _number(checked.get('stamp'))):
        return None
    pending = [(latest.get('stamp'), latest.get('received_at'))]
    if generation == previous + 2:
        intermediate = next(((stamp, receipt) for index, stamp, receipt in history
                             if index == previous + 1), None)
        if intermediate is None:
            return None
        pending.insert(0, intermediate)
    deadline, stamp_before, receipt_before = None, checked['stamp'], 0.
    for stamp, received in pending:
        if (not _number(stamp) or not _number(received) or not stamp_before < stamp <= received <= now
                or received < receipt_before):
            return None
        candidate = min(received + .070, stamp + max_age)
        deadline = candidate if deadline is None else min(deadline, candidate)
        stamp_before, receipt_before = stamp, received
    return deadline if now < deadline else None


class SplitSensorValidation(object):
    STREAMS = ('scan', 'depth')
    STOPS = ('new_scan', 'new_depth_unchecked', 'epoch_or_guard', 'transform_changed')
    LATCH = ('front_blocked', 'depth_obstacle', 'epoch_or_guard', 'transform_changed')

    def __init__(self, capture, check, combine):
        self.capture, self.check, self.combine = capture, check, combine
        self._cv = threading.Condition()
        self._job = None
        self._parts = {}
        self._fault = None
        self._job_fault = None
        self._closed = False
        self._counts = collections.Counter()
        self._input_ready = {stream: threading.Event() for stream in self.STREAMS}
        self._threads = []
        for stream in self.STREAMS:
            thread = threading.Thread(target=self._run, args=(stream,), name='limo-proof-' + stream)
            thread.daemon = True
            self._threads.append(thread)
            thread.start()

    def _run(self, stream):
        last_job = last_generation = None
        arrived = self._input_ready[stream]
        try:
            while True:
                with self._cv:
                    if self._closed or self._fault is not None:
                        return
                    job = self._job
                    if job is None:
                        self._cv.wait()
                        continue
                    if _monotonic() >= job['snapshot_at'] + .25:
                        self._cv.wait()
                        continue
                arrived.clear()
                generation, inputs = self.capture(stream, job)
                if job is last_job and generation == last_generation:
                    arrived.wait(.025)
                    continue
                part, issue = self.check(stream, job, inputs)
                # A not-yet-projected depth frame may become available with
                # the same generation. Other results are assessed once.
                last_job = job
                retry = issue in ('depth_preprojection_pending', 'scan_alignment_pending')
                last_generation = None if retry else generation
                with self._cv:
                    self._counts[stream + ':' + (issue or 'ready')] += 1
                    if issue in self.LATCH and self._fault is None:
                        self._fault = issue
                        self._cv.notifyAll()
                    if job is self._job:
                        waiting = retry or issue in ('prepared_scan_advanced', 'prepared_depth_advanced',
                                                     'scan_coverage_pending')
                        if not waiting or stream not in self._parts:
                            self._parts[stream] = (part, issue)
                        if issue in self.STOPS and self._job_fault is None:
                            self._job_fault = issue
                        self._cv.notifyAll()
                    if retry and not self._closed:
                        self._cv.wait(.005)
        except Exception:
            with self._cv:
                self._fault = 'split_worker_failed'
                self._cv.notifyAll()

    def _replace_locked(self, sample, parts):
        self._job, self._parts, self._job_fault = sample, parts, None
        for event in self._input_ready.values():
            event.set()
        self._cv.notifyAll()

    def seed(self, sample, scan, depth):
        """Publish the already checked initial proof without changing expiry."""
        with self._cv:
            self._replace_locked(sample, {'scan': (scan, None), 'depth': (depth, None)})

    def prepare(self, sample, budget=.075):
        deadline = min(_monotonic() + budget, sample['snapshot_at'] + .25)
        with self._cv:
            self._replace_locked(sample, {})
            while True:
                proof, issue = self._read_locked(sample)
                if proof is not None or issue not in (
                        'prepared_scan_advanced', 'prepared_depth_advanced', 'prepared_pending'):
                    return proof, issue
                remaining = deadline - _monotonic()
                if remaining <= 0:
                    # Keep the anchor alive until replaced. Either worker
                    # can still finish during the handoff to the controller.
                    return None, 'split_proof_pending'
                self._cv.wait(remaining)

    def _read_locked(self, sample, nonblocking=False):
        if self._closed or self._fault is not None:
            return None, self._fault or 'split_worker_closed'
        if sample is not self._job:
            return None, 'prepared_pending'
        if self._job_fault is not None:
            return None, self._job_fault
        if any(self._parts.get(name, (None,))[0] is None for name in self.STREAMS):
            return None, 'prepared_pending'
        if nonblocking:
            return self.combine(sample, self._parts['scan'][0], self._parts['depth'][0], nonblocking=True)
        return self.combine(sample, self._parts['scan'][0], self._parts['depth'][0])

    def read(self, sample):
        if not self._cv.acquire(False):
            return None, 'prepared_pending'
        try:
            return self._read_locked(sample, nonblocking=True)
        finally:
            self._cv.release()

    def counts(self):
        with self._cv:
            return dict(self._counts)

    def notify_input(self, stream):
        # ROS callbacks may hold their own lock. Never acquire the
        # coordinator lock here; combining proofs acquires the ROS lock.
        event = self._input_ready.get('depth' if stream == 'info' else stream)
        if event is not None:
            event.set()

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notifyAll()
            for event in self._input_ready.values():
                event.set()
        for thread in self._threads:
            thread.join(.15)
        return all(not thread.is_alive() for thread in self._threads)
