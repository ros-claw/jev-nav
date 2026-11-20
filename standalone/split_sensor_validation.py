"""Check lidar and depth independently against one owned forward sample."""

from __future__ import division

import collections
import threading

from relay import _monotonic
from processed_evidence import _number

PENDING_PROOF = ('prepared_pending', 'prepared_scan_advanced', 'prepared_depth_advanced',
                 'prepared_state_busy', 'prepared_scan_expired', 'prepared_depth_expired',
                 'prepared_snapshot_expired', 'prepared_state_expired', 'prepared_evidence_expired')

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


class _Anchor(object):
    def __init__(self, sample, parts=None):
        self.sample = sample
        self.expires = sample['snapshot_at'] + .25
        self.parts = parts or {}
        self.generations = {}
        self.issue = None


class SplitSensorValidation(object):
    STREAMS = ('scan', 'depth')
    STOPS = ('new_scan', 'new_depth_unchecked', 'epoch_or_guard', 'transform_changed')
    LATCH = ('front_blocked', 'depth_obstacle', 'epoch_or_guard', 'transform_changed')

    def __init__(self, capture, check, combine):
        self.capture, self.check, self.combine = capture, check, combine
        self._cv = threading.Condition()
        self._active = self._staged = None
        self._fault = None
        self._closed = False
        self._counts = collections.Counter()
        self._reads = collections.Counter()
        self._timeline = collections.deque(maxlen=192)
        self._changed = threading.Event()
        self.revision = 0
        self._input_ready = {stream: threading.Event() for stream in self.STREAMS}
        self._threads = []
        for name in self.STREAMS:
            thread = threading.Thread(target=self._run, args=(name,), name='limo-proof-' + name)
            thread.daemon = True
            self._threads.append(thread)
            thread.start()

    def _run(self, stream):
        arrived = self._input_ready[stream]
        try:
            while True:
                arrived.clear()
                with self._cv:
                    if self._closed or self._fault is not None:
                        return
                    now = _monotonic()
                    anchors = [anchor for anchor in (self._active, self._staged)
                               if anchor is not None and anchor.issue is None and now < anchor.expires]
                    if not anchors:
                        self._cv.wait()
                        continue
                retry = False
                for anchor in anchors:
                    started = _monotonic()
                    generation, inputs = self.capture(stream, anchor.sample)
                    if generation == anchor.generations.get(stream):
                        continue
                    part, issue = self.check(stream, anchor.sample, inputs)
                    completed = _monotonic()
                    unprocessed = issue in ('depth_preprojection_pending', 'depth_alignment_pending',
                                            'scan_alignment_pending',
                                            'prepared_scan_advanced', 'prepared_depth_advanced')
                    retry = retry or unprocessed
                    anchor.generations[stream] = None if unprocessed else generation
                    with self._cv:
                        self._counts[stream + ':' + (issue or 'ready')] += 1
                        event = {'stream': stream, 'sample_at': anchor.sample['snapshot_at'],
                                 'started_at': started, 'completed_at': completed, 'issue': issue}
                        if isinstance(part, dict):
                            entry = part['entry']
                            event.update({'frame_at': entry['stamp'], 'received_at': entry.get('received_at'),
                                          'generation': entry.get('generation'), 'valid_until': part['expires']})
                        self._timeline.append(event)
                        if issue in self.LATCH and self._fault is None:
                            self._fault = issue
                        if anchor in (self._active, self._staged):
                            if not (unprocessed or issue == 'scan_coverage_pending') or stream not in anchor.parts:
                                anchor.parts[stream] = part
                            if issue in self.STOPS and anchor.issue is None:
                                anchor.issue = issue
                        self._changed_locked()
                arrived.wait(.005 if retry else .025)
        except Exception:
            with self._cv:
                if self._fault is None:
                    self._fault = 'split_worker_failed'
                self._changed_locked()

    def _changed_locked(self):
        self.revision += 1
        self._changed.set()
        self._cv.notifyAll()

    def _wake_locked(self):
        for event in self._input_ready.values():
            event.set()
        self._changed_locked()

    def seed(self, sample, scan, depth):
        """Publish the already checked initial proof without changing expiry."""
        with self._cv:
            self._active = _Anchor(sample, {'scan': scan, 'depth': depth})
            self._wake_locked()

    def prepare(self, sample, budget=.075):
        deadline = min(_monotonic() + budget, sample['snapshot_at'] + .25)
        with self._cv:
            if self._closed or self._fault is not None:
                return None, self._fault or 'split_worker_closed'
            if self._staged is not None and _monotonic() < self._staged.expires:
                return None, 'split_proof_pending'
            anchor = _Anchor(sample)
            self._staged = anchor
            self._wake_locked()
        while True:
            with self._cv:
                revision = self.revision
                parts, issue = self._parts_locked(anchor)
            proof = None
            if parts is not None:
                proof, issue = self._combine_anchor(anchor, parts)
            with self._cv:
                if self._closed or self._fault is not None:
                    return None, self._fault or 'split_worker_closed'
                if self._staged is not anchor:
                    return None, 'split_proof_pending'
                remaining = deadline - _monotonic()
                if remaining <= 0:
                    self._staged = None
                    return None, 'split_proof_pending'
                if proof is not None and (not isinstance(proof, dict) or proof['valid_until'] - _monotonic() > .035):
                    return proof, None
                if proof is not None:
                    issue = 'prepared_pending'
                if issue in ('new_scan', 'new_depth_unchecked'):
                    self._staged = None
                    return None, 'split_proof_pending'
                if issue not in PENDING_PROOF:
                    self._staged = None
                    return None, issue
                if revision == self.revision:
                    self._cv.wait(min(.005, remaining) if issue == 'prepared_state_busy' else remaining)

    def _parts_locked(self, anchor, record=False):
        if self._closed or self._fault is not None:
            return None, self._fault or 'split_worker_closed'
        if anchor is None or _monotonic() >= anchor.expires:
            if record:
                self._reads['no_live_anchor'] += 1
            return None, 'prepared_pending'
        if anchor.issue is not None:
            return None, anchor.issue
        if any(anchor.parts.get(name) is None for name in self.STREAMS):
            if record:
                for stream in self.STREAMS:
                    if anchor.parts.get(stream) is None:
                        self._reads['missing_' + stream] += 1
            return None, 'prepared_pending'
        return (anchor.parts['scan'], anchor.parts['depth']), None

    def _combine_anchor(self, anchor, parts):
        scan, depth = parts
        proof, issue = self.combine(anchor.sample, scan, depth, nonblocking=True)
        if self._closed or self._fault is not None:
            return None, self._fault or 'split_worker_closed'
        if anchor.issue is not None:
            return None, anchor.issue
        if (anchor not in (self._active, self._staged) or _monotonic() >= anchor.expires
                or any(anchor.parts.get(name) is None for name in self.STREAMS)):
            return None, 'prepared_pending'
        return proof, issue

    def read(self, sample):
        if not self._cv.acquire(False):
            self._reads['coordinator_busy'] += 1
            return None, 'prepared_pending'
        try:
            anchor = next((a for a in (self._active, self._staged) if a is not None and a.sample is sample), None)
            parts, issue = self._parts_locked(anchor, record=True)
        finally:
            self._cv.release()
        proof = None
        if parts is not None:
            proof, issue = self._combine_anchor(anchor, parts)
        self._reads[issue or 'ready'] += 1
        return proof, issue

    def commit(self, sample):
        """Switch the active anchor only after a command using it is submitted."""
        if not self._cv.acquire(False):
            return
        try:
            if self._staged is not None and sample is self._staged.sample:
                self._active, self._staged = self._staged, None
                self._wake_locked()
        finally:
            self._cv.release()

    def counts(self):
        with self._cv:
            return dict(self._counts, **{'read:' + key: value for key, value in self._reads.items()})

    def timeline(self):
        with self._cv:
            return list(self._timeline)

    def wait_for_change(self, revision, timeout):
        """Wait for either stream to complete a check, without taking its lock."""
        if revision == self.revision:
            self._changed.clear()
            if revision == self.revision:
                self._changed.wait(timeout)

    def notify_input(self, stream):
        # ROS callbacks may hold their own lock. Never acquire the
        # coordinator lock here; combining proofs acquires the ROS lock.
        event = self._input_ready.get('depth' if stream == 'info' else stream)
        if event is not None:
            event.set()

    def close(self):
        with self._cv:
            self._closed = True
            self._wake_locked()
        for thread in self._threads:
            thread.join(.15)
        return all(not thread.is_alive() for thread in self._threads)
