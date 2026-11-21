"""Check lidar and depth independently against one owned forward sample."""

from __future__ import division

import collections
import errno
import fcntl
import math
import select
import socket
import threading

from relay import _monotonic
from processed_evidence import _number

PENDING_PROOF = ('prepared_pending', 'prepared_scan_advanced', 'prepared_depth_advanced',
                 'prepared_state_busy', 'prepared_scan_expired', 'prepared_depth_expired',
                 'prepared_snapshot_expired', 'prepared_state_expired', 'prepared_evidence_expired')

def pending_sensor_deadline(latest, checked, now, max_age, history=(), receipt_timeout=.070):
    """Shorten a live proof using the complete, bounded receipt history."""
    if not latest or not checked:
        return None
    generation, previous = latest.get('generation'), checked.get('generation')
    if (generation is None or previous is None or not 1 <= generation - previous <= 4
            or not _number(checked.get('stamp'))):
        return None
    retained = {index: (stamp, received) for index, stamp, received in history}
    pending = [retained.get(index) for index in range(previous + 1, generation)]
    if any(item is None for item in pending):
        return None
    pending.append((latest.get('stamp'), latest.get('received_at')))
    deadline, stamp_before, receipt_before = None, checked['stamp'], 0.
    for stamp, received in pending:
        if (not _number(stamp) or not _number(received) or not stamp_before < stamp <= received <= now
                or received < receipt_before):
            return None
        candidate = stamp + max_age
        if receipt_timeout is not None:
            candidate = min(candidate, received + receipt_timeout)
        deadline = candidate if deadline is None else min(deadline, candidate)
        stamp_before, receipt_before = stamp, received
    return deadline if now < deadline else None


class _Anchor(object):
    def __init__(self, sample, expires, parts=None):
        self.sample = sample
        self.expires = expires
        self.parts = parts or {}
        self.generations = {}
        self.issue = None


class SensorWakeup(object):
    """Wake one waiter on input without Python 2's timed sleep polling."""

    def __init__(self):
        self._reader, self._writer = socket.socketpair()
        for endpoint in (self._reader, self._writer):
            endpoint.setblocking(False)
            fd = endpoint.fileno()
            fcntl.fcntl(fd, fcntl.F_SETFD, fcntl.fcntl(fd, fcntl.F_GETFD) | fcntl.FD_CLOEXEC)
        self._poller = select.poll()
        self._poller.register(self._reader, select.POLLIN)
        self._lock = threading.Lock()
        self._closed = False

    def set(self):
        with self._lock:
            if self._closed:
                return
            try:
                self._writer.send(b'x')
            except socket.error as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise

    def clear(self):
        try:
            self._reader.recv(65536)
        except socket.error as exc:
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise

    def wait(self, timeout):
        return bool(self._poller.poll(max(0, int(math.ceil(timeout * 1000)))))

    def close(self):
        with self._lock:
            self._closed = True
            self._reader.close()
            self._writer.close()


class SplitSensorValidation(object):
    STREAMS = ('scan', 'depth')
    STOPS = ('new_scan', 'new_depth_unchecked', 'epoch_or_guard', 'transform_changed')
    LATCH = ('front_blocked', 'depth_obstacle', 'epoch_or_guard', 'transform_changed')

    def __init__(self, capture, check, combine, completed_depth_on_read=False, anchor_deadline=None):
        self.capture, self.check, self.combine = capture, check, combine
        self._anchor_deadline = anchor_deadline or (lambda sample: sample['snapshot_at'] + .25)
        self._cv = threading.Condition()
        self._anchors = (None, None)
        self._pending_commit = None
        self._fault = None
        self._closed = False
        self._counts = collections.Counter()
        self._reads = collections.Counter()
        self._timeline = collections.deque(maxlen=192)
        self._completed_depth_on_read = completed_depth_on_read
        self._changed = SensorWakeup()
        self._stage_changed = SensorWakeup()
        self.revision = 0
        streams = ('scan',) if completed_depth_on_read else self.STREAMS
        self._input_ready = {stream: SensorWakeup() for stream in streams}
        self._threads = []
        for name in streams:
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
                    self._promote_locked()
                    now = _monotonic()
                    anchors = [anchor for anchor in self._anchors
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
                                            'scan_alignment_pending', 'scan_validation_pending',
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
                        if anchor in self._anchors:
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
        self._stage_changed.set()
        self._cv.notifyAll()

    def _wake_locked(self):
        for event in self._input_ready.values():
            event.set()
        self._changed_locked()

    def seed(self, sample, scan, depth):
        """Publish the already checked initial proof without changing expiry."""
        with self._cv:
            self._anchors = (_Anchor(sample, self._anchor_deadline(sample), {'scan': scan, 'depth': depth}),
                             self._anchors[1])
            self._wake_locked()

    def prepare(self, sample, budget=.075, scan_seed=None):
        expires = self._anchor_deadline(sample)
        deadline = expires if budget is None else min(_monotonic() + budget, expires)
        with self._cv:
            if self._closed or self._fault is not None:
                return None, self._fault or 'split_worker_closed'
            self._promote_locked()
            if self._anchors[1] is not None and _monotonic() < self._anchors[1].expires:
                return None, 'split_proof_pending'
            anchor = _Anchor(sample, expires, {'scan': scan_seed} if scan_seed is not None else None)
            self._anchors = (self._anchors[0], anchor)
            self._wake_locked()
        while True:
            with self._cv:
                revision = self.revision
                parts, issue = self._parts(anchor)
            proof = None
            if parts is not None:
                proof, issue = self._combine_anchor(anchor, parts)
            with self._cv:
                if self._closed or self._fault is not None:
                    return None, self._fault or 'split_worker_closed'
                if self._anchors[1] is not anchor:
                    return None, 'split_proof_pending'
                remaining = deadline - _monotonic()
                if remaining <= 0:
                    self._anchors = (self._anchors[0], None)
                    return None, 'split_proof_pending'
                if proof is not None and (not isinstance(proof, dict)
                        or proof['valid_until'] - _monotonic() > (0. if budget is None else .035)):
                    return proof, None
                if proof is not None:
                    issue = 'prepared_pending'
                if issue in ('new_scan', 'new_depth_unchecked'):
                    self._anchors = (self._anchors[0], None)
                    return None, 'split_proof_pending'
                if issue not in PENDING_PROOF:
                    self._anchors = (self._anchors[0], None)
                    return None, issue
            self._wait_for_revision(self._stage_changed, revision,
                                    min(.005, remaining) if issue == 'prepared_state_busy' else remaining)

    def _parts(self, anchor, record=False):
        if self._closed or self._fault is not None:
            return None, self._fault or 'split_worker_closed'
        if anchor is None or _monotonic() >= anchor.expires:
            if record:
                self._reads['no_live_anchor'] += 1
            return None, 'prepared_pending'
        if anchor.issue is not None:
            return None, anchor.issue
        scan, depth = anchor.parts.get('scan'), anchor.parts.get('depth')
        if scan is None or (depth is None and not self._completed_depth_on_read):
            if record:
                if scan is None:
                    self._reads['missing_scan'] += 1
                if depth is None:
                    self._reads['missing_depth'] += 1
            return None, 'prepared_pending'
        return (scan, depth), None

    def _combine_anchor(self, anchor, parts):
        scan, depth = parts
        proof, issue = self.combine(anchor.sample, scan, depth, nonblocking=True)
        if self._closed or self._fault is not None:
            return None, self._fault or 'split_worker_closed'
        if anchor.issue is not None:
            return None, anchor.issue
        if anchor not in self._anchors or self._parts(anchor)[0] is None:
            return None, 'prepared_pending'
        return proof, issue

    def read(self, sample):
        active, staged = self._anchors
        if staged is not None and staged is self._pending_commit:
            active, staged = staged, None
        anchor = next((a for a in (active, staged) if a is not None and a.sample is sample), None)
        parts, issue = self._parts(anchor, record=True)
        proof = None
        if parts is not None:
            proof, issue = self._combine_anchor(anchor, parts)
        self._reads[issue or 'ready'] += 1
        return proof, issue

    def commit(self, sample):
        """Switch the active anchor only after a command using it is submitted."""
        staged = self._anchors[1]
        if staged is None or sample is not staged.sample:
            return
        self._pending_commit = staged
        if not self._cv.acquire(False):
            return
        try:
            self._promote_locked()
        finally:
            self._cv.release()

    def _promote_locked(self):
        pending = self._pending_commit
        if pending is not None and pending is self._anchors[1]:
            self._anchors = (pending, None)
            self._pending_commit = None
            self._wake_locked()

    def counts(self):
        with self._cv:
            return dict(self._counts, **{'read:' + key: value for key, value in self._reads.items()})

    def timeline(self):
        with self._cv:
            return list(self._timeline)

    def wait_for_change(self, revision, timeout):
        """Wait for either stream to complete a check, without taking its lock."""
        self._wait_for_revision(self._changed, revision, timeout)

    def wait_for_stage_slot(self, timeout):
        """Avoid assembling another scene while the current stage owns the slot."""
        revision = self.revision
        staged = self._anchors[1]
        if self._fault is not None or self._closed or staged is None or staged is self._pending_commit:
            return True
        remaining = staged.expires - _monotonic()
        if remaining <= 0:
            return True
        self._wait_for_revision(self._stage_changed, revision, min(remaining, timeout))
        return False

    def _wait_for_revision(self, event, revision, timeout):
        if revision == self.revision:
            event.clear()
            if revision == self.revision:
                event.wait(timeout)

    def notify_input(self, stream):
        # ROS callbacks may hold their own lock. Never acquire the
        # coordinator lock here; combining proofs acquires the ROS lock.
        event = self._input_ready.get('depth' if stream in ('info', 'odom') else stream)
        if event is not None:
            event.set()

    def note_depth_obstacle(self):
        with self._cv:
            if self._anchors[0] is not None and self._fault is None and not self._closed:
                self._fault = 'depth_obstacle'
                self._counts['depth:producer_obstacle'] += 1
                self._wake_locked()

    def note_completed_depth(self):
        if self._completed_depth_on_read:
            with self._cv:
                if not self._closed:
                    self._changed_locked()

    def close(self):
        with self._cv:
            self._closed = True
            self._wake_locked()
        for thread in self._threads:
            thread.join(.15)
        closed = all(not thread.is_alive() for thread in self._threads)
        if closed:
            for event in list(self._input_ready.values()) + [self._changed, self._stage_changed]:
                event.close()
        return closed
