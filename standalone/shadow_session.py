"""Read-only shadow wiring for observations, Decisions and ControlSession.

No ROS node, velocity publisher, relay process, socket or credential access is
created here. Pass a ``RosObservation``-compatible source, an asynchronous
model callback ``request_decision(snapshot, cancelled)``, and a report callback.
The model callback returns the Decisions result with ``ok``, ``choice`` and
``captured_at``; it must echo the earliest timestamp in the supplied snapshot.
This module attaches the capture epoch and image pose itself.
Before the first nonzero recommendation, a retained model result is eligible
only after a later fresh sample and a complete, stationary odometry history
from ``source.stationary_trace(start, end)``. The source must also expose
``continuity_snapshot()`` with epoch, current odometry, clock and guard state.
Sampling targets 0.09 s and skips missed slots; the heartbeat remains 0.05 s.
Incomplete history cannot authorize a result; no missing data recovers after
a nonzero recommendation. The model timeout still applies while waiting.

``start()`` launches independent observation, heartbeat, model and report
daemon threads. ``status()`` returns the most recent would-command and timing;
``stop(timeout)`` cancels work and reports threads that did not exit. A model or
report callback that ignores cancellation cannot be forcibly killed as a Python
thread. Callbacks and snapshot data must not mutate the source's shared state.

This is diagnostics only. A cached pose/epoch can change immediately after it
is checked, and an overloaded heartbeat may run late. Would-commands do not
establish clearance, relay delivery, driver acceptance or physical stopping.
"""

from __future__ import division

import threading

try:
    import Queue as queue  # noqa: N813 - Python 2 module name
except ImportError:
    import queue

from control_session import ControlSession, _finite, _pose, _valid_epoch
from explore import _pose_delta
from forward_trial import _stationary_trace_valid
from relay import _monotonic

_SENSOR_STAMPS = ('scan_at', 'odom_at', 'status_at', 'depth_at', 'color_at')
_PERIOD_SEC = 0.05
_OBSERVATION_PERIOD_SEC = 0.09
_MAX_OBSERVATION_AGE_SEC = 0.1
_MAX_SHADOW_SNAPSHOT_AGE_SEC = 0.15
_PREMOTION_STALE_GRACE_SEC = 0.2
_DIAGNOSTIC_STAGES = ('snapshot', 'sample_validation', 'current_pose', 'continuity_snapshot',
                      'continuity_validation', 'source_epoch_check', 'sample_clock',
                      'gap_recovery', 'sampling_schedule', 'stationary_trace')
_SAFE_EXCEPTION_TYPES = ('AttributeError', 'IndexError', 'KeyError', 'LookupError', 'OSError',
                         'IOError', 'OverflowError', 'RuntimeError', 'TypeError', 'ValueError',
                         'AssertionError', 'MemoryError', 'LookupException',
                         'ConnectivityException', 'ExtrapolationException', 'CvBridgeError')


def make_decisions_request(source, invoke, cv2_module=None):
    """Build the existing Decisions evidence on the model thread only.

    ``invoke(evidence, cancelled)`` is supplied by the caller and may wrap the
    existing bounded decision subprocess. This helper does not read keys or
    invoke a network service by itself.
    """
    if not callable(invoke):
        raise ValueError('decision invoker must be callable')

    def request(snapshot, cancelled):
        if cancelled():
            return None
        cv = cv2_module if cv2_module is not None else __import__('cv2')
        from explore import build_decision_evidence, physical_forward_view
        image = source.bridge.imgmsg_to_cv2(snapshot['color'], desired_encoding='bgr8')
        ok, encoded = cv.imencode('.jpg', image, [int(cv.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            raise ValueError('image encoding failed')
        scan = snapshot['scan']
        model_sample = dict(snapshot)
        model_sample['forward_view'] = physical_forward_view(
            scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min,
            scan.range_max, *snapshot['laser_extrinsics'])
        evidence = build_decision_evidence(model_sample, encoded.tostring())
        if evidence is None or cancelled():
            raise ValueError('decision evidence unavailable')
        return invoke(evidence, cancelled)
    return request


class ShadowSession(object):  # noqa: UP004 - ROS Melodic Python 2
    def __init__(self, source, request_decision, report, duration_sec=60.0,
                 decision_interval_sec=2.0, decision_timeout_sec=6.0, clock=_monotonic):
        if (not all(callable(getattr(source, name, None)) for name in
                    ('snapshot', 'current_pose', 'is_current_epoch', 'break_epoch'))
                or not callable(request_decision) or not callable(report) or not callable(clock)
                or not _finite(duration_sec) or not 0 < duration_sec <= 300
                or not _finite(decision_interval_sec) or not 0 < decision_interval_sec <= 8
                or not _finite(decision_timeout_sec) or not 0 < decision_timeout_sec <= 6):
            raise ValueError('invalid shadow session configuration')
        self.source = source
        self.request_decision = request_decision
        self.report = report
        self.clock = clock
        self.duration = duration_sec
        self.decision_interval = decision_interval_sec
        self.decision_timeout = decision_timeout_sec
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.model_jobs = queue.Queue(maxsize=1)
        self.report_jobs = queue.Queue(maxsize=1)
        self.threads = []
        self.started = False
        self.record = None
        self.generation = 0
        self.session_epoch = None
        self.control = None
        self.decision = None
        self.ready_decision = None
        self.pending_record = None
        self.stale_since = None
        self.gap_active = False
        self.gap_since = None
        self.gap_proof = None
        self.gap_discarded = False
        self.ever_decision = False
        self.ever_would_move = False
        self.pending = None
        self.next_decision_at = None
        self.latched = False
        self.latch_reason = None
        self.depth_diagnostic_counts = {'samples': 0, 'timestamp_stale_or_invalid': 0,
                                        'minimum_unavailable': 0, 'minimum_at_or_below_0_5m': 0,
                                        'coverage_unavailable': 0, 'coverage_insufficient': 0}
        self.model_diagnostic = {'callbacks_started': 0, 'callbacks_returned': 0,
                                 'callback_returned_none': 0, 'callback_error': 0,
                                 'discarded_stale': 0, 'discarded_on_fault': 0,
                                 'returned_after_discard': 0, 'session_timeout': 0,
                                 'worker_reported_timeout': 0, 'jobs_invalidated_stale': 0}
        self.current = {'mode': 'shadow', 'would_command': {'linear': 0.0, 'angular': 0.0},
                        'heartbeat_count': 0, 'heartbeat_interval_sec': 0.0,
                        'heartbeat_lateness_sec': 0.0, 'heartbeat_interval_samples': 0,
                        'heartbeat_interval_mean_sec': 0.0, 'heartbeat_interval_max_sec': 0.0,
                        'heartbeat_overrun_count': 0, 'latched': False,
                        'reason': 'not_started', 'model_status': 'waiting', 'epoch': None,
                        'depth_diagnostic_counts': dict(self.depth_diagnostic_counts),
                        'model_diagnostic': dict(self.model_diagnostic),
                        'diagnostic': {'stage': None, 'exception_type': None}}

    def _diagnose_locked(self, stage, error=None):
        """Keep only a fixed stage and a whitelisted exception type, never text."""
        if self.latched:
            return
        safe_stage = stage if stage in _DIAGNOSTIC_STAGES else 'sample_validation'
        name = type(error).__name__ if error is not None else None
        safe_name = name if name is None or name in _SAFE_EXCEPTION_TYPES else 'OtherException'
        self.current['diagnostic'] = {'stage': safe_stage, 'exception_type': safe_name}

    def _count_depth_locked(self, sample, now):
        """Count independent depth evidence failures, without inferring obstacles."""
        counts = self.depth_diagnostic_counts
        counts['samples'] += 1
        depth = sample.get('depth')
        if not isinstance(depth, dict):
            depth = {}
        stamp, minimum, grid = depth.get('stamp'), depth.get('min_m'), depth.get('valid_grid')
        if not _finite(stamp) or not _finite(now) or not 0 <= now - stamp <= 0.8:
            counts['timestamp_stale_or_invalid'] += 1
        if not _finite(minimum):
            counts['minimum_unavailable'] += 1
        elif minimum <= 0.5:
            counts['minimum_at_or_below_0_5m'] += 1
        if (not isinstance(grid, (tuple, list)) or len(grid) != 3
                or any(not isinstance(row, (tuple, list)) or len(row) != 3 for row in grid)
                or any(not _finite(value) or not 0 <= value <= 1 for row in grid for value in row)):
            counts['coverage_unavailable'] += 1
        elif any(value < 0.5 for row in grid for value in row):
            counts['coverage_insufficient'] += 1

    def _fault_locked(self, reason):
        if not self.latched:
            self.latched = True
            self.latch_reason = reason
            if self.pending is not None:
                self.model_diagnostic['discarded_on_fault'] += 1
                self.pending = None
                if self.current['model_status'] == 'pending':
                    self.current['model_status'] = 'discarded_on_fault'
        self.decision = None
        self.ready_decision = None
        self.pending_record = None
        self.generation += 1
        self.current['would_command'] = {'linear': 0.0, 'angular': 0.0}
        self.current['latched'] = True
        self.current['reason'] = self.latch_reason

    def _discard_gap_locked(self, now, epoch=None):
        """Discard all pre-motion provenance and restart with a fresh sample."""
        if self.ever_would_move:
            self._fault_locked('observation_unavailable')
            return
        had_model = self.pending is not None or self.ready_decision is not None or self.decision is not None
        if (self.pending_record is not None
                and self.pending_record.get('generation') == self.generation):
            self.model_diagnostic['jobs_invalidated_stale'] += 1
        if not self.gap_discarded or (_valid_epoch(epoch) and epoch != self.session_epoch):
            self.generation += 1
        self.record = None
        self.control = None
        self.decision = None
        if self.ready_decision is not None:
            self.model_diagnostic['discarded_stale'] += 1
        self.ready_decision = None
        self.pending_record = None
        self.ever_decision = False
        self.stale_since = None
        if _valid_epoch(epoch):
            self.session_epoch = epoch
        self.next_decision_at = now
        self.gap_discarded = True
        self.current['model_status'] = 'discarded_stale' if had_model else 'waiting'
        self.current['would_command'] = {'linear': 0.0, 'angular': 0.0}
        self.current['reason'] = 'waiting_for_fresh_observation'

    def _retained_reference_locked(self, now):
        """A quarantined result still needs a complete trace before use."""
        if (self.ever_would_move or self.ever_decision or self.decision is not None
                or not callable(getattr(self.source, 'stationary_trace', None))):
            return None
        reference = self.ready_decision
        if (reference is None and self.pending_record is not None
                and self.pending_record.get('generation') == self.generation):
            reference = self.pending_record
        if (reference is None or reference.get('epoch') != self.session_epoch
                or not _finite(reference.get('captured_at'))
                or not _finite(now) or not 0 <= now - reference['captured_at'] < 8.0):
            return None
        return reference

    def _validate_gap_proof_locked(self, proof, now):
        """Check source epoch, guard health, odom age and model pose budget."""
        required = ('epoch', 'pose', 'odom_at', 'checked_at', 'clock_fault', 'odom_fault')
        if (not isinstance(proof, dict) or not all(name in proof for name in required)
                or not _valid_epoch(proof.get('epoch'))):
            self._diagnose_locked('continuity_validation')
            self._fault_locked('observation_error')
            return False
        if proof.get('clock_fault') is not None or proof.get('odom_fault') is not None:
            self._diagnose_locked('continuity_validation')
            self._fault_locked('observation_error')
            return False
        epoch = proof['epoch']
        checked, odom = proof.get('checked_at'), proof.get('odom_at')
        if (not _finite(now) or not _finite(checked) or checked > now or
                (_finite(odom) and odom > checked)):
            self._fault_locked('clock_discontinuity')
            return False
        if self.session_epoch is None:
            self.session_epoch = epoch
        elif epoch != self.session_epoch:
            self._discard_gap_locked(now, epoch)
        try:
            pose = _pose(proof.get('pose'))
        except (TypeError, ValueError, OverflowError):
            self._discard_gap_locked(now, epoch)
            return False
        if (not _finite(odom) or not 0 <= now - checked <= _MAX_OBSERVATION_AGE_SEC
                or not 0 <= now - odom <= _MAX_OBSERVATION_AGE_SEC):
            self._discard_gap_locked(now, epoch)
            return False
        reference = self.ready_decision
        if (reference is None and self.pending_record is not None and
                self.pending_record['generation'] == self.generation):
            reference = self.pending_record
        if reference is not None:
            try:
                distance, rotation = _pose_delta(reference['image_pose'], pose)
                captured = reference['captured_at']
                if (reference['epoch'] != epoch or not _finite(captured)
                        or not 0 <= now - captured < 8.0
                        or distance > 0.15 or rotation > 0.35):
                    self._discard_gap_locked(now, epoch)
                    return False
            except (KeyError, TypeError, ValueError, OverflowError):
                self._discard_gap_locked(now, epoch)
                return False
        self.gap_proof = {'epoch': epoch, 'pose': pose, 'odom_at': odom, 'checked_at': checked}
        return True

    def _publish(self, item):
        try:
            self.report_jobs.put_nowait(item)
        except queue.Full:
            try:
                self.report_jobs.get_nowait()
            except queue.Empty:
                pass
            try:
                self.report_jobs.put_nowait(item)
            except queue.Full:
                pass

    def status(self):
        with self.lock:
            item = dict(self.current)
            item['would_command'] = dict(self.current['would_command'])
            item['depth_diagnostic_counts'] = dict(self.current['depth_diagnostic_counts'])
            item['model_diagnostic'] = dict(self.model_diagnostic)
            item['diagnostic'] = dict(self.current['diagnostic'])
            return item

    def start(self):
        with self.lock:
            if self.started or self.stop_event.is_set():
                raise RuntimeError('a shadow session cannot be restarted')
            now = self.clock()
            if not _finite(now) or now < 0:
                raise ValueError('invalid monotonic clock')
            self.started = True
            self.started_at = now
            self.deadline = now + self.duration
            self.next_decision_at = now
            targets = [('observation', self._observe), ('model', self._model),
                       ('report', self._report), ('heartbeat', self._heartbeat)]
            self.threads = []
            for name, target in targets:
                thread = threading.Thread(name='shadow-' + name, target=target)
                thread.daemon = True
                self.threads.append((name, thread))
        for _, thread in self.threads:
            thread.start()

    def stop(self, timeout=1.0):
        if not _finite(timeout) or not 0 <= timeout <= 10:
            raise ValueError('invalid join timeout')
        with self.lock:
            self._fault_locked('stopped')
            self.current['would_command'] = {'linear': 0.0, 'angular': 0.0}
            self.current['latched'] = True
            self.current['reason'] = self.latch_reason
        self.stop_event.set()
        share = timeout / max(1, len(self.threads))
        for _, thread in self.threads:
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(share)
        return {'alive': [name for name, thread in self.threads if thread.is_alive()]}

    def _observe(self):
        next_deadline = self.started_at
        previous_end = None
        while not self.stop_event.is_set():
            stage = 'snapshot'
            try:
                sample = self.source.snapshot()
                if sample is None:
                    with self.lock:
                        self._diagnose_locked('snapshot')
                else:
                    stage = 'sample_validation'
                    if not isinstance(sample, dict) or not _valid_epoch(sample.get('epoch')):
                        raise ValueError('invalid observation epoch')
                    epoch = sample['epoch']
                    stage = 'current_pose'
                    live_pose = self.source.current_pose()
                    if live_pose is None:
                        with self.lock:
                            self._diagnose_locked('current_pose')
                        sample = None
                    else:
                        # Keep the latest odom stamp with the pose so its 0.1s
                        # bound is rechecked at the actual heartbeat, not merely
                        # when this asynchronous sampling thread fetched it.
                        stage = 'continuity_snapshot'
                        continuity = getattr(self.source, 'continuity_snapshot', None)
                        if not callable(continuity):
                            raise ValueError('atomic continuity unavailable')
                        proof = continuity()
                        stage = 'continuity_validation'
                        required = ('epoch', 'pose', 'odom_at', 'checked_at', 'clock_fault', 'odom_fault')
                        if (not isinstance(proof, dict) or not all(name in proof for name in required)
                                or not _valid_epoch(proof['epoch']) or proof['epoch'] != epoch
                                or proof['clock_fault'] is not None
                                or proof['odom_fault'] is not None):
                            raise ValueError('invalid continuity proof')
                        proof_now = self.clock()
                        checked, odom = proof['checked_at'], proof['odom_at']
                        if (not _finite(proof_now) or not _finite(checked) or checked > proof_now
                                or (_finite(odom) and odom > checked)):
                            raise ValueError('invalid continuity clock')
                        if (proof['pose'] is None or not _finite(odom)
                                or not 0 <= proof_now - checked <= _MAX_OBSERVATION_AGE_SEC
                                or not 0 <= proof_now - odom <= _MAX_OBSERVATION_AGE_SEC):
                            with self.lock:
                                self._diagnose_locked('continuity_validation')
                            sample = None
                        else:
                            live_pose = _pose(proof['pose'])
                            live_odom_at = odom
                if sample is None:
                    # Quarantine first; continuity acquisition may block, but
                    # no stale would-command remains visible while it runs.
                    with self.lock:
                        now = self.clock()
                        if not _finite(now):
                            self._fault_locked('clock_discontinuity')
                        elif self.ever_would_move:
                            self._fault_locked('observation_unavailable')
                        elif not self.latched:
                            if not self.gap_active:
                                self.gap_active = True
                                self.gap_since = now
                                self.gap_proof = None
                                self.gap_discarded = False
                                # An already-applied decision is never reused
                                # after a missing observation.
                                if self.decision is not None and self.pending_record is None:
                                    self._discard_gap_locked(now, self.session_epoch)
                                self.decision = None
                                self.ever_decision = False
                            self.record = None
                            self.current['would_command'] = {'linear': 0.0, 'angular': 0.0}
                            self.current['reason'] = 'waiting_for_fresh_observation'
                    stage = 'continuity_snapshot'
                    continuity = getattr(self.source, 'continuity_snapshot', None)
                    if callable(continuity) and not self.stop_event.is_set():
                        proof = continuity()
                        with self.lock:
                            if not self.latched and self.gap_active:
                                self._validate_gap_proof_locked(proof, self.clock())
                    elif not callable(continuity):
                        # Legacy sources cannot distinguish odom/clock faults.
                        with self.lock:
                            if self.session_epoch is not None:
                                self._fault_locked('observation_unavailable')
                else:
                    stage = 'sample_validation'
                    pose = _pose(live_pose)
                    image_pose = _pose(sample.get('image_pose'))
                    times = [sample[name] for name in _SENSOR_STAMPS]
                    if not all(_finite(value) for value in times):
                        raise ValueError('invalid observation timestamps')
                    stage = 'source_epoch_check'
                    if not self.source.is_current_epoch(epoch):
                        raise ValueError('observation epoch changed')
                    stage = 'sample_clock'
                    now = self.clock()
                    if not _finite(now):
                        raise ValueError('invalid observation clock')
                    # Only a later sample can authorize a pre-motion model
                    # result, and only after all odometry since capture has
                    # proved stationary. Source access stays outside our lock.
                    with self.lock:
                        ready = self.ready_decision if not self.ever_would_move else None
                        trace_generation = self.generation
                    trace_attempted = False
                    trace_ok = False
                    if (ready is not None and _finite(sample.get('snapshot_at'))
                            and sample['snapshot_at'] >= ready['ready_at']
                            and 0 <= now - sample['snapshot_at'] <= _MAX_SHADOW_SNAPSHOT_AGE_SEC
                            and 0 <= now - sample['odom_at'] <= _MAX_SHADOW_SNAPSHOT_AGE_SEC
                            and 0 <= now - live_odom_at <= _MAX_OBSERVATION_AGE_SEC):
                        stage = 'stationary_trace'
                        trace_attempted = True
                        get_trace = getattr(self.source, 'stationary_trace', None)
                        if callable(get_trace):
                            trace = get_trace(ready['captured_at'], now)
                            if not self.source.is_current_epoch(epoch):
                                raise ValueError('observation epoch changed during trace')
                            latest = self.source.continuity_snapshot()
                            after = self.clock()
                            if (not isinstance(latest, dict) or not all(name in latest for name in required)
                                    or latest.get('epoch') != epoch
                                    or latest.get('clock_fault') is not None
                                    or latest.get('odom_fault') is not None):
                                raise ValueError('invalid continuity after trace')
                            latest_odom, checked = latest.get('odom_at'), latest.get('checked_at')
                            if (not _finite(after) or not _finite(checked) or not _finite(latest_odom)
                                    or not 0 <= after - checked <= _MAX_OBSERVATION_AGE_SEC
                                    or not 0 <= after - latest_odom <= _MAX_OBSERVATION_AGE_SEC
                                    or latest_odom > checked):
                                raise ValueError('invalid continuity time after trace')
                            pose = _pose(latest.get('pose'))
                            live_odom_at = latest_odom
                            now = after
                            trace_ok = (ready['epoch'] == epoch
                                        and 0 <= now - ready['captured_at'] < 8.0
                                        and _stationary_trace_valid(trace, ready['captured_at'], now,
                                                                    epoch, ready['image_pose']))
                    stage = 'gap_recovery'
                    with self.lock:
                        if self.gap_active and not self.latched:
                            proof = self.gap_proof
                            retained = self._retained_reference_locked(now)
                            if (retained is None and
                                    (self.gap_since is None or now - self.gap_since >= _PREMOTION_STALE_GRACE_SEC
                                     or proof is None or not 0 <= now - proof['odom_at'] <= _MAX_OBSERVATION_AGE_SEC
                                     or proof['epoch'] != epoch)):
                                self._discard_gap_locked(now, epoch)
                            else:
                                # Compare the returning pose with the original
                                # model capture before retaining its result.
                                reference = self.ready_decision
                                if (reference is None and self.pending_record is not None and
                                        self.pending_record['generation'] == self.generation):
                                    reference = self.pending_record
                                if reference is not None:
                                    distance, rotation = _pose_delta(reference['image_pose'], pose)
                                    if (reference['epoch'] != epoch or
                                            not 0 <= now - reference['captured_at'] < 8.0 or
                                            distance > 0.15 or rotation > 0.35):
                                        self._discard_gap_locked(now, epoch)
                            self.gap_active = False
                            self.gap_since = None
                            self.gap_proof = None
                            self.gap_discarded = False
                        if self.session_epoch is None:
                            self.session_epoch = epoch
                        elif epoch != self.session_epoch:
                            self._fault_locked('observation_epoch_changed')
                        if (trace_attempted and not trace_ok and self.ready_decision is ready
                                and self.generation == trace_generation and not self.latched):
                            self.gap_discarded = False
                            self._discard_gap_locked(now, epoch)
                        if not self.latched:
                            self._count_depth_locked(sample, now)
                            self.record = {'sample': sample, 'pose': pose, 'epoch': epoch,
                                           'image_pose': image_pose, 'captured_at': min(times),
                                           'live_odom_at': live_odom_at, 'received_at': now,
                                           'generation': self.generation,
                                           'trace_for': ready if (trace_ok and self.ready_decision is ready
                                                                  and trace_generation == self.generation) else None,
                                           'trace_at': now if trace_ok else None}
            except Exception as error:  # noqa: BLE001 - source failure cannot authorize motion
                with self.lock:
                    self.record = None
                    self._diagnose_locked(stage, error)
                    self._fault_locked('observation_error')
            # Sample at approximately the lidar rate; skip missed slots so a
            # persistently slow source cannot continuously monopolize the CPU.
            next_deadline += _OBSERVATION_PERIOD_SEC
            try:
                end = self.clock()
                if not _finite(end) or (previous_end is not None and end < previous_end):
                    raise ValueError('invalid or regressed monotonic clock')
                previous_end = end
                if end >= next_deadline:
                    missed = int((end - next_deadline) // _OBSERVATION_PERIOD_SEC) + 1
                    next_deadline += missed * _OBSERVATION_PERIOD_SEC
                self.stop_event.wait(max(0.0, next_deadline - end))
            except Exception as error:  # noqa: BLE001 - failed clock cannot authorize a sample
                with self.lock:
                    self.record = None
                    self._diagnose_locked('sampling_schedule', error)
                    self._fault_locked('clock_discontinuity')
                self.stop_event.wait(_OBSERVATION_PERIOD_SEC)

    @staticmethod
    def _model_snapshot(record):
        sample = record['sample']
        names = ('epoch', 'snapshot_at', 'image_pose', 'scan_at', 'odom_at', 'status_at',
                 'depth_at', 'color_at', 'color', 'scan', 'laser_extrinsics')
        result = {name: sample[name] for name in names if name in sample}
        result['image_pose'] = record['image_pose']
        result['state'] = dict(sample['state'])
        return result

    def _model(self):
        while not self.stop_event.is_set():
            try:
                job = self.model_jobs.get(timeout=_PERIOD_SEC)
            except queue.Empty:
                continue
            token, generation, record = job
            try:
                with self.lock:
                    if (self.stop_event.is_set() or self.latched or self.pending is None
                            or self.pending[0] != token):
                        continue
                    self.model_diagnostic['callbacks_started'] += 1
                result = self.request_decision(self._model_snapshot(record), self.stop_event.is_set)
                if isinstance(result, dict):
                    result = dict(result)
                with self.lock:
                    self.model_diagnostic['callbacks_returned'] += 1
                    if result is None:
                        self.model_diagnostic['callback_returned_none'] += 1
                epoch_current = self.source.is_current_epoch(record['epoch'])
                with self.lock:
                    if (self.stop_event.is_set() or self.latched or self.pending is None
                            or self.pending[0] != token):
                        self.model_diagnostic['returned_after_discard'] += 1
                        continue
                    now = self.clock()
                    if not _finite(now) or now < self.pending[1]:
                        self.pending = None
                        self._fault_locked('clock_discontinuity')
                    elif now - self.pending[1] >= self.decision_timeout:
                        self.pending = None
                        self.model_diagnostic['session_timeout'] += 1
                        self._fault_locked('decision_timeout')
                        self.current['model_status'] = 'timeout'
                    elif (isinstance(result, dict) and result.get('ok') is False
                          and result.get('failure') == 'subprocess_timeout'):
                        self.pending = None
                        self.model_diagnostic['worker_reported_timeout'] += 1
                        self._fault_locked('decision_timeout')
                        self.current['model_status'] = 'worker_timeout'
                    elif (not isinstance(result, dict) or result.get('ok') is not True
                          or result.get('choice') not in ('forward', 'left', 'right', 'stop')
                          or result.get('captured_at') != record['captured_at']
                          or not 0 <= now - record['captured_at'] < 8.0):
                        self.pending = None
                        self._fault_locked('decision_error')
                        self.current['model_status'] = 'failed'
                    elif self.generation != generation:
                        # A prolonged or unprovable gap invalidated this job.
                        self.pending = None
                        self.pending_record = None
                        self.model_diagnostic['discarded_stale'] += 1
                        self.next_decision_at = now
                        self.current['model_status'] = 'discarded_stale'
                    elif not epoch_current or self.session_epoch != record['epoch']:
                        self.pending = None
                        self._fault_locked('observation_epoch_changed')
                    else:
                        # Quarantine the result until a subsequent fresh sample
                        # rechecks epoch, capture time and current pose.
                        self.ready_decision = {'ok': True, 'choice': result['choice'],
                                               'captured_at': record['captured_at'],
                                               'image_pose': record['image_pose'], 'epoch': record['epoch'],
                                               'ready_at': now}
                        self.pending = None
                        self.pending_record = None
                        self.current['model_status'] = 'waiting_for_fresh_observation'
            except Exception:  # noqa: BLE001 - failed/stale model output locks zero
                with self.lock:
                    self.model_diagnostic['callback_error'] += 1
                    if not self.stop_event.is_set() and self.pending is not None and self.pending[0] == token:
                        self.pending = None
                        self._fault_locked('decision_error')
                        self.current['model_status'] = 'failed'

    def _report(self):
        while not self.stop_event.is_set() or not self.report_jobs.empty():
            try:
                item = self.report_jobs.get(timeout=_PERIOD_SEC)
            except queue.Empty:
                continue
            try:
                with self.lock:
                    if self.latched or self.stop_event.is_set():
                        item = dict(self.current)
                        item['would_command'] = dict(self.current['would_command'])
                        item['depth_diagnostic_counts'] = dict(self.current['depth_diagnostic_counts'])
                        item['model_diagnostic'] = dict(self.model_diagnostic)
                        item['diagnostic'] = dict(self.current['diagnostic'])
                self.report(item)
            except Exception:  # noqa: BLE001 - reporter errors are visible and fail closed
                with self.lock:
                    self._fault_locked('report_error')
                return

    def _heartbeat(self):
        previous = None
        next_deadline = self.started_at
        interval_count = 0
        interval_sum = 0.0
        interval_max = 0.0
        overruns = 0
        token = 0
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    now = self.clock()
                    if not _finite(now) or now < 0 or (previous is not None and now <= previous):
                        self._fault_locked('clock_discontinuity')
                        interval = 0.0
                    else:
                        interval = 0.0 if previous is None else now - previous
                        if previous is not None:
                            interval_count += 1
                            interval_sum += interval
                            interval_max = max(interval_max, interval)
                            if interval >= 0.2:
                                overruns += 1
                                self._fault_locked('heartbeat_discontinuity')
                        previous = now
                    if _finite(now) and now >= self.deadline:
                        self._fault_locked('deadline')
                    record, generation = self.record, self.generation
                    stale_waiting = self.gap_active and not self.latched
                    if stale_waiting and _finite(now):
                        proof = self.gap_proof
                        if (self._retained_reference_locked(now) is None and
                                (self.gap_since is None or now - self.gap_since >= _PREMOTION_STALE_GRACE_SEC or
                                 (proof is None and now - self.gap_since >= _MAX_OBSERVATION_AGE_SEC) or
                                 (proof is not None and not 0 <= now - proof['odom_at'] <= _MAX_OBSERVATION_AGE_SEC))):
                            self._discard_gap_locked(now, self.session_epoch)
                            generation = self.generation
                        record = None
                    if record is not None and _finite(now):
                        sample = record['sample']
                        received, live_odom = record['received_at'], record['live_odom_at']
                        snapshot_at, sample_odom = sample.get('snapshot_at'), sample.get('odom_at')
                        stamps = (received, live_odom, snapshot_at, sample_odom)
                        if not all(_finite(stamp) and 0 <= stamp <= now for stamp in stamps):
                            self._fault_locked('clock_discontinuity')
                            self.record = None
                            record = None
                        else:
                            # The decision's original image pose must remain in
                            # budget even while the model call is still pending.
                            reference = self.ready_decision
                            if (reference is None and self.pending_record is not None and
                                    self.pending_record['generation'] == self.generation):
                                reference = self.pending_record
                            if reference is not None:
                                try:
                                    image_pose = reference['image_pose']
                                    captured = reference['captured_at']
                                    distance, rotation = _pose_delta(image_pose, record['pose'])
                                    if (reference['epoch'] != self.session_epoch or
                                            not _finite(captured) or not 0 <= now - captured < 8.0):
                                        self._fault_locked('decision_expired')
                                    elif distance > 0.15 or rotation > 0.35:
                                        self._fault_locked('decision_pose_changed')
                                except (KeyError, TypeError, ValueError, OverflowError):
                                    self._fault_locked('decision_error')
                            record_stale = (now - received > _MAX_OBSERVATION_AGE_SEC or
                                            now - live_odom > _MAX_OBSERVATION_AGE_SEC)
                            sample_stale = (now - snapshot_at > _MAX_SHADOW_SNAPSHOT_AGE_SEC or
                                            now - sample_odom > _MAX_SHADOW_SNAPSHOT_AGE_SEC)
                            if not self.latched and (record_stale or sample_stale):
                                try:
                                    distance, rotation = _pose_delta(sample.get('pose'), record['pose'])
                                    matching_pose = distance <= 0.01 and rotation <= 0.03
                                except (TypeError, ValueError, OverflowError):
                                    matching_pose = False
                                if not matching_pose:
                                    self._fault_locked('observation_pose_changed')
                                    self.record = None
                                elif not self.ever_would_move:
                                    # An accepted decision may still have led
                                    # only to zero (e.g. insufficient depth).
                                    # Never replay it after missing evidence.
                                    previous_decision = self.decision
                                    self.decision = None
                                    self.ever_decision = False
                                    if previous_decision is not None:
                                        prior_capture = previous_decision['captured_at']
                                        if (self.pending_record is not None and
                                                self.pending_record['captured_at'] <= prior_capture):
                                            self.model_diagnostic['jobs_invalidated_stale'] += 1
                                            self.pending_record = None
                                            self.generation += 1
                                            generation = self.generation
                                        if (self.ready_decision is not None and
                                                self.ready_decision['captured_at'] <= prior_capture):
                                            self.ready_decision = None
                                            self.model_diagnostic['discarded_stale'] += 1
                                        if self.pending is None and self.ready_decision is None:
                                            self.current['model_status'] = 'discarded_stale'
                                    if self.stale_since is None:
                                        self.stale_since = now
                                    if (self._retained_reference_locked(now) is not None or
                                            (not record_stale and
                                             now - self.stale_since < _PREMOTION_STALE_GRACE_SEC)):
                                        # Keep the job quarantined, never use the
                                        # stale sample to compute a recommendation.
                                        stale_waiting = True
                                    else:
                                        # Pose continuity cannot be demonstrated;
                                        # retain the previous discard behavior.
                                        self.record = None
                                        self.control = None
                                        self.ready_decision = None
                                        if (self.pending_record is not None and
                                                self.pending_record['generation'] == self.generation):
                                            self.model_diagnostic['jobs_invalidated_stale'] += 1
                                        self.generation += 1
                                        generation = self.generation
                                        self.stale_since = None
                                        stale_waiting = True
                                else:
                                    self._fault_locked('observation_stale')
                                    self.record = None
                                record = None
                            elif not self.latched:
                                self.stale_since = None
                                trace_recent = (_finite(record.get('trace_at')) and
                                                0 <= now - record['trace_at'] <= _MAX_OBSERVATION_AGE_SEC)
                                if (self.ready_decision is not None and
                                        (self.ever_would_move or
                                         (record.get('trace_for') is self.ready_decision and trace_recent))):
                                    self.decision = self.ready_decision
                                    self.ready_decision = None
                                    self.ever_decision = True
                                    self.current['model_status'] = 'accepted'
                    if self.pending is not None and _finite(now) and now - self.pending[1] >= self.decision_timeout:
                        self.pending = None
                        self.model_diagnostic['session_timeout'] += 1
                        self._fault_locked('decision_timeout')
                        self.current['model_status'] = 'timeout'
                    if not self.latched and self.control is None and record is not None:
                        self.session_epoch = record['epoch']
                        self.control = ControlSession(now, self.deadline, self.session_epoch,
                                                      snapshot_max_age_sec=_MAX_SHADOW_SNAPSHOT_AGE_SEC)
                        reason = 'starting'
                    elif stale_waiting or (self.control is None and self.session_epoch is not None
                                           and record is None):
                        reason = 'waiting_for_fresh_observation'
                    else:
                        reason = 'waiting_for_observation'
                    control, approved = self.control, self.decision
                    latched = self.latched
                    if (not latched and control is not None and record is not None and
                            self.pending is None and self.ready_decision is None and
                            now >= self.next_decision_at and
                            record['captured_at'] >= control.policy.started_at and
                            (self.decision is None or record['captured_at'] > self.decision['captured_at']) and
                            (control.decision_key is None or
                             record['captured_at'] > control.decision_key[0])):
                        token += 1
                        self.pending = (token, now)
                        self.pending_record = record
                        self.next_decision_at = now + self.decision_interval
                        self.current['model_status'] = 'pending'
                        try:
                            self.model_jobs.put_nowait((token, generation, record))
                        except queue.Full:
                            self.pending = None
                            self._fault_locked('decision_queue_error')
                linear = 0.0
                if not latched and stale_waiting and control is not None and now > control.last_at:
                    # Advance only the no-observation heartbeat; no stale
                    # decision or sensor data can produce a would-command.
                    waiting = control.step(None, None, None, now, self.session_epoch)
                    if waiting['frame'] is None or waiting['latched']:
                        with self.lock:
                            self._fault_locked(waiting['reason'])
                if not latched and control is not None and record is not None and now > control.last_at:
                    result = control.step(record['sample'], approved, record['pose'], now, record['epoch'])
                    if result['frame'] is None:
                        with self.lock:
                            self._fault_locked(result['reason'])
                    else:
                        linear = result['frame']['linear']
                        reason = result['reason']
                        if result['latched']:
                            with self.lock:
                                self._fault_locked(result['reason'])
                with self.lock:
                    # Policy work itself can take tens of milliseconds. Recheck
                    # the original evidence and current-odom timestamps at the
                    # time a nonzero would-command would actually be reported.
                    finished = self.clock()
                    if not _finite(finished) or (_finite(now) and finished < now):
                        self._fault_locked('clock_discontinuity')
                    elif finished >= self.deadline:
                        self._fault_locked('deadline')
                    elif linear > 0 and record is not None:
                        sample = record['sample']
                        if (not 0 <= finished - record['live_odom_at'] <= _MAX_OBSERVATION_AGE_SEC or
                                not 0 <= finished - sample['snapshot_at'] <= _MAX_SHADOW_SNAPSHOT_AGE_SEC or
                                not 0 <= finished - sample['odom_at'] <= _MAX_SHADOW_SNAPSHOT_AGE_SEC):
                            if self.ever_would_move:
                                self._fault_locked('observation_stale')
                            else:
                                # This candidate never reached a report. Reset
                                # the control object, whose internal step did
                                # tentatively record a nonzero result.
                                self.gap_discarded = False
                                self._discard_gap_locked(finished, self.session_epoch)
                                generation = self.generation
                                linear = 0.0
                                reason = 'waiting_for_fresh_observation'
                        elif (not isinstance(approved, dict) or not _finite(approved.get('captured_at')) or
                              not 0 <= finished - approved['captured_at'] < 8.0):
                            self._fault_locked('decision_expired')
                    if self.latched or self.stop_event.is_set():
                        linear = 0.0
                        reason = self.latch_reason or 'stopped'
                    elif self.gap_active:
                        if linear > 0 and not self.ever_would_move:
                            self.gap_discarded = False
                            self._discard_gap_locked(finished, self.session_epoch)
                        linear = 0.0
                        reason = 'waiting_for_fresh_observation'
                    elif self.generation != generation:
                        if linear > 0 and not self.ever_would_move:
                            self.gap_discarded = False
                            self._discard_gap_locked(finished, self.session_epoch)
                        linear = 0.0
                        reason = 'observation_changed'
                    if linear > 0:
                        self.ever_would_move = True
                    item = {'mode': 'shadow', 'would_command': {'linear': linear, 'angular': 0.0},
                            'heartbeat_count': self.current['heartbeat_count'] + 1,
                            'heartbeat_interval_sec': interval,
                            'heartbeat_lateness_sec': max(0.0, now - next_deadline) if _finite(now) else 0.0,
                            'heartbeat_interval_samples': interval_count,
                            'heartbeat_interval_mean_sec': interval_sum / interval_count if interval_count else 0.0,
                            'heartbeat_interval_max_sec': interval_max,
                            'heartbeat_overrun_count': overruns,
                            'latched': self.latched, 'reason': reason,
                            'model_status': self.current['model_status'], 'epoch': self.session_epoch,
                            'depth_diagnostic_counts': dict(self.depth_diagnostic_counts),
                            'model_diagnostic': dict(self.model_diagnostic),
                            'diagnostic': dict(self.current['diagnostic'])}
                    self.current = item
                self._publish(dict(item, would_command=dict(item['would_command']),
                                   depth_diagnostic_counts=dict(item['depth_diagnostic_counts']),
                                   model_diagnostic=dict(item['model_diagnostic']),
                                   diagnostic=dict(item['diagnostic'])))
                if _finite(now) and now >= self.deadline:
                    self.stop_event.set()
                    break
            except Exception:  # noqa: BLE001 - never retain a stale forward recommendation
                with self.lock:
                    self._fault_locked('heartbeat_error')
                    self.current['would_command'] = {'linear': 0.0, 'angular': 0.0}
                    self.current['latched'] = True
                    self.current['reason'] = self.latch_reason
            # Advance an absolute monotonic schedule; skip missed slots rather
            # than sleeping an extra period after each control calculation.
            next_deadline += _PERIOD_SEC
            try:
                end = self.clock()
                if not _finite(end) or (_finite(now) and end < now):
                    raise ValueError('invalid or regressed monotonic clock')
                if end >= next_deadline:
                    missed = int((end - next_deadline) // _PERIOD_SEC) + 1
                    next_deadline += missed * _PERIOD_SEC
                self.stop_event.wait(max(0.0, next_deadline - end))
            except Exception:  # noqa: BLE001 - a failed clock cannot authorize motion
                with self.lock:
                    self._fault_locked('clock_discontinuity')
                self.stop_event.wait(_PERIOD_SEC)
