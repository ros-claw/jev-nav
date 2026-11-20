"""Pure, serialized control-session decisions for a standalone relay.

Call ``step(sample, decision, live_pose, now, live_epoch)`` once per heartbeat.
A non-None decision must contain ``ok``, ``choice``, ``captured_at``,
``image_pose``, and ``epoch`` from the same model observation. The session
owns the DecisionGate and rechecks decision age and capture pose on every tick.
``live_pose`` and ``live_epoch`` must be current, acquisition-aligned values;
the caller must synchronize them with observation changes and the relay write.

A return with ``frame`` set to None is not sendable: the clock/heartbeat has
failed and the caller must close or terminate the relay and supervise the base
watchdog. For sendable frames the caller must use a single ordered writer,
fail closed on write errors, supervise relay status, and verify physical
stopping independently. No ROS data is published by this class.
"""

from __future__ import division

import math
import threading

from explore import DecisionGate, snapshot_is_fresh
from policy import ForwardPolicy

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - ROS Melodic Python 2
except NameError:
    INTEGER_TYPES = (int,)
NUMBER_TYPES = INTEGER_TYPES + (float,)


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _valid_epoch(value):
    return not isinstance(value, bool) and isinstance(value, INTEGER_TYPES) and value >= 0


def _pose(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or not all(_finite(item) for item in value)
            or not -math.pi <= value[2] <= math.pi):
        raise ValueError('invalid image pose')
    return tuple(value)


class ControlSession(object):  # noqa: UP004 - ROS Melodic Python 2
    def __init__(self, started_at, deadline, epoch, snapshot_max_age_sec=0.1):
        if (not _valid_epoch(epoch) or not _finite(snapshot_max_age_sec)
                or not 0 < snapshot_max_age_sec <= 0.15):
            raise ValueError('invalid observation epoch')
        self.policy = ForwardPolicy(started_at, deadline)
        self.decision_gate = DecisionGate()
        self.lock = threading.Lock()
        self.epoch = epoch
        self.snapshot_max_age_sec = snapshot_max_age_sec
        self.last_at = started_at
        self.sequence = 0
        self.had_motion = False
        self.latched = False
        self.latch_reason = None
        self.clock_faulted = False
        self.decision_key = None

    def _latch(self, reason):
        if not self.latched:
            self.latch_reason = reason
        self.latched = True
        self.decision_gate.invalidate()

    def _verified_decision(self, value, now, live_pose):
        if value is None:
            self.decision_gate.invalidate()
            return None
        if not isinstance(value, dict):
            raise ValueError('invalid decision provenance')
        value = dict(value)
        if (value.get('ok') is not True
                or value.get('choice') not in ('forward', 'left', 'right', 'stop')
                or not _valid_epoch(value.get('epoch')) or value['epoch'] != self.epoch
                or not _finite(value.get('captured_at'))
                or value['captured_at'] < self.policy.started_at):
            raise ValueError('invalid decision provenance')
        captured = value['captured_at']
        image_pose = _pose(value.get('image_pose'))
        key = (captured, value['choice'], image_pose, value['epoch'])
        if self.decision_key is None or captured > self.decision_key[0]:
            if not self.decision_gate.accept(value, captured, image_pose, now, live_pose):
                raise ValueError('decision rejected')
            self.decision_key = key
        elif key != self.decision_key or not self.decision_gate.valid:
            raise ValueError('decision replay or changed provenance')
        current = self.decision_gate.current(now, live_pose)
        if current is None:
            raise ValueError('decision expired or capture pose changed')
        return current

    def step(self, sample, decision, live_pose, now, live_epoch):
        """Return a sendable frame or None; serialize all session state changes."""
        with self.lock:
            if (self.clock_faulted or not _finite(now)
                    or not 0 < now - self.last_at < 0.2):
                self.clock_faulted = True
                self._latch('heartbeat_discontinuity')
                return {'frame': None, 'latched': True, 'reason': 'heartbeat_discontinuity'}
            linear = 0.0
            reason = self.latch_reason if self.latched else 'waiting_for_observation'
            try:
                if not self.latched:
                    if (not _valid_epoch(live_epoch) or live_epoch != self.epoch
                            or (sample is not None and
                                (not isinstance(sample, dict) or not _valid_epoch(sample.get('epoch'))
                                 or sample['epoch'] != self.epoch))):
                        self._latch('observation_epoch_changed')
                    elif sample is None:
                        self.decision_gate.invalidate()
                        self.policy.step(None, None, None, None, now)
                        reason = 'observation_unavailable'
                    elif not snapshot_is_fresh(sample, now, live_pose, self.snapshot_max_age_sec):
                        self._latch('observation_unavailable')
                    else:
                        try:
                            verified = self._verified_decision(decision, now, live_pose)
                        except ValueError:
                            verified = None
                            self._latch('decision_unavailable')
                        if not self.latched:
                            command = self.policy.step(sample.get('scene'), sample.get('state'),
                                                       sample.get('depth'), verified, now)
                            linear = command['linear']
                            reason = command['reason']
                            if (not _finite(linear) or not 0 <= linear <= 0.05
                                    or not _finite(command['angular']) or command['angular'] != 0):
                                linear = 0.0
                                self._latch('invalid_policy_output')
                    if linear > 0:
                        self.had_motion = True
                    elif self.had_motion:
                        self._latch(reason)
                if self.latched:
                    linear = 0.0
                    reason = self.latch_reason
            except Exception:  # noqa: BLE001 - any recoverable control failure must lock zero
                self._latch('control_error')
                linear = 0.0
                reason = self.latch_reason
            frame = {'sequence': self.sequence, 'stamp': now, 'linear': linear, 'angular': 0.0}
            self.sequence += 1
            self.last_at = now
            return {'frame': frame, 'latched': self.latched, 'reason': reason}
