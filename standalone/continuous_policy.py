#!/usr/bin/env python2
"""Pure, forward-only intent policy for offline simulation and later review.

No ROS, process, socket, serial or motor operations occur here. ``step`` must
be called with one immutable, already pose-aligned current snapshot, an
optional newly completed Decisions result, and the same monotonic clock used
by the sensors. ``None`` means no new model result, not a failed model call;
the caller must pass an explicit failed result on failure. The caller must
provide bounded, side-effect-free front/depth checks and guarantee odometry
epoch continuity before presenting samples. The eventual executor must enforce
``valid_until`` independently and arbitrate every actual command.

The checks are sampling vetoes, not proof that blind or unobserved volumes are
clear. Intent, a zero output, or an executor acknowledgement does not verify
physical stopping. No backwards or turning intention is produced.
"""
from __future__ import division

import collections
import math
import threading

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - ROS Melodic Python 2
    STRING_TYPES = (str, unicode)  # noqa: F821
except NameError:
    INTEGER_TYPES = (int,)
    STRING_TYPES = (str,)
NUMBER_TYPES = INTEGER_TYPES + (float,)


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _epoch(value):
    return not isinstance(value, bool) and isinstance(value, INTEGER_TYPES) and value >= 0


def _pose(value):
    return (isinstance(value, (tuple, list)) and len(value) == 3
            and all(_finite(v) for v in value) and -math.pi <= value[2] <= math.pi)


def _delta(a, b):
    heading = b[2] - a[2]
    return math.hypot(b[0] - a[0], b[1] - a[1]), abs(math.atan2(math.sin(heading), math.cos(heading)))


def _fresh(stamp, now, limit):
    return _finite(stamp) and stamp >= 0 and 0 <= now - stamp < limit


def front_scan_check(sample, now):
    """Combine latest raw-hit veto with the existing sampled fusion check."""
    result = {"candidate": False, "reason": "sensor_unavailable", "scan_at": None,
              "geometric_clearance_verified": False}
    try:
        from forward_trial import _raw_scan
        from policy import assess_front
        result["scan_at"] = sample["scan_at"]
        raw_reason, _ = _raw_scan(sample)
        if raw_reason:
            result["reason"] = raw_reason
            return result
        assessed = assess_front(sample["scene"], now)
        if (not isinstance(assessed, dict) or type(assessed.get("candidate")) is not bool
                or not isinstance(assessed.get("reason"), STRING_TYPES) or not assessed["reason"]
                or assessed.get("geometric_clearance_verified") is not False):
            return result
        result["candidate"] = assessed.get("candidate") is True
        result["reason"] = assessed.get("reason", "sensor_unavailable")
        return result
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return result


class ContinuousPolicy(object):
    """One permanently latched, time-bounded stream of forward intentions."""
    def __init__(self, started_at, epoch, front_check, depth_check, speed_mps=0.02,
                 max_duration_sec=30.0, decision_ttl_sec=8.0, max_step_gap_sec=0.2,
                 acceleration_mps2=0.04, max_intent_age_sec=0.15):
        numbers = (started_at, speed_mps, max_duration_sec, decision_ttl_sec,
                   max_step_gap_sec, acceleration_mps2, max_intent_age_sec)
        if (not all(_finite(v) for v in numbers) or started_at < 0 or not _epoch(epoch)
                or not callable(front_check)
                or not (callable(depth_check) or callable(getattr(depth_check, "assess", None)))
                or not 0 < speed_mps <= 0.02 or not 0 < max_duration_sec <= 300
                or not 0 < decision_ttl_sec <= 8 or not 0 < max_step_gap_sec <= 0.2
                or not 0 < acceleration_mps2 <= 0.12 or not 0 < max_intent_age_sec <= 0.15
                or not _finite(started_at + max_duration_sec) or started_at + max_duration_sec <= started_at):
            raise ValueError("invalid continuous-policy settings")
        self.started_at, self.deadline, self.epoch = started_at, started_at + max_duration_sec, epoch
        self.front_check, self.depth_check = front_check, depth_check
        self.speed_mps, self.decision_ttl_sec = speed_mps, decision_ttl_sec
        self.max_step_gap_sec, self.acceleration_mps2 = max_step_gap_sec, acceleration_mps2
        self.max_intent_age_sec = max_intent_age_sec
        self.lock = threading.Lock()
        self.last_at = None
        self.last_odom = None
        self.history = collections.deque(maxlen=4096)
        self.decision = None
        self.decision_key = None
        self.requests = collections.OrderedDict()
        self.linear = 0.0
        self.motion_started = False
        self.stationary_anchor = None
        self.latched = False
        self.latch_reason = None
        self.sequence = 0

    def _stop(self, reason):
        self.linear = 0.0
        if not self.latched:
            self.latched, self.latch_reason = True, reason

    def _output(self, now, reason, valid_until=None):
        self.sequence += 1
        return {"linear": self.linear, "angular": 0.0, "latched": self.latched,
                "reason": self.latch_reason if self.latched else reason,
                "valid_until": valid_until, "sequence": self.sequence - 1,
                "intent_only": True, "physical_stop_verified": False,
                "geometric_clearance_verified": False}

    def _sample_ok(self, sample, now):
        try:
            ages = {"snapshot_at": 0.15, "odom_at": 0.1, "scan_at": 0.3,
                    "status_at": 1.0, "depth_at": 0.5, "color_at": 0.5}
            if (not isinstance(sample, dict) or not _epoch(sample.get("epoch"))
                    or sample["epoch"] != self.epoch
                    or not all(_fresh(sample.get(name), now, age) for name, age in ages.items())):
                return False
            stamps = [sample[name] for name in ("color_at", "scan_at", "odom_at", "depth_at")]
            if max(stamps) - min(stamps) > 0.25:
                return False
            pose, image_pose, depth_pose = sample["pose"], sample["image_pose"], sample["depth_pose"]
            scene, raw, depth = sample["scene"], sample["raw_scan"], sample["depth"]
            if (not all(_pose(p) for p in (pose, image_pose, depth_pose))
                    or not isinstance(scene, dict) or not isinstance(scene.get("coverage"), dict)
                    or scene["coverage"].get("scan_stamp") != sample["scan_at"]
                    or not _pose(scene["coverage"].get("scan_pose"))
                    or not isinstance(raw, dict) or raw.get("stamp") != sample["scan_at"]
                    or not isinstance(raw.get("frame_id"), STRING_TYPES)
                    or raw["frame_id"].lstrip("/") != "laser_link"
                    or not isinstance(depth, dict) or depth.get("stamp") != sample["depth_at"]):
                return False
            for other in (image_pose, depth_pose, scene["coverage"]["scan_pose"]):
                distance, angle = _delta(pose, other)
                if not _finite(distance) or not _finite(angle) or distance > 0.01 or angle > 0.02:
                    return False
            state = sample["state"]
            if (not isinstance(state, dict) or not _finite(state.get("stamp"))
                    or state["stamp"] != min(sample["odom_at"], sample["status_at"])
                    or type(state.get("motion_mode")) is not int or state["motion_mode"] != 0
                    or type(state.get("vehicle_state")) is not int or state["vehicle_state"] != 0
                    or type(state.get("control_mode")) is not int or state["control_mode"] != 1
                    or type(state.get("error_code")) is not int or state["error_code"] != 0
                    or not _finite(state.get("battery_voltage")) or state["battery_voltage"] < 10.0
                    or not _finite(state.get("linear_x_mps")) or not 0 <= state["linear_x_mps"] <= 0.02
                    or not _finite(state.get("linear_y_mps")) or abs(state["linear_y_mps"]) > 0.005
                    or not _finite(state.get("angular_z_radps")) or abs(state["angular_z_radps"]) > 0.02):
                return False
            if not self.motion_started:
                if (state["linear_x_mps"] > 0.005 or abs(state["angular_z_radps"]) > 0.01):
                    return False
                if self.stationary_anchor is None:
                    self.stationary_anchor = tuple(pose)
                displacement, rotation = _delta(self.stationary_anchor, pose)
                if displacement > 0.005 or rotation > 0.01:
                    return False
            odom_at = sample["odom_at"]
            if self.last_odom is not None:
                last_stamp, last_pose = self.last_odom
                dt = odom_at - last_stamp
                distance, angle = _delta(last_pose, pose)
                dx, dy = pose[0] - last_pose[0], pose[1] - last_pose[1]
                forward = math.cos(last_pose[2]) * dx + math.sin(last_pose[2]) * dy
                if (not all(_finite(v) for v in (dt, distance, angle, forward)) or not 0 <= dt <= 0.2
                        or dt == 0 and (distance > 0 or angle > 0)
                        or forward < -0.001 or distance > 0.02 * dt + 0.002
                        or angle > 0.02 * dt + 0.003):
                    return False
            pose_copy = tuple(pose)
            if self.last_odom is None or odom_at > self.last_odom[0]:
                self.history.append((odom_at, pose_copy))
            self.last_odom = (odom_at, pose_copy)
            while self.history and self.history[0][0] < now - self.decision_ttl_sec - 0.3:
                self.history.popleft()
            return True
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            return False

    def _check(self, check, sample, now, stamp_field):
        try:
            invoke = check if callable(check) else check.assess
            result = invoke(sample, now)
            if (not isinstance(result, dict) or type(result.get("candidate")) is not bool
                    or not isinstance(result.get("reason"), STRING_TYPES)
                    or not result["reason"] or len(result["reason"]) > 128
                    or result.get(stamp_field) != sample[stamp_field]
                    or result.get("geometric_clearance_verified") is not False):
                return None
            bound = result.get("valid_until")
            if bound is not None and (not _finite(bound) or bound <= now):
                return None
            return result
        except BaseException:
            return None

    def _provenance(self, value, now):
        try:
            captured, image_at, scan_at = value["captured_at"], value["image_at"], value["scan_at"]
            image_pose, token = value["image_pose"], value["observation_id"]
            if (not _epoch(value.get("epoch")) or value["epoch"] != self.epoch
                    or not all(_finite(v) for v in (captured, image_at, scan_at))
                    or not self.started_at <= captured <= min(image_at, scan_at)
                    or image_at > now or scan_at > now or abs(image_at - scan_at) > 0.25
                    or not _fresh(captured, now, self.decision_ttl_sec) or not _pose(image_pose)
                    or not isinstance(token, STRING_TYPES) or len(token) != 64
                    or any(ch not in "0123456789abcdef" for ch in token)):
                return None
            return (captured, image_at, scan_at, tuple(image_pose), value["epoch"], token)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            return None

    def register_request(self, observation, now):
        """Bind one model request to trusted capture metadata and its digest.

        The caller must derive the digest from the actual JPEG, raw scan and
        epoch, retain the same metadata for the reply, and register it before
        submitting the request. This method itself does not inspect the image.
        """
        with self.lock:
            if self.latched:
                return False
            try:
                if (not isinstance(observation, dict) or not _finite(now)
                        or not self.started_at <= now < self.deadline
                        or self.last_at is not None and now < self.last_at):
                    self._stop("request_invalid")
                    return False
                provenance = self._provenance(observation, now)
                if provenance is None:
                    self._stop("request_invalid")
                    return False
                for token in list(self.requests):
                    if not _fresh(self.requests[token][0], now, self.decision_ttl_sec):
                        del self.requests[token]
                token = provenance[-1]
                if token in self.requests:
                    if self.requests[token] != provenance:
                        self._stop("request_changed")
                        return False
                    return True
                if len(self.requests) >= 16:
                    self._stop("request_queue_full")
                    return False
                self.requests[token] = provenance
                return True
            except BaseException:
                self._stop("request_invalid")
                return False

    def _accept_decision(self, value, now):
        if not isinstance(value, dict) or value.get("ok") is not True:
            return "decision_invalid"
        choice = value.get("choice")
        if choice != "forward":
            return "model_" + choice if choice in ("stop", "left", "right") else "decision_invalid"
        try:
            provenance = self._provenance(value, now)
            if provenance is None or not self.history:
                return "decision_invalid"
            captured, image_at, scan_at, image_pose, epoch, token = provenance
            key = (captured, image_at, scan_at, image_pose, epoch, choice, token)
            if self.decision_key is not None:
                if key == self.decision_key:
                    return None
                previous = self.decision_key
                if captured < previous[0] or image_at <= previous[1] or scan_at < previous[2]:
                    return "decision_reordered"
            if self.requests.get(token) != provenance:
                return "decision_unregistered"
            nearest = min(self.history, key=lambda item: abs(item[0] - image_at))
            skew = abs(nearest[0] - image_at)
            distance, angle = _delta(nearest[1], image_pose)
            if (not _finite(distance) or not _finite(angle) or skew > 0.1
                    or distance > 0.005 + 0.02 * skew or angle > 0.01 + 0.02 * skew):
                return "decision_pose_unverified"
            self.decision = {"captured_at": captured, "image_at": image_at,
                             "scan_at": scan_at, "observation_id": token}
            self.decision_key = key
            return None
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            return "decision_invalid"

    def step(self, sample, decision, now):
        """Return an intent; a fault always emits zero and permanently latches."""
        with self.lock:
            if self.latched:
                return self._output(now, self.latch_reason)
            try:
                if (not _finite(now) or now < self.started_at
                        or self.last_at is None and now - self.started_at > self.max_step_gap_sec
                        or self.last_at is not None and not 0 < now - self.last_at <= self.max_step_gap_sec):
                    self._stop("clock_or_heartbeat_invalid")
                    return self._output(now, self.latch_reason)
                dt = 0.0 if self.last_at is None else now - self.last_at
                self.last_at = now
                if now >= self.deadline:
                    self._stop("duration_limit")
                    return self._output(now, self.latch_reason)
                # A stop, unsupported turn, or explicit model failure is a
                # veto; it must not wait for potentially expensive gate calls.
                if decision is not None and (not isinstance(decision, dict)
                                             or decision.get("ok") is not True
                                             or decision.get("choice") != "forward"):
                    self._stop(self._accept_decision(decision, now))
                    return self._output(now, self.latch_reason)
                if not self._sample_ok(sample, now):
                    self._stop("sensor_unavailable")
                    return self._output(now, self.latch_reason)
                front = self._check(self.front_check, sample, now, "scan_at")
                if front is None:
                    self._stop("gate_invalid")
                    return self._output(now, self.latch_reason)
                if not front["candidate"]:
                    self._stop(front["reason"])
                    return self._output(now, self.latch_reason)
                depth = self._check(self.depth_check, sample, now, "depth_at")
                if depth is None:
                    self._stop("gate_invalid")
                    return self._output(now, self.latch_reason)
                if not depth["candidate"]:
                    self._stop(depth["reason"])
                    return self._output(now, self.latch_reason)
                if decision is not None:
                    reason = self._accept_decision(decision, now)
                    if reason:
                        self._stop(reason)
                        return self._output(now, self.latch_reason)
                if self.decision is None:
                    self.linear = 0.0
                    if now - self.started_at >= self.decision_ttl_sec:
                        self._stop("decision_missing")
                    return self._output(now, "waiting_for_decision")
                if not _fresh(self.decision["captured_at"], now, self.decision_ttl_sec):
                    self._stop("decision_expired")
                    return self._output(now, self.latch_reason)
                valid_until = min(self.deadline, self.decision["captured_at"] + self.decision_ttl_sec,
                                  sample["snapshot_at"] + 0.15, sample["odom_at"] + 0.1,
                                  sample["scan_at"] + 0.3, sample["status_at"] + 1.0,
                                  sample["depth_at"] + 0.5, sample["color_at"] + 0.5,
                                  now + self.max_intent_age_sec)
                for result in (front, depth):
                    if result.get("valid_until") is not None:
                        valid_until = min(valid_until, result["valid_until"])
                if not _finite(valid_until) or valid_until <= now:
                    self._stop("sensor_unavailable")
                    return self._output(now, self.latch_reason)
                self.linear = min(self.speed_mps, self.linear + self.acceleration_mps2 * dt)
                if self.linear > 0:
                    self.motion_started = True
                return self._output(now, "forward_intent", valid_until)
            except BaseException:
                self._stop("policy_error")
                return self._output(now, self.latch_reason)
