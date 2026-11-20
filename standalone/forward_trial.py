#!/usr/bin/env python2
"""Offline-only one-shot model-driven forward trial candidate.

There is intentionally no ROS, motor, process, socket, or real-relay adapter
here. Injected source, model and relay objects must identify as simulation-only.
An eventual real adapter would require the relay's independently enforced
first-motion limit and verified shutdown, sensor calibration, braking, and
command arbitration. Supervision and an emergency stop do not establish that
unknown or unobserved parts of the robot's path are clear.
"""

from __future__ import division

import copy
import hashlib
import json
import math
import sys
import threading

from native_scan_loader import load_native_scan
from policy import assess_front

_native_scan = load_native_scan()

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - ROS Melodic Python 2
    STRING_TYPES = (str, unicode)  # noqa: F821
except NameError:
    INTEGER_TYPES = (int,)
    STRING_TYPES = (str,)
NUMBER_TYPES = INTEGER_TYPES + (float,)

MAX_DURATION_SEC = 0.6
MAX_SPEED_MPS = 0.02
MAX_DISTANCE_M = 0.02
MAX_PERIOD_SEC = 0.05
RELAY_MOTION_LIMIT_SEC = 0.3
RELAY_READY_LINE = b"READY motion_limit_sec=0.300000\n"
MODEL_MAX_AGE_SEC = 8.0


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _pose(value):
    return (isinstance(value, (tuple, list)) and len(value) == 3
            and all(_finite(item) for item in value) and -math.pi <= value[2] <= math.pi)


def _delta(first, second):
    distance = math.hypot(second[0] - first[0], second[1] - first[1])
    angle = second[2] - first[2]
    return distance, abs(math.atan2(math.sin(angle), math.cos(angle)))


def _fresh(stamp, now, age):
    return _finite(stamp) and stamp >= 0 and 0 <= now - stamp <= age


def _timestamps_fresh(sample, now):
    try:
        ages = {"snapshot_at": 0.1, "color_at": 0.5, "scan_at": 0.3,
                "odom_at": 0.1, "status_at": 1.0, "depth_at": 0.5}
        return isinstance(sample, dict) and all(_fresh(sample[name], now, age)
                                                for name, age in ages.items())
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _raw_scan(sample, need_canonical=True, return_counts=False):
    """Validate and classify raw sensor-frame rays; retain unknown encodings."""
    try:
        if (type(need_canonical) is not bool or type(return_counts) is not bool
                or return_counts and need_canonical):
            return "sensor_unavailable", None
        raw, extrinsics = sample["raw_scan"], sample["laser_extrinsics"]
        if (not isinstance(raw, dict) or not isinstance(raw.get("frame_id"), STRING_TYPES)
                or raw["frame_id"].lstrip("/") != "laser_link" or not _finite(raw.get("stamp"))
                or raw.get("stamp") != sample["scan_at"]
                or not _pose(extrinsics)):
            return "sensor_unavailable", None
        angle_min, increment = raw["angle_min"], raw["angle_increment"]
        range_min, range_max = raw["range_min"], raw["range_max"]
        time_increment, ranges = raw["time_increment"], raw["ranges"]
        if (not all(_finite(value) for value in (angle_min, increment, range_min, range_max, time_increment))
                or increment == 0 or not 0 < range_min < range_max <= 100
                or time_increment < 0 or not isinstance(ranges, (tuple, list))
                or not 1 <= len(ranges) <= 8192):
            return "sensor_unavailable", None
        angle_end = angle_min + (len(ranges) - 1) * increment
        duration = (len(ranges) - 1) * time_increment
        if (not _finite(angle_end) or not _finite(duration) or duration > 0.3
                or not -math.pi - 1e-5 <= angle_min <= math.pi + 1e-5
                or not -math.pi - 1e-5 <= angle_end <= math.pi + 1e-5):
            return "sensor_unavailable", None
        scan_pose = sample["scene"]["coverage"]["scan_pose"]
        pose = sample["pose"]
        if not _pose(scan_pose) or not _pose(pose):
            return "sensor_unavailable", None
        if not need_canonical and _native_scan is not None:
            try:
                accelerated = _native_scan.fast_raw_veto(
                    ranges, angle_min, increment, range_min, range_max, extrinsics,
                    scan_pose, pose, duration, MAX_SPEED_MPS, math.pi / 2)
                if (isinstance(accelerated, tuple) and len(accelerated) == 3
                        and type(accelerated[0]) is bool
                        and all(not isinstance(value, bool) and isinstance(value, INTEGER_TYPES)
                                for value in accelerated[1:])
                        and 0 <= accelerated[2] <= accelerated[1] <= len(ranges)):
                    blocked, front_count, valid_count = accelerated
                    if blocked:
                        return "front_blocked", None
                    if return_counts:
                        return None, (front_count, valid_count)
                    coverage = sample["scene"]["coverage"]
                    if (front_count != coverage.get("front_samples")
                            or valid_count != coverage.get("latest_valid_samples")):
                        return "sensor_unavailable", None
                    return None, None
            except Exception:  # noqa: BLE001 - retain the existing checked fallback
                pass
        sanitized = []
        front_count = valid_count = 0
        blocked = False
        scan_c, scan_s = math.cos(scan_pose[2]), math.sin(scan_pose[2])
        base_c, base_s = math.cos(pose[2]), math.sin(pose[2])
        margin = MAX_SPEED_MPS * duration
        side_margin = margin + 0.6 * 0.02 * duration
        for index, value in enumerate(ranges):
            if value is None:
                sanitized.append(None)
                measured = False
            elif isinstance(value, bool) or not isinstance(value, NUMBER_TYPES):
                return "sensor_unavailable", None
            elif not _finite(value):
                sanitized.append("nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf"))
                measured = False
            else:
                sanitized.append(value)
                measured = value > 0 and range_min <= value < range_max
            angle = angle_min + index * increment
            if not -math.pi / 2 <= angle <= math.pi / 2:
                continue
            front_count += 1
            if not measured:
                continue
            valid_count += 1
            bearing = angle + extrinsics[2]
            bx = extrinsics[0] + value * math.cos(bearing)
            by = extrinsics[1] + value * math.sin(bearing)
            wx = scan_pose[0] + scan_c * bx - scan_s * by
            wy = scan_pose[1] + scan_s * bx + scan_c * by
            dx, dy = wx - pose[0], wy - pose[1]
            x, y = base_c * dx + base_s * dy, -base_s * dx + base_c * dy
            if not _finite(x) or not _finite(y):
                return "sensor_unavailable", None
            if -margin <= x <= 0.6 + margin and abs(y) <= 0.15 + side_margin:
                blocked = True
        if blocked:
            return "front_blocked", None
        if return_counts:
            return None, (front_count, valid_count)
        coverage = sample["scene"]["coverage"]
        if (front_count != coverage.get("front_samples")
                or valid_count != coverage.get("latest_valid_samples")):
            return "sensor_unavailable", None
        canonical = {"stamp": raw["stamp"], "frame_id": raw["frame_id"], "angle_min": angle_min,
                     "angle_increment": increment, "range_min": range_min, "range_max": range_max,
                     "time_increment": time_increment, "ranges": sanitized}
        return None, canonical if need_canonical else None
    except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError, IndexError):
        return "sensor_unavailable", None


def _stationary_trace_valid(trace, start, end, epoch, anchor, max_gap_sec=0.1):
    """Check a continuous odometry history during model inference and arming."""
    try:
        if (not isinstance(trace, (tuple, list)) or not 1 <= len(trace) <= 4096
                or not _finite(start) or not _finite(end) or start > end or not _pose(anchor)
                or not _finite(max_gap_sec) or not 0 < max_gap_sec <= 0.15):
            return False
        prior_stamp = prior_seq = None
        for item in trace:
            if not isinstance(item, dict):
                return False
            stamp, seq, pose = item["stamp"], item["seq"], item["pose"]
            if (not _finite(stamp) or not 0 <= stamp <= end or not _pose(pose)
                    or isinstance(seq, bool) or not isinstance(seq, INTEGER_TYPES) or not 0 <= seq <= 4294967295
                    or isinstance(item.get("epoch"), bool) or not isinstance(item.get("epoch"), INTEGER_TYPES)
                    or item.get("epoch") != epoch
                    or not _finite(item.get("linear_x_mps")) or abs(item["linear_x_mps"]) > 0.005
                    or not _finite(item.get("linear_y_mps")) or abs(item["linear_y_mps"]) > 0.005
                    or not _finite(item.get("angular_z_radps")) or abs(item["angular_z_radps"]) > 0.01
                    or (prior_stamp is not None and
                        (not 0 < stamp - prior_stamp <= max_gap_sec or seq <= prior_seq))):
                return False
            distance, yaw = _delta(anchor, pose)
            if not _finite(distance) or not _finite(yaw) or distance > 0.005 or yaw > 0.01:
                return False
            prior_stamp, prior_seq = stamp, seq
        return (0 <= start - trace[0]["stamp"] <= max_gap_sec
                and 0 <= end - trace[-1]["stamp"] <= max_gap_sec)
    except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError, IndexError):
        return False


def _snapshot_reason(sample, now):
    """Check a current, already pose-aligned synthetic observation."""
    try:
        if not isinstance(sample, dict):
            return "sensor_unavailable"
        if not _timestamps_fresh(sample, now):
            return "sensor_unavailable"
        epoch = sample["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, INTEGER_TYPES) or epoch < 0:
            return "sensor_unavailable"
        image = sample["image_jpeg"]
        if (not isinstance(image, bytes) or not image.startswith(b"\xff\xd8")
                or not image.endswith(b"\xff\xd9") or not 4 < len(image) <= 2000000):
            return "sensor_unavailable"
        pose, image_pose, depth_pose = sample["pose"], sample["image_pose"], sample["depth_pose"]
        if not all(_pose(item) for item in (pose, image_pose, depth_pose)):
            return "sensor_unavailable"
        scene = sample["scene"]
        if (not isinstance(scene, dict) or not isinstance(scene.get("coverage"), dict)
                or scene["coverage"].get("scan_stamp") != sample["scan_at"]
                or not _pose(scene["coverage"].get("scan_pose"))):
            return "sensor_unavailable"
        for other in (image_pose, depth_pose, scene["coverage"]["scan_pose"]):
            distance, rotation = _delta(pose, other)
            if not _finite(distance) or not _finite(rotation) or distance > 0.01 or rotation > 0.02:
                return "sensor_unavailable"
        stamps = [sample[name] for name in ("color_at", "scan_at", "odom_at", "depth_at")]
        if max(stamps) - min(stamps) > 0.25:
            return "sensor_unavailable"
        raw_reason, _canonical = _raw_scan(sample)
        if raw_reason:
            return raw_reason
        state = sample["state"]
        if (not isinstance(state, dict) or not _finite(state.get("stamp"))
                or state.get("stamp") != min(sample["odom_at"], sample["status_at"])
                or type(state.get("motion_mode")) is not int or state["motion_mode"] != 0
                or type(state.get("vehicle_state")) is not int or state["vehicle_state"] != 0
                or type(state.get("control_mode")) is not int or state["control_mode"] != 1
                or type(state.get("error_code")) is not int or state["error_code"] != 0
                or not _finite(state.get("battery_voltage")) or state["battery_voltage"] < 10.0
                or not _finite(state.get("linear_x_mps")) or not 0 <= state["linear_x_mps"] <= MAX_SPEED_MPS
                or not _finite(state.get("linear_y_mps")) or abs(state["linear_y_mps"]) > 0.005
                or not _finite(state.get("angular_z_radps")) or abs(state["angular_z_radps"]) > 0.02):
            return "sensor_unavailable"
        depth = sample["depth"]
        if (not isinstance(depth, dict) or not _finite(depth.get("stamp"))
                or depth.get("stamp") != sample["depth_at"]
                or not _finite(depth.get("min_m")) or depth["min_m"] <= 0.5):
            return "sensor_unavailable"
        grid = depth.get("valid_grid")
        if (not isinstance(grid, (tuple, list)) or len(grid) != 3
                or any(not isinstance(row, (tuple, list)) or len(row) != 3 for row in grid)
                or any(not _finite(value) or not 0.5 <= value <= 1.0 for row in grid for value in row)):
            return "sensor_unavailable"
        front = assess_front(scene, now)
        if not front["candidate"]:
            return "front_blocked" if front["reason"] == "forward_obstacle" else "sensor_unavailable"
        return None
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return "sensor_unavailable"


def _evidence(sample):
    scene = sample["scene"]
    scan_stamp = sample["scan_at"]
    raw_reason, raw_scan = _raw_scan(sample)
    if raw_reason:
        raise _StopError(raw_reason)
    evidence = {"image_jpeg": sample["image_jpeg"], "color_at": sample["color_at"],
                "scan_at": scan_stamp, "epoch": sample["epoch"],
                "raw_scan": raw_scan, "nonreturns_are_unknown": True,
                "latest_rays": copy.deepcopy([ray for ray in scene["rays"] if ray["stamp"] == scan_stamp]),
                "retained_hits": copy.deepcopy(scene["occupied_hits"])}
    metadata = {key: evidence[key] for key in
                ("color_at", "scan_at", "epoch", "raw_scan", "latest_rays", "retained_hits")}
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(evidence["image_jpeg"])
    digest.update(encoded)
    evidence["observation_id"] = digest.hexdigest()
    return evidence


class _StopError(Exception):
    pass


class ForwardTrial(object):  # noqa: UP004 - ROS Melodic Python 2
    """One simulation session. Injected adapters must be nonblocking fixtures."""

    def __init__(self, source, model, relay, clock, duration_sec=0.6,
                 speed_mps=0.02, period_sec=0.05):
        if (not all(getattr(adapter, "simulation_only", None) is True for adapter in (source, model, relay))
                or not callable(getattr(clock, "now", None)) or not callable(getattr(clock, "sleep", None))
                or not all(_finite(v) for v in (duration_sec, speed_mps, period_sec))
                or not 0 < duration_sec <= MAX_DURATION_SEC
                or not 0 < speed_mps <= MAX_SPEED_MPS or not 0 < period_sec <= MAX_PERIOD_SEC):
            raise ValueError("offline fixtures and bounded trial settings required")
        self.source, self.model, self.relay, self.clock = source, model, relay, clock
        self.duration_sec, self.speed_mps, self.period_sec = duration_sec, speed_mps, period_sec
        self._used = False
        self._start_lock = threading.Lock()
        self._last_time = None

    def _now(self):
        now = self.clock.now()
        if not _finite(now) or now < 0 or (self._last_time is not None and now < self._last_time):
            raise _StopError("clock_invalid")
        self._last_time = now
        return now

    def _read(self):
        sample = self.source.snapshot()
        now = self._now()
        reason = _snapshot_reason(sample, now)
        if reason:
            raise _StopError(reason)
        return sample, now

    def _require_stationary(self, before, now):
        start = min(before["color_at"], before["scan_at"])
        try:
            trace = self.source.stationary_trace(start, now)
            if self.source.is_current_epoch(before["epoch"]) is not True:
                raise _StopError("sensor_unavailable")
        except _StopError:
            raise
        except BaseException:
            raise _StopError("sensor_unavailable")
        after = self._now()
        if (not _fresh(before["color_at"], after, MODEL_MAX_AGE_SEC)
                or not _fresh(before["scan_at"], after, MODEL_MAX_AGE_SEC)):
            raise _StopError("model_stale")
        if not _stationary_trace_valid(trace, start, after, before["epoch"], before["image_pose"]):
            raise _StopError("sensor_unavailable")
        return after

    def run(self):
        with self._start_lock:
            if self._used:
                return {"reason": "already_used", "diagnostic_only": True, "physical_stop_verified": False}
            self._used = True
        result = {"reason": "exception", "diagnostic_only": True, "physical_stop_verified": False,
                  "stop_requested": False, "motion_requested": False}
        try:
            started = self._now()
            before, now = self._read()
            if before["color_at"] < started or before["scan_at"] < started:
                raise _StopError("sensor_unavailable")
            if (before["state"]["linear_x_mps"] > 0.005
                    or abs(before["state"]["angular_z_radps"]) > 0.01):
                raise _StopError("sensor_unavailable")
            evidence = _evidence(before)
            observation_id = evidence["observation_id"]
            decision = self.model.decide(copy.deepcopy(evidence))
            now = self._now()
            if (not _fresh(before["color_at"], now, MODEL_MAX_AGE_SEC)
                    or not _fresh(before["scan_at"], now, MODEL_MAX_AGE_SEC)):
                raise _StopError("model_stale")
            if (not isinstance(decision, dict) or decision.get("ok") is not True
                    or decision.get("observation_id") != observation_id):
                raise _StopError("model_invalid")
            if decision.get("choice") != "forward":
                raise _StopError("model_not_forward")
            current, now = self._read()
            if current["epoch"] != before["epoch"]:
                raise _StopError("sensor_unavailable")
            if (current["state"]["linear_x_mps"] > 0.005
                    or abs(current["state"]["angular_z_radps"]) > 0.01):
                raise _StopError("sensor_unavailable")
            distance, rotation = _delta(before["image_pose"], current["pose"])
            if distance > 0.005 or rotation > 0.01:
                raise _StopError("sensor_unavailable")
            now = self._require_stationary(before, now)
            if not _timestamps_fresh(current, now):
                raise _StopError("sensor_unavailable")
            deadline = now + self.duration_sec
            if not _finite(deadline) or deadline <= now:
                raise _StopError("clock_invalid")
            try:
                contract = self.relay.arm(RELAY_MOTION_LIMIT_SEC)
            except BaseException:
                raise _StopError("relay_failure")
            if (not isinstance(contract, dict) or contract.get("ready_line") != RELAY_READY_LINE
                    or contract.get("simulation_only") is not True
                    or contract.get("relative_motion_limit") is not True
                    or contract.get("latches_zero") is not True
                    or contract.get("zero_on_parent_loss") is not True
                    or contract.get("independent_heartbeat") is not True):
                raise _StopError("relay_contract")
            anchor = previous = current["pose"]
            previous_odom = current["odom_at"]
            path_distance = 0.0
            sequence = 0
            last_command_at = None
            first_command_at = None
            while True:
                now = self._now()
                if now >= deadline:
                    raise _StopError("time_limit")
                if first_command_at is not None and now >= first_command_at + RELAY_MOTION_LIMIT_SEC:
                    raise _StopError("motion_limit")
                if (not _fresh(before["color_at"], now, MODEL_MAX_AGE_SEC)
                        or not _fresh(before["scan_at"], now, MODEL_MAX_AGE_SEC)):
                    raise _StopError("model_stale")
                if sequence == 0:
                    now = self._require_stationary(before, now)
                    if now >= deadline:
                        raise _StopError("time_limit")
                    if not _timestamps_fresh(current, now):
                        raise _StopError("sensor_unavailable")
                current, now = self._read()
                if now >= deadline:
                    raise _StopError("time_limit")
                if (not _fresh(before["color_at"], now, MODEL_MAX_AGE_SEC)
                        or not _fresh(before["scan_at"], now, MODEL_MAX_AGE_SEC)):
                    raise _StopError("model_stale")
                if current["epoch"] != before["epoch"] or current["odom_at"] < previous_odom:
                    raise _StopError("odom_discontinuity")
                step_distance, step_rotation = _delta(previous, current["pose"])
                displacement, yaw = _delta(anchor, current["pose"])
                if not all(_finite(v) for v in (step_distance, step_rotation, displacement, yaw)):
                    raise _StopError("odom_discontinuity")
                delta_time = current["odom_at"] - previous_odom
                dx, dy = current["pose"][0] - previous[0], current["pose"][1] - previous[1]
                forward_delta = math.cos(previous[2]) * dx + math.sin(previous[2]) * dy
                if (delta_time == 0 and (step_distance > 0 or step_rotation > 0)
                        or not _finite(forward_delta) or forward_delta < 0
                        or step_distance > MAX_SPEED_MPS * delta_time + 0.002 or yaw > 0.03):
                    raise _StopError("odom_discontinuity")
                path_distance += step_distance
                previous, previous_odom = current["pose"], current["odom_at"]
                result["odom_displacement_m"] = displacement
                result["odom_path_m"] = path_distance
                if max(path_distance, displacement) + self.speed_mps * self.period_sec >= MAX_DISTANCE_M:
                    raise _StopError("distance_limit")
                try:
                    if self.relay.healthy() is not True:
                        raise _StopError("relay_failure")
                except _StopError:
                    raise
                except BaseException:
                    raise _StopError("relay_failure")
                try:
                    if (self.source.is_current_epoch(current["epoch"]) is not True
                            or self.source.is_current_sample(current) is not True):
                        raise _StopError("sensor_unavailable")
                except _StopError:
                    raise
                except BaseException:
                    raise _StopError("sensor_unavailable")
                now = self._now()
                if now >= deadline:
                    raise _StopError("time_limit")
                if (not _fresh(before["color_at"], now, MODEL_MAX_AGE_SEC)
                        or not _fresh(before["scan_at"], now, MODEL_MAX_AGE_SEC)):
                    raise _StopError("model_stale")
                reason = _snapshot_reason(current, now)
                if reason:
                    raise _StopError(reason)
                try:
                    if (self.source.is_current_epoch(current["epoch"]) is not True
                            or self.source.is_current_sample(current) is not True):
                        raise _StopError("sensor_unavailable")
                except _StopError:
                    raise
                except BaseException:
                    raise _StopError("sensor_unavailable")
                now = self._now()
                if now >= deadline:
                    raise _StopError("time_limit")
                if first_command_at is not None and now >= first_command_at + RELAY_MOTION_LIMIT_SEC:
                    raise _StopError("motion_limit")
                if not _timestamps_fresh(current, now):
                    raise _StopError("sensor_unavailable")
                if (not _fresh(before["color_at"], now, MODEL_MAX_AGE_SEC)
                        or not _fresh(before["scan_at"], now, MODEL_MAX_AGE_SEC)):
                    raise _StopError("model_stale")
                if last_command_at is not None and now <= last_command_at:
                    raise _StopError("clock_invalid")
                if first_command_at is None:
                    if now + RELAY_MOTION_LIMIT_SEC <= now:
                        raise _StopError("clock_invalid")
                    first_command_at = now
                try:
                    # A failed acknowledgement can still follow a write side effect.
                    result["motion_requested"] = True
                    if self.relay.send_frame(sequence, now, self.speed_mps, 0.0) is not True:
                        raise _StopError("relay_failure")
                except _StopError:
                    raise
                except BaseException:
                    raise _StopError("relay_failure")
                last_command_at = now
                sequence += 1
                now = self._now()
                if now >= deadline:
                    raise _StopError("time_limit")
                self.clock.sleep(min(self.period_sec, deadline - now))
        except _StopError as exc:
            result["reason"] = str(exc)
        except BaseException:
            result["reason"] = "exception"
        finally:
            try:
                result["stop_requested"] = self.relay.stop_and_latch() is True
            except BaseException:
                result["stop_requested"] = False
            try:
                self.relay.close()
            except BaseException:
                pass
            if not result["stop_requested"]:
                result["trigger_reason"] = result["reason"]
                result["reason"] = "relay_stop_failed"
        return result


if __name__ == "__main__":
    sys.stderr.write("Offline fixtures only; no physical relay or robot adapter.\n")
    sys.exit(2)
