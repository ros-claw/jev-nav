"""ROS-independent continuity and alignment checks for sensor observations.

Callers must feed each stateful guard serially and in actual acquisition order.
Odometry and sensor timestamps must use the same local monotonic clock;
ClockGuard instead receives a paired ROS-wall and monotonic sample. Thresholds
are caller-supplied assumptions, not measured hardware limits or clearance
proof. Clock and odometry faults latch for the lifetime of their instances.
"""

from __future__ import division

import collections
import math

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


def _valid_pose(pose):
    return (isinstance(pose, (tuple, list)) and len(pose) == 3
            and all(_finite(value) for value in pose)
            and -math.pi <= pose[2] <= math.pi)


def _pose_delta(first, second):
    distance = math.hypot(second[0] - first[0], second[1] - first[1])
    angle = second[2] - first[2]
    rotation = abs(math.atan2(math.sin(angle), math.cos(angle)))
    return distance, rotation


class ClockGuard(object):  # noqa: UP004 - ROS Melodic Python 2
    """Latch if wall/monotonic clock offsets jump or either clock regresses."""

    def __init__(self, max_offset_jump_sec):
        if not _finite(max_offset_jump_sec) or max_offset_jump_sec < 0:
            raise ValueError("invalid clock offset threshold")
        self.max_offset_jump_sec = max_offset_jump_sec
        self.faulted = False
        self.fault_reason = None
        self._last_ros = None
        self._last_mono = None
        self._last_offset = None

    def _fault(self, reason):
        self.faulted = True
        self.fault_reason = reason
        return False

    def observe(self, ros_now, mono_now):
        if self.faulted:
            return False
        if (not _finite(ros_now) or not _finite(mono_now)
                or ros_now <= 0 or mono_now < 0):
            return self._fault("invalid_clock")
        offset = ros_now - mono_now
        if not _finite(offset):
            return self._fault("invalid_clock")
        if self._last_mono is not None:
            if mono_now < self._last_mono or ros_now < self._last_ros:
                return self._fault("clock_regression")
            if abs(offset - self._last_offset) > self.max_offset_jump_sec:
                return self._fault("clock_offset_jump")
        self._last_ros, self._last_mono, self._last_offset = ros_now, mono_now, offset
        return True


class OdomGuard(object):  # noqa: UP004 - ROS Melodic Python 2
    """Latch on an odometry epoch, time, freshness, or motion-bound violation."""

    def __init__(self, max_linear_mps, max_angular_radps,
                 position_slack_m, yaw_slack_rad, max_gap_sec, history_window_sec=1.0):
        values = (max_linear_mps, max_angular_radps, position_slack_m, yaw_slack_rad,
                  max_gap_sec, history_window_sec)
        if (not all(_finite(value) for value in values)
                or any(value < 0 for value in values) or max_gap_sec <= 0 or history_window_sec <= 0):
            raise ValueError("invalid odometry limits")
        self.max_linear_mps = max_linear_mps
        self.max_angular_radps = max_angular_radps
        self.position_slack_m = position_slack_m
        self.yaw_slack_rad = yaw_slack_rad
        self.max_gap_sec = max_gap_sec
        self.history_window_sec = history_window_sec
        self.faulted = False
        self.fault_reason = None
        self._initial = None
        self._previous = None
        self._history = collections.deque()

    def _fault(self, reason):
        self.faulted = True
        self.fault_reason = reason
        return False

    def _within_bounds(self, before, after):
        elapsed = after[1] - before[1]
        distance, rotation = _pose_delta(before[2], after[2])
        translation_limit = self.max_linear_mps * elapsed + self.position_slack_m
        rotation_limit = self.max_angular_radps * elapsed + self.yaw_slack_rad
        return (all(_finite(value) for value in (elapsed, distance, rotation,
                                                 translation_limit, rotation_limit))
                and elapsed > 0 and distance <= translation_limit and rotation <= rotation_limit)

    def observe(self, seq, stamp, pose):
        if self.faulted:
            return False
        if (isinstance(seq, bool) or not isinstance(seq, INTEGER_TYPES)
                or not 0 <= seq <= 4294967295 or not _finite(stamp) or stamp < 0
                or not _valid_pose(pose)):
            return self._fault("invalid_odometry")
        sample = (seq, stamp, tuple(pose))
        if self._previous is not None:
            if seq <= self._previous[0]:
                return self._fault("odometry_sequence_reset")
            gap = stamp - self._previous[1]
            if not _finite(gap) or gap <= 0 or gap > self.max_gap_sec:
                return self._fault("odometry_time_discontinuity")
            if (not self._within_bounds(self._previous, sample)
                    or not self._within_bounds(self._initial, sample)):
                return self._fault("odometry_pose_discontinuity")
            while self._history and stamp - self._history[0][1] > self.history_window_sec:
                self._history.popleft()
            if len(self._history) >= 2048:
                return self._fault("odometry_history_capacity")
            if any(not self._within_bounds(prior, sample) for prior in self._history):
                return self._fault("odometry_pose_discontinuity")
        else:
            self._initial = sample
        self._previous = sample
        self._history.append(sample)
        return True


def sensors_aligned(stamps, odom_samples, now, max_sensor_age_sec, max_sensor_skew_sec,
                    max_odom_skew_sec, max_translation_m, max_rotation_rad,
                    max_linear_mps, max_angular_radps):
    """Check each scan/color/depth time against its nearest odometry pose.

    Stamps has exactly the monotonic keys scan, color and depth; each entry of
    odom_samples has monotonic stamp and planar pose. For every sensor pair, the
    allowed pose difference includes worst-case uncertainty due to each nearest
    odometry time gap at the supplied speed limits. This does not calibrate
    sensors or deskew individual scan rays.
    """
    try:
        limits = (max_sensor_age_sec, max_sensor_skew_sec, max_odom_skew_sec,
                  max_translation_m, max_rotation_rad, max_linear_mps, max_angular_radps)
        if (not _finite(now) or now < 0 or not all(_finite(value) for value in limits)
                or max_sensor_age_sec <= 0 or any(value < 0 for value in limits[1:])
                or not isinstance(stamps, dict) or set(stamps) != set(("scan", "color", "depth"))
                or not isinstance(odom_samples, (tuple, list))
                or not 1 <= len(odom_samples) <= 8192):
            return False
        times = [stamps[name] for name in ("scan", "color", "depth")]
        if (not all(_finite(stamp) and stamp >= 0 and 0 <= now - stamp <= max_sensor_age_sec
                    for stamp in times)
                or max(times) - min(times) > max_sensor_skew_sec):
            return False
        previous_stamp = None
        samples = []
        for sample in odom_samples:
            if not isinstance(sample, dict):
                return False
            stamp, pose = sample["stamp"], sample["pose"]
            if (not _finite(stamp) or not 0 <= stamp <= now or not _valid_pose(pose)
                    or (previous_stamp is not None and stamp <= previous_stamp)):
                return False
            samples.append((stamp, tuple(pose)))
            previous_stamp = stamp
        selected = []
        for stamp in times:
            closest = min(samples, key=lambda sample: abs(sample[0] - stamp))
            skew = abs(closest[0] - stamp)
            if not _finite(skew) or skew > max_odom_skew_sec:
                return False
            selected.append((closest[1], skew))
        for first in range(len(selected)):
            for second in range(first + 1, len(selected)):
                distance, rotation = _pose_delta(selected[first][0], selected[second][0])
                skew = selected[first][1] + selected[second][1]
                translation = distance + max_linear_mps * skew
                yaw = rotation + max_angular_radps * skew
                if (not all(_finite(value) for value in (translation, yaw))
                        or translation > max_translation_m or yaw > max_rotation_rad):
                    return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False
