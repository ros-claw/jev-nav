#!/usr/bin/env python2
"""Standalone, bounded, forward-only LIMO exploration candidate.

The Decisions API chooses the action; local code checks current sensor
samples and streams bounded commands through a separate relay. Observation
mode does not create a velocity publisher. The current implementation does
not execute turns, verify 3-D clearance or prove physical stopping.
"""

from __future__ import division, print_function

import argparse
import base64
import collections
import copy
import json
import math
import os
import sys
import threading
import time

from native_scan_loader import load_native_scan

try:
    import numpy as _np
except ImportError:
    _np = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_native_scan = load_native_scan()
WORKER = os.path.join(ROOT, "worker")
if not os.path.isdir(WORKER):
    WORKER = os.path.join(ROOT, "worktrees", "limo-mcp", "worker")
if WORKER not in sys.path:
    sys.path.insert(0, WORKER)

from limo_mapless_worker import run_decision_process, summarize_forward_view
import limo_scan_fusion as _fusion_module
from limo_scan_fusion import RollingScanFusion
from observation_guards import ClockGuard, OdomGuard, sensors_aligned
from policy import ForwardPolicy, assess_front
from projected_depth_gate import ProjectedDepthGate, project_depth_frame
from relay import _monotonic

# The worker may have been imported before the optional local accelerator was
# discoverable. Reuse the loader's verified binary regardless of import order.
_fusion_module._native_scan = _native_scan

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - ROS Melodic Python 2
except NameError:
    INTEGER_TYPES = (int,)
NUMBER_TYPES = INTEGER_TYPES + (float,)
LATEST_INPUT_MAX_AGE_SEC = 0.15


def _valid_epoch(value):
    return not isinstance(value, bool) and isinstance(value, INTEGER_TYPES) and value >= 0


def sensor_queue_size(name):
    """Keep short odometry backlogs for trajectory checks; use latest other data."""
    return 20 if name == "odom" else 1


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (ValueError, TypeError, OverflowError):
        return False


def _same_depth_intrinsics(first, second):
    """Compare exactly the camera fields consumed by depth projection."""
    if first is second:
        return True
    try:
        a, b = first['message'], second['message']
        return (a.header.frame_id == b.header.frame_id and a.width == b.width and a.height == b.height
                and tuple(a.K) == tuple(b.K))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def nearest_odometry(samples, stamp, max_skew=0.05):
    if not _finite(stamp) or not _finite(max_skew) or max_skew < 0:
        return None
    valid = [sample for sample in samples if isinstance(sample, dict) and _finite(sample.get("stamp"))]
    if not valid:
        return None
    closest = min(valid, key=lambda sample: abs(sample["stamp"] - stamp))
    return closest if abs(closest["stamp"] - stamp) <= max_skew else None


def refreshed_odometry(original, latest, now):
    try:
        if (not isinstance(original, dict) or not isinstance(latest, dict)
                or not all(_finite(v) for v in (original["stamp"], latest["stamp"], now))
                or latest["stamp"] < original["stamp"] or not 0 <= now - latest["stamp"] <= 0.1):
            return None
        _pose_delta(latest["pose"], latest["pose"])
        return latest
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def sample_clocks(monotonic, wall, max_window=0.01):
    """Discard pairs interrupted by scheduling instead of inventing a clock jump."""
    try:
        before = monotonic()
        ros_now = wall()
        after = monotonic()
        if (not all(_finite(value) for value in (before, ros_now, after, max_window))
                or max_window <= 0 or not 0 <= after - before <= max_window):
            return None
        return ros_now, before + (after - before) / 2.0
    except (TypeError, ValueError, OverflowError):
        return None


def source_time(ros_now, mono_now, sensor_stamp, max_age):
    """Map ROS wall time to the local monotonic clock without rejuvenating it."""
    if (not all(_finite(x) for x in (ros_now, mono_now, sensor_stamp, max_age))
            or max_age <= 0 or sensor_stamp <= 0):
        return None
    age = ros_now - sensor_stamp
    if not 0 <= age <= max_age:
        return None
    return mono_now - age


def depth_grid(array, encoding, stamp, min_depth_m=0.3, max_depth_m=3.0):
    """Summarize each depth-image region; these ratios don't establish clearance."""
    try:
        import numpy as np
        if (not _finite(stamp) or not _finite(min_depth_m) or not _finite(max_depth_m)
                or not 0 < min_depth_m < max_depth_m or len(array.shape) != 2
                or array.shape[0] < 3 or array.shape[1] < 3
                or encoding not in ("16UC1", "32FC1")):
            return None
        values = array.astype("float32")
        if encoding == "16UC1":
            values *= 0.001
        valid = np.isfinite(values) & (values >= min_depth_m) & (values <= max_depth_m)
        grid = []
        for row in np.array_split(valid, 3, axis=0):
            grid.append([float(np.count_nonzero(cell)) / float(cell.size)
                         for cell in np.array_split(row, 3, axis=1)])
        return {"stamp": stamp, "valid_grid": grid,
                "min_m": float(np.min(values[valid])) if np.any(valid) else None}
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def depth_summary_for_snapshot(array, encoding, stamp, projection_only):
    """Projected safety mode validates the depth array in its own gate."""
    if projection_only:
        return {"stamp": stamp} if _finite(stamp) else None
    return depth_grid(array, encoding, stamp)


def base_state(odom, status, stamp):
    try:
        twist = odom.twist.twist
        state = {"stamp": stamp, "motion_mode": int(status.motion_mode),
                 "error_code": int(status.error_code), "vehicle_state": int(status.vehicle_state),
                 "control_mode": int(status.control_mode), "battery_voltage": float(status.battery_voltage),
                 "linear_x_mps": float(twist.linear.x), "linear_y_mps": float(twist.linear.y),
                 "angular_z_radps": float(twist.angular.z)}
        if not all(_finite(value) for value in state.values()):
            return None
        return state
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def build_decision_evidence(sample, jpeg):
    try:
        stamps = [sample[name] for name in
                  ("scan_at", "odom_at", "status_at", "depth_at", "color_at")]
        if (not all(_finite(stamp) for stamp in stamps) or not isinstance(jpeg, bytes)
                or not jpeg.startswith(b"\xff\xd8") or len(jpeg) > 2000000):
            return None
        state = sample["state"]
        base = {name: state[name] for name in ("linear_x_mps", "angular_z_radps", "battery_voltage")}
        if not all(_finite(value) for value in base.values()):
            return None
        return {"image_b64": base64.b64encode(jpeg).decode("ascii"),
                "captured_at": min(stamps),
                "laser": {"front_min_m": None, "left_min_m": None,
                          "right_min_m": None, "rear_min_m": None,
                          "forward_view": sample["forward_view"]},
                "base": base}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def physical_forward_view(ranges, angle_min, increment, range_min, range_max, tx, ty, yaw):
    """Use the sensor's physical sector and the fusion no-return convention."""
    cleaned = []
    for index, distance in enumerate(ranges):
        angle = angle_min + index * increment
        if (-math.pi / 2 <= angle <= math.pi / 2 and _finite(distance)
                and range_min <= distance < range_max):
            cleaned.append(distance)
        else:
            cleaned.append(float("nan"))
    view = summarize_forward_view(cleaned, angle_min, increment, range_min, range_max, tx, ty, yaw)
    view["rear_total_rays"] = 0
    view["rear_valid_rays"] = 0
    view["rear_nearest_to_base_m"] = None
    return view


def transform_matches(first, second):
    if (not isinstance(first, (tuple, list)) or not isinstance(second, (tuple, list))
            or len(first) != 6 or len(second) != 6
            or not all(_finite(v) for v in tuple(first) + tuple(second))):
        return False
    if any(abs(first[i] - second[i]) > 0.001 for i in range(3)):
        return False
    return all(abs(math.atan2(math.sin(first[i] - second[i]),
                              math.cos(first[i] - second[i]))) <= 0.001
               for i in range(3, 6))


def lookup_rigid_transform(listener, tf_module, rospy_module, frame, stamp, known_signature):
    """Allow a nearby latest rigid transform only after a stamped baseline.

    Some ROS 1 camera TF publishers lag the image header slightly. The fallback
    uses the latest transform only within 150 ms of that header and only when
    it still matches a previously obtained timestamped rigid transform.
    """
    try:
        return listener.lookupTransform("base_link", frame, stamp)
    except tf_module.ExtrapolationException:
        if known_signature is None:
            raise
        common = listener.getLatestCommonTime("base_link", frame).to_sec()
        captured = stamp.to_sec()
        if (not all(_finite(value) and value > 0 for value in (common, captured))
                or abs(common - captured) > 0.15):
            raise ValueError("rigid transform timestamp is not nearby")
        trans, rot = listener.lookupTransform("base_link", frame, rospy_module.Time(0))
        roll, pitch, yaw = tf_module.transformations.euler_from_quaternion(rot)
        signature = tuple(trans) + (roll, pitch, yaw)
        if not transform_matches(known_signature, signature):
            raise ValueError("rigid transform changed")
        return trans, rot


def _fast_newest_history(rays, hits, crossings, now, scan_at, scene_pose, current_pose):
    """Vectorize the immutable ray/hit tuples created by the scan fusion."""
    if _np is None or len(rays) < 32:
        return None
    if _native_scan is not None:
        try:
            accelerated = _native_scan.fast_newest_history(
                rays, hits, crossings, now, scan_at, scene_pose, current_pose)
            if type(accelerated) is bool:
                return accelerated
        except Exception:  # noqa: BLE001 - retain the existing checked fallback
            pass
    rows = []
    for ray, hit in zip(rays, hits):
        if type(ray) is not dict or type(hit) is not dict:
            return None
        try:
            start, end, stamp = ray['start_base'], ray['hit_base'], ray['stamp']
            if (type(start) is not tuple or len(start) != 2 or type(end) is not tuple or len(end) != 2
                    or hit['point_base'] is not end or hit['stamp'] is not stamp or type(stamp) is not float
                    or any(type(value) is not float for value in start + end)):
                return None
            rows.append((start[0], start[1], end[0], end[1], stamp))
        except (KeyError, IndexError, TypeError):
            return None
    try:
        data = _np.asarray(rows, dtype=_np.float64)
        sx, sy, hx, hy, stamps = data.T
        if not _np.all(_np.isfinite(data)) or _np.any(stamps < 0) or _np.any(stamps > scan_at):
            return None
        theta = scene_pose[2] - current_pose[2]
        c, s = math.cos(theta), math.sin(theta)
        cc, cs = math.cos(current_pose[2]), math.sin(current_pose[2])
        dx, dy = scene_pose[0] - current_pose[0], scene_pose[1] - current_pose[1]
        tx, ty = cc * dx + cs * dy, -cs * dx + cc * dy
        start_x, start_y = tx + c * sx - s * sy, ty + s * sx + c * sy
        end_x, end_y = tx + c * hx - s * hy, ty + s * hx + c * hy
        if not all(_np.all(_np.isfinite(values)) for values in (start_x, start_y, end_x, end_y)):
            return None
        if _np.any((end_x >= -0.20) & (end_x <= 0.65) & (_np.abs(end_y) <= 0.17)):
            return False
        age = now - stamps
        recent = ((age >= 0) & (age <= 0.65) & (scan_at - stamps <= 0.4)
                  & (start_x <= 0.60) & (end_x > 0.60))
        indices = _np.flatnonzero(recent)
        with _np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            ys = start_y[indices] + ((0.60 - start_x[indices]) / (end_x[indices] - start_x[indices])) * (
                end_y[indices] - start_y[indices])
        if not _np.all(_np.isfinite(ys)):
            return None
        values = list(crossings) + ys[(ys >= -0.15) & (ys <= 0.15)].tolist()
        values.sort()
        gaps = [values[0] + 0.15, 0.15 - values[-1]]
        gaps.extend(values[index] - values[index - 1] for index in range(1, len(values)))
        return max(gaps) <= 0.035
    except (ArithmeticError, AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return None


def newest_scan_veto(scan, extrinsics, scene=None, now=None, scan_at=None,
                     scene_pose=None, scan_pose=None, current_pose=None):
    """Reject newly measured front hits; non-returns remain unknown.

    The latest scan must have bounded gaps and no measured hit in the padded
    corridor. Recent pose-aligned historical rays may fill the remaining small
    gaps, subject to the same age and spacing limits as the forward policy.
    Missing returns remain unknown and sampling does not certify clearance.
    """
    try:
        if (scan.header.frame_id.lstrip("/") != "laser_link"
                or not transform_matches(extrinsics, extrinsics)):
            return False
        angle_min, increment = scan.angle_min, scan.angle_increment
        lower, upper, ranges, time_increment = scan.range_min, scan.range_max, scan.ranges, scan.time_increment
        if (not all(_finite(item) for item in (angle_min, increment, lower, upper, time_increment))
                or increment == 0 or not 0 < lower < upper <= 100
                or time_increment < 0 or not isinstance(ranges, (tuple, list))
                or not 1 <= len(ranges) <= 8192 or (len(ranges) - 1) * time_increment > 0.3):
            return False
        angle_end = angle_min + (len(ranges) - 1) * increment
        if (not _finite(angle_end) or not -math.pi - 1e-5 <= angle_min <= math.pi + 1e-5
                or not -math.pi - 1e-5 <= angle_end <= math.pi + 1e-5):
            return False
        have_history = scene is not None
        if have_history:
            if (not _finite(now) or not _finite(scan_at) or not 0 <= now - scan_at <= 0.3
                    or not isinstance(scene, dict) or scene.get('status') != 'ok'
                    or not isinstance(scene.get('coverage'), dict)
                    or scene['coverage'].get('sensor_fresh') is not True):
                return False
            for pose in (scene_pose, scan_pose, current_pose):
                _pose_delta(pose, pose)
            if any(distance > 0.02 or rotation > 0.02 for distance, rotation in
                   (_pose_delta(scene_pose, current_pose), _pose_delta(scan_pose, current_pose))):
                return False
        else:
            scene_pose = scan_pose = current_pose = (0.0, 0.0, 0.0)

        def make_transform(source_pose):
            theta = source_pose[2] - current_pose[2]
            c, s = math.cos(theta), math.sin(theta)
            cc, cs = math.cos(current_pose[2]), math.sin(current_pose[2])
            dx, dy = source_pose[0] - current_pose[0], source_pose[1] - current_pose[1]
            tx, ty = cc * dx + cs * dy, -cs * dx + cc * dy
            return lambda point: (tx + c * point[0] - s * point[1],
                                  ty + s * point[0] + c * point[1])

        from_scan = make_transform(scan_pose)
        from_scene = make_transform(scene_pose)

        def crossing(start, end):
            if not start[0] <= 0.60 < end[0]:
                return None
            fraction = (0.60 - start[0]) / (end[0] - start[0])
            value = start[1] + fraction * (end[1] - start[1])
            if not _finite(value):
                raise ValueError('invalid ray crossing')
            return value if -0.15 <= value <= 0.15 else None

        def gap(values):
            if not values:
                return 0.3
            values = sorted(values)
            gaps = [values[0] + 0.15, 0.15 - values[-1]]
            gaps.extend(values[index] - values[index - 1] for index in range(1, len(values)))
            return max(gaps)

        accelerated = None
        if _native_scan is not None:
            try:
                candidate = _native_scan.fast_newest_raw(
                    ranges, angle_min, increment, lower, upper, extrinsics,
                    scan_pose, current_pose, math.pi / 2)
                if (isinstance(candidate, tuple) and len(candidate) == 2
                        and type(candidate[0]) is bool and isinstance(candidate[1], list)
                        and len(candidate[1]) <= len(ranges)
                        and all(type(value) is float and _finite(value) for value in candidate[1])):
                    accelerated = candidate
            except Exception:  # noqa: BLE001 - preserve checked scalar fallback
                pass
        if accelerated is not None:
            blocked, crossings = accelerated
            if blocked:
                return False
        else:
            start = from_scan((extrinsics[0], extrinsics[1]))
            crossings = []
            for index, value in enumerate(ranges):
                if isinstance(value, bool) or not isinstance(value, NUMBER_TYPES):
                    return False
                if not _finite(value) or not lower <= value < upper:
                    continue
                angle = angle_min + index * increment + extrinsics[5]
                if not _finite(angle):
                    return False
                angle = math.atan2(math.sin(angle), math.cos(angle))
                if not -math.pi / 2 <= angle <= math.pi / 2:
                    continue
                x = extrinsics[0] + value * math.cos(angle)
                y = extrinsics[1] + value * math.sin(angle)
                x, y = from_scan((x, y))
                if -0.20 <= x <= 0.65 and abs(y) <= 0.17:
                    return False
                value = crossing(start, (x, y))
                if value is not None:
                    crossings.append(value)
        if not crossings or gap(crossings) > 0.07:
            return False
        if not have_history:
            return gap(crossings) <= 0.035
        rays, hits = scene['rays'], scene['occupied_hits']
        if (not isinstance(rays, list) or not isinstance(hits, list)
                or len(rays) != len(hits) or len(rays) > 200000):
            return False
        fast = _fast_newest_history(rays, hits, crossings, now, scan_at, scene_pose, current_pose)
        if fast is not None:
            return fast
        for ray, hit in zip(rays, hits):
            start_old, end_old, stamp = ray['start_base'], ray['hit_base'], ray['stamp']
            point, hit_stamp = hit['point_base'], hit['stamp']
            if (not isinstance(start_old, (tuple, list)) or len(start_old) != 2
                    or not isinstance(end_old, (tuple, list)) or len(end_old) != 2
                    or not isinstance(point, (tuple, list)) or len(point) != 2
                    or not all(_finite(item) for item in tuple(start_old) + tuple(end_old) + tuple(point)
                               + (stamp, hit_stamp))
                    or point[0] != end_old[0] or point[1] != end_old[1]
                    or hit_stamp != stamp or not 0 <= stamp <= scan_at):
                return False
            end = from_scene(end_old)
            if -0.20 <= end[0] <= 0.65 and abs(end[1]) <= 0.17:
                return False
            if not 0 <= now - stamp <= 0.65 or scan_at - stamp > 0.4:
                continue
            value = crossing(from_scene(start_old), end)
            if value is not None:
                crossings.append(value)
        return gap(crossings) <= 0.035
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return False


def _pose_delta(first, second):
    if (not isinstance(first, (tuple, list)) or not isinstance(second, (tuple, list))
            or len(first) != 3 or len(second) != 3
            or not all(_finite(v) for v in tuple(first) + tuple(second))):
        raise ValueError("invalid pose")
    distance = math.hypot(second[0] - first[0], second[1] - first[1])
    yaw = second[2] - first[2]
    return distance, abs(math.atan2(math.sin(yaw), math.cos(yaw)))


def snapshot_is_fresh(sample, now, live_pose, max_age_sec=0.1):
    """Recheck pose freshness after potentially slow processing."""
    try:
        if (not isinstance(sample, dict) or not all(_finite(v) for v in
                (now, sample.get("snapshot_at"), sample.get("odom_at"), max_age_sec))
                or not 0 < max_age_sec <= 0.15
                or not 0 <= now - sample["snapshot_at"] <= max_age_sec
                or not 0 <= now - sample["odom_at"] <= max_age_sec):
            return False
        distance, rotation = _pose_delta(sample["pose"], live_pose)
        return distance <= 0.01 and rotation <= 0.03
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


class DecisionGate(object):  # noqa: UP004 - ROS Melodic Python 2
    def __init__(self):
        self.choice = "stop"
        self.stamp = None
        self.pose = None
        self.valid = False

    def accept(self, result, captured_at, observed_pose, now, current_pose):
        if (not isinstance(result, dict) or result.get("ok") is not True
                or result.get("choice") not in ("forward", "left", "right", "stop")
                or not _finite(captured_at) or not _finite(now)
                or result.get("captured_at") != captured_at
                or not 0 <= now - captured_at < 8.0
                or (self.stamp is not None and captured_at <= self.stamp)):
            return False
        if result["choice"] != "stop":
            try:
                distance, rotation = _pose_delta(observed_pose, current_pose)
            except (TypeError, ValueError):
                return False
            if distance > 0.15 or rotation > 0.35:
                return False
        self.choice = result["choice"]
        self.stamp = captured_at
        self.pose = observed_pose
        self.valid = True
        return True

    def invalidate(self):
        self.valid = False

    def current(self, now, pose):
        if (not self.valid or not _finite(now) or self.stamp is None
                or not 0 <= now - self.stamp < 8.0):
            self.invalidate()
            return None
        if self.choice != "stop":
            try:
                distance, rotation = _pose_delta(self.pose, pose)
            except (TypeError, ValueError):
                self.invalidate()
                return None
            if distance > 0.15 or rotation > 0.35:
                self.invalidate()
                return None
        return {"choice": self.choice, "stamp": self.stamp}


class RosObservation(object):  # noqa: UP004 - ROS Melodic Python 2
    """Read ROS sensors directly; this object never creates a velocity publisher."""

    def __init__(self, depth_projection_config=None, sensor_names=None):
        import rospy
        import tf
        from cv_bridge import CvBridge
        from limo_base.msg import LimoStatus
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image, LaserScan

        self.rospy, self.tf, self.bridge = rospy, tf, CvBridge()
        self.listener = tf.TransformListener()
        self.lock = threading.Lock()
        self.latest = {}
        self.odom_history = collections.deque(maxlen=600)
        self.fusion = None
        self.extrinsics = None
        self.scan_stamp = None
        self.depth_projection_config = (dict(depth_projection_config)
                                        if depth_projection_config is not None else None)
        self.depth_extrinsics = None
        self._depth_projections = collections.deque(maxlen=4)
        self._scan_veto_cache = []
        self.projection_only = False
        self.epoch = 0
        self.epoch_issue = None
        self.fusion_epoch = None
        self.snapshot_last_issue = None
        self.current_last_issue = None
        self.clock_guard = ClockGuard(max_offset_jump_sec=0.02)
        self.odom_guard = OdomGuard(max_linear_mps=0.12, max_angular_radps=0.1,
                                    position_slack_m=0.002, yaw_slack_rad=0.003,
                                    max_gap_sec=0.3)
        self.limits = {"scan": 0.5, "odom": 0.25, "status": 1.5,
                       "depth": 0.8, "color": 0.8, "info": 1.0}
        topics = [("scan", "/scan", LaserScan), ("odom", "/odom", Odometry),
                  ("status", "/limo_status", LimoStatus),
                  ("depth", "/camera/depth/image_raw", Image),
                  ("color", "/camera/color/image_raw", Image),
                  ("info", "/camera/depth/camera_info", CameraInfo)]
        if sensor_names is not None:
            names = set(sensor_names)
            if "odom" not in names or not names.issubset(name for name, _, _ in topics):
                raise ValueError("invalid observation sensor selection")
            topics = [spec for spec in topics if spec[0] in names]
        self.subscriptions = [rospy.Subscriber(topic, typ, self._receive, callback_args=name,
                                               queue_size=sensor_queue_size(name))
                              for name, topic, typ in topics]

    def _receive(self, message, name):
        try:
            with self.lock:
                pair = sample_clocks(_monotonic, lambda: self.rospy.Time.now().to_sec())
                if pair is None:
                    return
                ros_now, received = pair
                if not self.clock_guard.observe(ros_now, received):
                    self.latest.clear()
                    self.epoch += 1
                    self.epoch_issue = {'reason': 'clock_fault', 'sensor': name}
                    return
                header_stamp = message.header.stamp.to_sec()
                stamp = source_time(ros_now, received, header_stamp, self.limits[name])
                if stamp is None:
                    self.latest.pop(name, None)
                    self.epoch += 1
                    self.epoch_issue = {'reason': 'source_time_invalid', 'sensor': name}
                    if _finite(header_stamp) and _finite(ros_now):
                        age = ros_now - header_stamp
                        if _finite(age):
                            self.epoch_issue['sensor_age_sec'] = age
                    return
                # Keep the receipt in the same monotonic/ROS clock pairing as
                # the acquisition stamp; it must never predate that stamp.
                entry = {"message": message, "stamp": stamp, "received_at": received}
                if name == "odom":
                    if (message.header.frame_id.lstrip("/") != "odom"
                            or message.child_frame_id.lstrip("/") != "base_link"):
                        self.latest.pop(name, None)
                        self.epoch += 1
                        self.epoch_issue = {'reason': 'odom_frame_invalid', 'sensor': name}
                        return
                    p = message.pose.pose.position
                    q = message.pose.pose.orientation
                    quaternion = (q.x, q.y, q.z, q.w)
                    if (not all(_finite(x) for x in (p.x, p.y) + quaternion)
                            or not 0.99 <= sum(x * x for x in quaternion) <= 1.01):
                        self.latest.pop(name, None)
                        self.epoch += 1
                        self.epoch_issue = {'reason': 'odom_pose_invalid', 'sensor': name}
                        return
                    roll, pitch, yaw = self.tf.transformations.euler_from_quaternion(quaternion)
                    if abs(roll) > 0.1 or abs(pitch) > 0.1:
                        self.latest.pop(name, None)
                        self.epoch += 1
                        self.epoch_issue = {'reason': 'odom_tilt_invalid', 'sensor': name}
                        return
                    entry["pose"] = (p.x, p.y, yaw)
                    if not self.odom_guard.observe(message.header.seq, stamp, entry["pose"]):
                        self.latest.clear()
                        self.odom_history.clear()
                        self.epoch += 1
                        self.epoch_issue = {'reason': 'odom_discontinuity', 'sensor': name}
                        return
                previous = self.latest.get(name)
                if previous is not None and stamp <= previous["stamp"]:
                    self.latest.pop(name, None)
                    self.epoch += 1
                    self.epoch_issue = {'reason': 'sensor_time_regression', 'sensor': name}
                    return
                self.latest[name] = entry
                if name == "odom":
                    entry["epoch"] = self.epoch
                    self.odom_history.append(entry)
        except (AttributeError, TypeError, ValueError, OverflowError):
            with self.lock:
                self.latest.pop(name, None)
                self.epoch += 1
                self.epoch_issue = {'reason': 'callback_invalid', 'sensor': name}
            return

    def snapshot(self):
        """Take aligned observations, independently of model/network activity."""
        def reject(reason):
            self.snapshot_last_issue = reason
            return None

        with self.lock:
            if self.clock_guard.faulted or self.odom_guard.faulted:
                return reject("guard_fault")
            current = dict(self.latest)
            odom_history = list(self.odom_history)[-150:]
            epoch = self.epoch
            projection_only = getattr(self, "projection_only", False)
        now = _monotonic()
        for name in self.limits:
            if name not in current:
                return reject("missing_" + name)
            if not 0 <= now - current[name]["stamp"] <= self.limits[name]:
                return reject("stale_" + name)
        if not sensors_aligned({name: current[name]["stamp"] for name in ("scan", "color", "depth")},
                               odom_history, now, max_sensor_age_sec=0.5,
                               max_sensor_skew_sec=0.25, max_odom_skew_sec=0.05,
                               max_translation_m=0.02, max_rotation_rad=0.02,
                               max_linear_mps=0.12, max_angular_radps=0.1):
            return reject("sensor_alignment")
        scan = current["scan"]["message"]
        if scan.header.frame_id.lstrip("/") != "laser_link":
            return reject("scan_frame")
        status = current["status"]["message"]
        depth = current["depth"]["message"]
        info = current["info"]["message"]
        color = current["color"]["message"]
        if (depth.header.frame_id.lstrip("/") != "camera_depth_optical_frame"
                or color.header.frame_id.lstrip("/") != "camera_color_optical_frame"
                or depth.header.frame_id != info.header.frame_id
                or depth.width != info.width or depth.height != info.height
                or abs(current["depth"]["stamp"] - current["info"]["stamp"]) > 0.2):
            return reject("camera_alignment")
        try:
            trans, rot = self.listener.lookupTransform("base_link", scan.header.frame_id,
                                                       scan.header.stamp)
            roll, pitch, yaw = self.tf.transformations.euler_from_quaternion(rot)
            extrinsics = (trans[0], trans[1], yaw)
            signature = (trans[0], trans[1], trans[2], roll, pitch, yaw)
            if (not all(_finite(x) for x in signature)
                    or abs(roll) > 0.1 or abs(pitch) > 0.1):
                return reject("laser_transform_invalid")
            if self.fusion is None or self.fusion_epoch != epoch:
                self.extrinsics = signature
                self.fusion = RollingScanFusion(extrinsics)
                self.fusion_epoch = epoch
                self.scan_stamp = None
            elif not transform_matches(self.extrinsics, signature):
                # Never continue across an extrinsic change without a restart.
                return reject("laser_transform_changed")
            if current["scan"]["stamp"] != self.scan_stamp:
                aligned = nearest_odometry(odom_history, current["scan"]["stamp"])
                if aligned is None or aligned.get("epoch") != epoch:
                    return reject("scan_odom_alignment")
                packet = {"stamp": current["scan"]["stamp"], "ranges": list(scan.ranges),
                          "angle_min": scan.angle_min, "angle_increment": scan.angle_increment,
                          "range_min": scan.range_min, "range_max": scan.range_max}
                if not self.fusion.add_scan(packet, aligned["pose"], aligned["stamp"], now):
                    return reject("fusion_add")
                self.scan_stamp = current["scan"]["stamp"]
            depth_array = (None if projection_only else
                           self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough"))
            depth_summary = depth_summary_for_snapshot(
                depth_array, depth.encoding, current["depth"]["stamp"], projection_only)
            if depth_summary is None:
                return reject("depth_grid")
            if self.depth_projection_config is not None:
                depth_trans, depth_rot = lookup_rigid_transform(
                    self.listener, self.tf, self.rospy, depth.header.frame_id,
                    depth.header.stamp, self.depth_extrinsics)
                depth_roll, depth_pitch, depth_yaw = self.tf.transformations.euler_from_quaternion(depth_rot)
                depth_signature = tuple(depth_trans) + (depth_roll, depth_pitch, depth_yaw)
                if self.depth_extrinsics is None:
                    if not transform_matches(depth_signature, depth_signature):
                        return reject("depth_transform_invalid")
                    self.depth_extrinsics = depth_signature
                elif not transform_matches(self.depth_extrinsics, depth_signature):
                    return reject("depth_transform_changed")
                depth_summary["corridor"] = self._depth_projection(
                    depth, info, current["depth"]["stamp"], epoch, depth_trans, depth_rot, depth_array)
            image_pose = nearest_odometry(odom_history, current["color"]["stamp"])
            depth_pose = nearest_odometry(odom_history, current["depth"]["stamp"])
            if (image_pose is None or image_pose.get("epoch") != epoch
                    or depth_pose is None or depth_pose.get("epoch") != epoch):
                return reject("camera_odom_alignment")
            with self.lock:
                if self.epoch != epoch or self.clock_guard.faulted or self.odom_guard.faulted:
                    return reject("epoch_changed")
                snapshot_now = _monotonic()
                latest_odom = refreshed_odometry(current["odom"], self.latest.get("odom"), snapshot_now)
            if latest_odom is None:
                return reject("refreshed_odom")
            pose = latest_odom["pose"]
            scene = self.fusion.snapshot(pose, latest_odom["stamp"], snapshot_now)
            odom = latest_odom["message"]
            state = base_state(odom, status, min(latest_odom["stamp"], current["status"]["stamp"]))
            if state is None:
                return reject("base_state")
            with self.lock:
                if self.epoch != epoch or self.clock_guard.faulted or self.odom_guard.faulted:
                    return reject("epoch_changed")
            self.snapshot_last_issue = None
            return {"scene": scene, "state": state, "depth": depth_summary, "snapshot_at": snapshot_now, "epoch": epoch,
                    "pose": pose, "image_pose": image_pose["pose"], "depth_pose": depth_pose["pose"],
                    "color": current["color"]["message"], "scan": scan,
                    "laser_extrinsics": extrinsics,
                    "scan_at": current["scan"]["stamp"], "odom_at": latest_odom["stamp"],
                    "status_at": current["status"]["stamp"], "depth_at": current["depth"]["stamp"],
                    "color_at": current["color"]["stamp"]}
        except Exception as exc:  # noqa: BLE001 - missing/invalid observation cannot authorize motion
            # Record only the exception type, never sensor values or exception text.
            return reject("snapshot_exception_" + type(exc).__name__)

    def enable_projection_only(self):
        """Skip image-grid statistics only when a projected-depth gate is used."""
        with self.lock:
            if self.depth_projection_config is None:
                raise ValueError("depth projection is not configured")
            self.projection_only = True

    def disable_projection_only(self):
        with self.lock:
            self.projection_only = False

    def current_pose(self):
        with self.lock:
            odom = self.latest.get("odom")
            if self.clock_guard.faulted or self.odom_guard.faulted:
                return None
        if odom is None or not 0 <= _monotonic() - odom["stamp"] <= 0.1:
            return None
        return odom["pose"]

    def break_epoch(self):
        with self.lock:
            self.epoch += 1
            self.epoch_issue = {'reason': 'explicit_epoch_break', 'sensor': 'unknown'}

    def is_current_epoch(self, epoch):
        with self.lock:
            return (_valid_epoch(epoch) and epoch == self.epoch and not self.clock_guard.faulted
                    and not self.odom_guard.faulted)

    def _depth_projection(self, image, info, stamp, epoch, trans, rotation, array=None):
        """Reuse geometry only; callers still check age, TF and aligned pose."""
        k = info.K
        key = (epoch, stamp, image.encoding, image.width, image.height, image.header.frame_id,
               image.step, image.is_bigendian,
               tuple(k), tuple(trans), tuple(rotation), tuple(sorted(self.depth_projection_config.items())))
        cache = getattr(self, '_depth_projections', None)
        data = image.data
        if type(data) is not bytes:
            cache = None
        if cache is not None:
            for old_image, old_data, old_key, projection in list(cache):
                if old_image is image and old_data is data and old_key == key:
                    return copy.deepcopy(projection)
        matrix = self.tf.transformations.quaternion_matrix(rotation)
        matrix[:3, 3] = trans
        intrinsics = {'width': image.width, 'height': image.height, 'fx': k[0], 'fy': k[4],
                      'cx': k[2], 'cy': k[5], 'rectified': True}
        if array is None:
            array = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        projection = project_depth_frame(array, image.encoding, intrinsics, matrix, stamp,
                                         self.depth_projection_config)
        if cache is not None:
            cache.appendleft((image, data, key, copy.deepcopy(projection)))
        return projection

    def _newest_depth_safe_locked(self, sample, depth, odom, now):
        """Project a newer depth frame before using an older fused snapshot.

        The caller holds the observation lock so this check is tied to the
        latest depth generation. It remains a sampled veto, not a clearance
        certificate, and retains the configured unverified calibration.
        """
        try:
            if self.depth_projection_config is None or not transform_matches(
                    self.depth_extrinsics, self.depth_extrinsics):
                return False
            stamp, image = depth['stamp'], depth['message']
            info_entry = self.latest['info']
            info = info_entry['message']
            if (not _finite(stamp) or not 0 <= now - stamp <= LATEST_INPUT_MAX_AGE_SEC
                    or not _finite(info_entry['stamp']) or abs(stamp - info_entry['stamp']) > 0.2
                    or image.header.frame_id.lstrip('/') != 'camera_depth_optical_frame'
                    or image.header.frame_id != info.header.frame_id
                    or image.width != info.width or image.height != info.height):
                return False
            aligned = nearest_odometry(self.odom_history, stamp)
            if aligned is None or aligned.get('epoch') != self.epoch:
                return False
            for pose in (sample['pose'], odom['pose']):
                distance, angle = _pose_delta(aligned['pose'], pose)
                if not _finite(distance) or not _finite(angle) or distance > 0.01 or angle > 0.02:
                    return False
            trans = self.depth_extrinsics[:3]
            rotation = self.tf.transformations.quaternion_from_euler(*self.depth_extrinsics[3:])
            corridor = self._depth_projection(image, info, stamp, self.epoch, trans, rotation)
            candidate = {'depth_at': stamp, 'depth': {'stamp': stamp, 'corridor': corridor}}
            return ProjectedDepthGate().assess(candidate, now).get('candidate') is True
        except Exception:  # noqa: BLE001 - an unprocessed depth frame must veto
            return False

    def _cached_newest_scan_safe_locked(self, sample, scan, aligned, odom, now):
        """Reuse a scan assessment only for the same owned scene and poses."""
        try:
            scene = sample['scene']
            message = scan['message']
            rays, hits = scene['rays'], scene['occupied_hits']
            if (not isinstance(rays, list) or not isinstance(hits, list)
                    or len(rays) != len(hits) or len(rays) > 200000):
                return False
            scene_signature = tuple((tuple(ray['start_base']), tuple(ray['hit_base']), ray['stamp'],
                                     tuple(hit['point_base']), hit['stamp']) for ray, hit in zip(rays, hits))
            key = (self.epoch, scan['stamp'], tuple(sample['pose']), tuple(aligned['pose']),
                   tuple(odom['pose']), tuple(self.extrinsics), message.header.frame_id,
                   message.angle_min, message.angle_increment,
                   message.range_min, message.range_max, message.time_increment, tuple(message.ranges),
                   scene.get('status'), scene.get('coverage', {}).get('sensor_fresh'), scene_signature)
            entries = getattr(self, '_scan_veto_cache', [])
            for cached_scene, cached_key, expires, accepted in entries:
                if cached_scene is scene and cached_key == key and now < expires:
                    self._scan_veto_valid_until = expires if accepted else None
                    return accepted
            accepted = newest_scan_veto(message, self.extrinsics, scene, now, scan['stamp'],
                                        sample['pose'], aligned['pose'], odom['pose'])
            expiry = min(now + 0.1, scan['stamp'] + 0.3)
            if accepted:
                stamps = [ray['stamp'] for ray in scene['rays'] if _finite(ray.get('stamp'))
                          and 0 <= now - ray['stamp'] <= 0.65 and 0 <= scan['stamp'] - ray['stamp'] <= 0.4]
                if stamps:
                    conservative_expiry = min(expiry, min(stamps) + 0.65)
                    # Some old rays do not contribute to coverage. Only use a
                    # longer lifetime if the same checker still passes after
                    # all rays expiring before that time have been removed.
                    future = expiry - 1e-6
                    if (future > conservative_expiry + .015
                            and newest_scan_veto(message, self.extrinsics, scene, future, scan['stamp'],
                                                sample['pose'], aligned['pose'], odom['pose'])):
                        expiry = future
                    else:
                        expiry = conservative_expiry
            self._scan_veto_cache = (entries + [(scene, key, expiry, accepted)])[-4:]
            self._scan_veto_valid_until = expiry if accepted else None
            return accepted
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return False

    def is_current_sample(self, sample):
        return self.current_valid_until(sample) is not None

    def current_valid_until(self, sample):
        """Retry only replaced sensor generations, within the original lifetime."""
        for attempt in range(3):
            failure = [None]
            result = self._current_valid_until_once(sample, failure)
            if result is not None:
                now = _monotonic()
                stamp = sample.get('snapshot_at') if isinstance(sample, dict) else None
                if (not _finite(now) or not _finite(stamp) or not _finite(result)
                        or not 0 <= now - stamp < .25):
                    self.current_last_issue = 'timestamps'
                    return None
                result = min(result, stamp + .25)
                if result <= now:
                    self.current_last_issue = 'timestamps'
                    return None
                self.current_last_issue = None
                return result
            reason = failure[0] or 'sensor_unavailable'
            self.current_last_issue = reason
            if reason not in ('scan_generation_changed', 'depth_generation_changed') or attempt == 2:
                return None
            now = _monotonic()
            stamp = sample.get('snapshot_at') if isinstance(sample, dict) else None
            if not _finite(now) or not _finite(stamp) or not 0 <= now - stamp < .25:
                return None
        return None

    def prepare_command_evidence(self, sample):
        """Build the same final proof on the perception worker."""
        failure, evidence = [None], {}
        deadline = self._current_valid_until_once(sample, failure, evidence)
        return (evidence, None) if deadline is not None else (None, failure[0])

    def read_command_evidence(self, sample, evidence):
        """Recheck freshness and continuity without projecting a sensor frame."""
        try:
            with self.lock:
                now = _monotonic()
                if (sample['epoch'] != self.epoch or self.clock_guard.faulted or self.odom_guard.faulted):
                    return None, 'epoch_or_guard'
                if evidence is None:
                    return None, 'prepared_pending'
                if self.epoch != evidence['epoch']:
                    return None, 'epoch_or_guard'
                if self.latest.get('scan') is not evidence['scan']:
                    return None, 'prepared_scan_advanced'
                if self.latest.get('depth') is not evidence['depth']:
                    return None, 'prepared_depth_advanced'
                if not _same_depth_intrinsics(self.latest.get('info'), evidence['info']):
                    return None, 'camera_info_changed'
                if (self.extrinsics != evidence['extrinsics'] or self.depth_extrinsics != evidence['depth_extrinsics']
                        or self.depth_projection_config != evidence['config']):
                    return None, 'transform_changed'
                if not 0 <= now - evidence['checked_at'] or now >= evidence['valid_until']:
                    return None, 'prepared_pending'
                odom, status = self.latest['odom'], self.latest['status']
                if (odom['stamp'] < evidence['odom_at'] or status['stamp'] < evidence['status_at']
                        or not 0 <= now - odom['stamp'] <= LATEST_INPUT_MAX_AGE_SEC
                        or not 0 <= now - status['stamp'] <= 1.0):
                    return None, 'stale_latest_state'
                for pose in evidence['poses']:
                    distance, turn = _pose_delta(pose, odom['pose'])
                    if not _finite(distance) or not _finite(turn) or distance > .005 or turn > .01:
                        return None, 'pose_changed'
                state = base_state(odom['message'], status['message'], now)
                if (state is None or state['motion_mode'] != 0 or state['error_code'] != 0
                        or state['vehicle_state'] != 0 or state['control_mode'] != 1
                        or state['battery_voltage'] < 10 or not 0 <= state['linear_x_mps'] <= .02
                        or abs(state['linear_y_mps']) > .005 or abs(state['angular_z_radps']) > .02):
                    return None, 'base_state'
                return evidence['valid_until'], None
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return None, 'malformed'

    def _current_valid_until_once(self, sample, failure, evidence=None):
        """Validate a captured generation without blocking sensor callbacks.

        Heavy checks use an owned view of the input records. New laser or
        depth data invalidates that view; newer odometry and status are
        independently checked before the result is returned.
        """
        def reject(reason):
            failure[0] = reason
            self.current_last_issue = reason
            return None

        try:
            with self.lock:
                probe = copy.copy(self)
                probe.lock = threading.Lock()
                probe.latest = dict(self.latest)
                probe.odom_history = list(getattr(self, 'odom_history', []))
                probe.clock_guard = copy.copy(self.clock_guard)
                probe.odom_guard = copy.copy(self.odom_guard)
                probe.defer_depth_validation = True
                config = getattr(self, 'depth_projection_config', None)
                probe.depth_projection_config = dict(config) if isinstance(config, dict) else config
                captured = probe.latest
            # Only this private copy takes the lock during expensive work.
            deadline = probe._captured_valid_until(sample)
            if deadline is None:
                return reject(getattr(probe, 'current_last_issue', 'sensor_unavailable'))
            # Recheck one newer scan without blocking callbacks. A still
            # newer generation at the final check invalidates this result.
            with self.lock:
                if self.epoch != probe.epoch or self.clock_guard.faulted or self.odom_guard.faulted:
                    return reject('epoch_or_guard')
                selected_scan = self.latest.get('scan')
                scan_odom = self.latest.get('odom')
                if selected_scan is not captured.get('scan'):
                    if (not isinstance(selected_scan, dict) or not _finite(selected_scan.get('stamp'))
                            or selected_scan['stamp'] <= captured['scan']['stamp']):
                        return reject('scan_timestamp_regression')
                    probe.latest = dict(self.latest)
                    probe.odom_history = list(getattr(self, 'odom_history', []))
            if selected_scan is not captured.get('scan'):
                scan_checked_at = _monotonic()
                aligned = nearest_odometry(probe.odom_history, selected_scan['stamp'])
                if (not _finite(scan_checked_at) or scan_checked_at < probe.validation_checked_at
                        or not 0 <= scan_checked_at - selected_scan['stamp'] <= .3
                        or aligned is None or aligned.get('epoch') != probe.epoch
                        or not probe._cached_newest_scan_safe_locked(
                            sample, selected_scan, aligned, scan_odom, scan_checked_at)):
                    return reject('new_scan')
                scan_until = getattr(probe, '_scan_veto_valid_until', None)
                if not _finite(scan_until):
                    return reject('new_scan_expiry')
                probe.validation_checked_at = scan_checked_at
                limits = dict(probe.validation_limits)
                limits['latest_scan'] = selected_scan['stamp'] + .3
                limits['new_scan_veto'] = scan_until
                probe.validation_limits = limits
                deadline = min(limits.values())
            # Retry just the depth stage once if a frame arrives while it is
            # projected. A changed scan or epoch still invalidates the check.
            previous_depth = None
            for _ in range(2):
                with self.lock:
                    if self.epoch != probe.epoch or self.clock_guard.faulted or self.odom_guard.faulted:
                        return reject('epoch_or_guard')
                    if self.latest.get('scan') is not selected_scan:
                        return reject('scan_generation_changed')
                    selected_depth = self.latest.get('depth')
                    if (previous_depth is not None and
                            (not isinstance(selected_depth, dict)
                             or not _finite(selected_depth.get('stamp'))
                             or selected_depth['stamp'] <= previous_depth['stamp'])):
                        return reject('depth_timestamp_regression')
                    selected_info = self.latest.get('info')
                    depth_odom = self.latest.get('odom')
                    if selected_depth is captured.get('depth'):
                        selected_info = captured.get('info')
                    if selected_depth is not captured.get('depth'):
                        if (not isinstance(selected_depth, dict) or not _finite(selected_depth.get('stamp'))
                                or selected_depth['stamp'] <= captured['depth']['stamp']):
                            return reject('depth_timestamp_regression')
                        probe.latest = dict(self.latest)
                        probe.odom_history = list(getattr(self, 'odom_history', []))
                if selected_depth['stamp'] != sample['depth_at']:
                    depth_checked_at = _monotonic()
                    if (not _finite(depth_checked_at) or depth_checked_at < probe.validation_checked_at
                            or not 0 <= depth_checked_at - sample['snapshot_at'] < .25
                            or not probe._newest_depth_safe_locked(sample, selected_depth, depth_odom, depth_checked_at)):
                        return reject('new_depth_unchecked')
                    probe.validation_checked_at = depth_checked_at
                    limits = dict(probe.validation_limits)
                    limits['latest_depth'] = selected_depth['stamp'] + LATEST_INPUT_MAX_AGE_SEC
                    probe.validation_limits = limits
                    deadline = min(limits.values())
                with self.lock:
                    if self.latest.get('depth') is selected_depth:
                        break
                    previous_depth = selected_depth
            else:
                return reject('depth_replaced_during_retry')
            with self.lock:
                now = _monotonic()
                if (not _finite(now) or now < probe.validation_checked_at
                        or self.epoch != probe.epoch or self.clock_guard.faulted or self.odom_guard.faulted):
                    return reject('epoch_or_guard')
                if self.latest.get('scan') is not selected_scan:
                    return reject('scan_generation_changed')
                if self.latest.get('depth') is not selected_depth:
                    return reject('depth_generation_changed')
                # Camera calibration is used when validating a newer frame.
                if (selected_depth['stamp'] != sample['depth_at']
                        and not _same_depth_intrinsics(self.latest.get('info'), selected_info)):
                    return reject('camera_info_changed')
                if (getattr(self, 'extrinsics', None) != getattr(probe, 'extrinsics', None)
                        or getattr(self, 'depth_extrinsics', None) != getattr(probe, 'depth_extrinsics', None)
                        or getattr(self, 'depth_projection_config', None) != probe.depth_projection_config):
                    return reject('transform_changed')
                odom, status = self.latest['odom'], self.latest['status']
                if (not _finite(odom.get('stamp')) or odom['stamp'] < captured['odom']['stamp']
                        or not 0 <= now - odom['stamp'] <= LATEST_INPUT_MAX_AGE_SEC
                        or not _finite(status.get('stamp')) or status['stamp'] < captured['status']['stamp']
                        or not 0 <= now - status['stamp'] <= 1.0):
                    return reject('stale_latest_state')
                distance, rotation = _pose_delta(sample['pose'], odom['pose'])
                odom_distance, odom_rotation = _pose_delta(captured['odom']['pose'], odom['pose'])
                scan_distance, scan_rotation = _pose_delta(scan_odom['pose'], odom['pose'])
                depth_distance, depth_rotation = _pose_delta(depth_odom['pose'], odom['pose'])
                if (not all(_finite(value) for value in (distance, rotation, odom_distance, odom_rotation,
                                                          scan_distance, scan_rotation, depth_distance,
                                                          depth_rotation))
                        or max(distance, odom_distance, scan_distance, depth_distance) > .005
                        or max(rotation, odom_rotation, scan_rotation, depth_rotation) > .01):
                    return reject('pose_changed')
                state = base_state(odom['message'], status['message'], now)
                if (state is None or state['motion_mode'] != 0 or state['error_code'] != 0
                        or state['vehicle_state'] != 0 or state['control_mode'] != 1
                        or state['battery_voltage'] < 10):
                    return reject('base_state')
                if (not 0 <= state['linear_x_mps'] <= .02 or abs(state['linear_y_mps']) > .005
                        or abs(state['angular_z_radps']) > .02):
                    return reject('base_speed')
                limits = {'captured_sensors': deadline, 'latest_odom': odom['stamp'] + LATEST_INPUT_MAX_AGE_SEC,
                          'latest_status': status['stamp'] + 1.0}
                limit_kind = min(limits, key=limits.get)
                result = limits[limit_kind]
                self.current_limit_kind = (min(probe.validation_limits, key=probe.validation_limits.get)
                                           if limit_kind == 'captured_sensors' else limit_kind)
                if not _finite(result) or result <= now:
                    return reject('timestamps')
                if evidence is not None:
                    evidence.update({
                        'epoch': self.epoch, 'scan': selected_scan, 'depth': selected_depth,
                        'info': self.latest.get('info'), 'odom_at': odom['stamp'], 'status_at': status['stamp'],
                        'poses': (sample['pose'], captured['odom']['pose'], scan_odom['pose'], depth_odom['pose']),
                        'valid_until': result, 'checked_at': now,
                        'extrinsics': probe.extrinsics, 'depth_extrinsics': probe.depth_extrinsics,
                        'config': probe.depth_projection_config,
                    })
                # Only reuse caches from a view which survived the generation,
                # pose and state rechecks above; their keys and expiries are
                # still checked again on every later lookup.
                self._scan_veto_cache = getattr(probe, '_scan_veto_cache', [])
                self.current_last_issue = None
                return result
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return reject('malformed')

    def _captured_valid_until(self, sample):
        """Recheck the candidate and return its earliest local expiry."""
        def reject(reason):
            self.current_last_issue = reason
            return None

        try:
            with self.lock:
                now = _monotonic()
                if (not isinstance(sample, dict) or not _valid_epoch(sample.get("epoch"))
                        or sample.get("epoch") != self.epoch
                        or self.clock_guard.faulted or self.odom_guard.faulted):
                    return reject("epoch_or_guard")
                odom, status, scan, depth = (self.latest["odom"], self.latest["status"],
                                             self.latest["scan"], self.latest["depth"])
                if not all(_finite(sample.get(name)) for name in
                           ("snapshot_at", "scan_at", "status_at", "depth_at", "color_at")):
                    return reject("invalid_sample_timestamps")
                ages = (("latest_scan", scan.get("stamp"), 0.3),
                        ("latest_odom", odom.get("stamp"), LATEST_INPUT_MAX_AGE_SEC),
                        ("latest_status", status.get("stamp"), 1.0),
                        ("snapshot", sample["snapshot_at"], 0.25),
                        ("scan", sample["scan_at"], 0.5),
                        ("depth", sample["depth_at"], 0.5),
                        ("color", sample["color_at"], 0.5))
                for name, stamp, limit in ages:
                    if not _finite(stamp) or not 0 <= now - stamp <= limit:
                        return reject("stale_" + name)
                if scan["stamp"] < sample["scan_at"] or status["stamp"] < sample["status_at"]:
                    return reject("timestamp_regression")
                # A newer frame must be projected and checked rather than
                # silently authorizing motion from the earlier depth image.
                if (not _finite(depth.get("stamp")) or depth["stamp"] < sample["depth_at"]
                        or (not getattr(self, 'defer_depth_validation', False)
                        and depth["stamp"] != sample["depth_at"]
                        and not self._newest_depth_safe_locked(sample, depth, odom, now))):
                    return reject("new_depth_unchecked")
                scan_veto_until = None
                if scan["stamp"] != sample["scan_at"]:
                    aligned_scan = nearest_odometry(self.odom_history, scan["stamp"])
                    if (not isinstance(sample.get("scene"), dict)
                            or aligned_scan is None or aligned_scan.get("epoch") != self.epoch
                            or not self._cached_newest_scan_safe_locked(sample, scan, aligned_scan, odom, now)):
                        return reject("new_scan")
                    scan_veto_until = getattr(self, '_scan_veto_valid_until', None)
                    if not _finite(scan_veto_until):
                        return reject('new_scan_expiry')
                distance, rotation = _pose_delta(sample["pose"], odom["pose"])
                state = base_state(odom["message"], status["message"], now)
                if not _finite(distance) or not _finite(rotation) or distance > 0.005 or rotation > 0.01:
                    return reject("pose_changed")
                if (state is None or state["motion_mode"] != 0 or state["error_code"] != 0
                        or state["vehicle_state"] != 0 or state["control_mode"] != 1
                        or state["battery_voltage"] < 10):
                    return reject("base_state")
                if (not 0 <= state["linear_x_mps"] <= 0.02
                        or abs(state["linear_y_mps"]) > 0.005
                        or abs(state["angular_z_radps"]) > 0.02):
                    return reject("base_speed")
                limits = {"latest_odom": odom["stamp"] + LATEST_INPUT_MAX_AGE_SEC,
                          "latest_scan": scan["stamp"] + 0.3,
                          "latest_depth": depth["stamp"] + LATEST_INPUT_MAX_AGE_SEC,
                          "latest_status": status["stamp"] + 1.0,
                          "snapshot": sample["snapshot_at"] + 0.25, "scan": sample["scan_at"] + 0.5,
                          "status": sample["status_at"] + 1.0, "depth": sample["depth_at"] + 0.5,
                          "color": sample["color_at"] + 0.5}
                if scan_veto_until is not None:
                    limits['new_scan_veto'] = scan_veto_until
                # The live odometry is rechecked after this captured view is
                # evaluated, so its old expiry must not mask newer odometry.
                del limits['latest_odom']
                self.validation_limits = dict(limits)
                self.current_limit_kind = min(limits, key=limits.get)
                deadline = limits[self.current_limit_kind]
                checked_at = _monotonic()
                self.validation_checked_at = checked_at
                if (not _finite(checked_at) or checked_at < now
                        or not _finite(deadline) or deadline <= checked_at):
                    return reject("timestamps")
                self.current_last_issue = None
                return deadline
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return reject("malformed")

    def continuity_snapshot(self):
        """Read odom, fault and epoch together without I/O or mixed generations."""
        with self.lock:
            now = _monotonic()
            odom = self.latest.get("odom")
            stamp = odom.get("stamp") if isinstance(odom, dict) else None
            pose = None
            if (not self.clock_guard.faulted and not self.odom_guard.faulted
                    and _finite(stamp) and 0 <= now - stamp <= 0.1):
                pose = odom.get("pose")
            return {"epoch": self.epoch, "pose": pose, "odom_at": stamp, "checked_at": now,
                    "clock_fault": self.clock_guard.fault_reason,
                    "odom_fault": self.odom_guard.fault_reason}

    def stationary_trace(self, start, end):
        """Return the epoch-tagged odometry samples spanning a model request."""
        if not all(_finite(value) for value in (start, end)) or not 0 <= start <= end:
            return []
        try:
            with self.lock:
                if self.clock_guard.faulted or self.odom_guard.faulted:
                    return []
                history = list(self.odom_history)
            earlier = [entry for entry in history if entry["stamp"] <= start]
            if not earlier:
                return []
            first = earlier[-1]
            selected = [first] + [entry for entry in history if start < entry["stamp"] <= end]
            result = []
            for entry in selected:
                message = entry["message"]
                twist = message.twist.twist
                result.append({"stamp": entry["stamp"], "seq": message.header.seq,
                               "pose": entry["pose"], "epoch": entry["epoch"],
                               "linear_x_mps": float(twist.linear.x),
                               "linear_y_mps": float(twist.linear.y),
                               "angular_z_radps": float(twist.angular.z)})
            return result
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return []

    def diagnostic_status(self):
        with self.lock:
            now = _monotonic()
            ages = {name: max(0.0, now - record["stamp"])
                    for name, record in self.latest.items()}
            return {"epoch": self.epoch, "clock_fault": self.clock_guard.fault_reason,
                    "odom_fault": self.odom_guard.fault_reason, "sensor_ages_sec": ages,
                    "epoch_issue": getattr(self, 'epoch_issue', None),
                    "snapshot_issue": getattr(self, "snapshot_last_issue", None)}


def observe(seconds):
    """Run model and sensor diagnostics without publishing any velocity commands."""
    import cv2
    import rospy

    rospy.init_node("limo_standalone_observer", anonymous=True, disable_signals=True)
    if rospy.get_param("/use_sim_time", False):
        raise ValueError("standalone observation requires ROS wall time")
    source = RosObservation()
    started = _monotonic()
    policy = ForwardPolicy(started, started + seconds)
    gate = DecisionGate()
    gate_lock = threading.Lock()
    shutdown = threading.Event()
    query_running = threading.Event()
    next_query, next_report = started, started
    query_result = {"status": "waiting"}
    os.environ.setdefault("ROSCLAW_LIMO_DECISIONS_KEY_FILE",
                          os.path.expanduser("~/rosclaw/jev/.env"))

    def query(sample):
        try:
            image = source.bridge.imgmsg_to_cv2(sample["color"], desired_encoding="bgr8")
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            scan = sample["scan"]
            model_sample = dict(sample)
            model_sample["forward_view"] = physical_forward_view(
                scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min,
                scan.range_max, *sample["laser_extrinsics"])
            evidence = build_decision_evidence(model_sample, encoded.tostring()) if ok else None
            if evidence is None or evidence["captured_at"] < started:
                with gate_lock:
                    gate.invalidate()
                query_result["status"] = "invalid_observation"
                return
            result = run_decision_process(
                ["/usr/bin/python3", os.path.join(WORKER, "limo_mapless_decision.py")], evidence,
                cancelled=shutdown.is_set, clock=_monotonic, timeout=6.0)
            with gate_lock:
                accepted = (source.is_current_epoch(sample["epoch"])
                            and gate.accept(result, evidence["captured_at"], sample["image_pose"],
                                            _monotonic(), source.current_pose()))
                if not accepted:
                    gate.invalidate()
            query_result["status"] = "accepted" if accepted else "unavailable_or_expired"
        except Exception:  # noqa: BLE001 - unavailable model remains no decision
            with gate_lock:
                gate.invalidate()
            query_result["status"] = "unavailable_or_expired"
        finally:
            query_running.clear()

    try:
        while not rospy.is_shutdown() and _monotonic() < started + seconds:
            sample = source.snapshot()
            now = _monotonic()
            if sample is not None:
                if now >= next_query and not query_running.is_set():
                    query_running.set()
                    thread = threading.Thread(target=query, args=(sample,))
                    thread.daemon = True
                    thread.start()
                    next_query = now + 2.0
                with gate_lock:
                    decision = gate.current(now, sample["pose"])
                if snapshot_is_fresh(sample, now, source.current_pose()):
                    command = policy.step(sample["scene"], sample["state"], sample["depth"], decision, now)
                else:
                    command = {"linear": 0.0, "angular": 0.0, "reason": "pose_or_snapshot_stale"}
                front = assess_front(sample["scene"], now)
                if now >= next_report:
                    coverage = sample["scene"].get("coverage", {})
                    print(json.dumps({"mode": "observe_only", "decision": decision,
                                      "model_status": query_result["status"],
                                      "would_command": command, "front": front,
                                      "latest_scan_valid_fraction": coverage.get("latest_valid_fraction"),
                                      "latest_scan_bin_counts": {
                                          "nan": coverage.get("latest_nan_bins"),
                                          "zero": coverage.get("latest_zero_bins"),
                                          "valid": coverage.get("latest_finite_valid_samples"),
                                          "other": coverage.get("latest_other_bins")},
                                      "depth": sample["depth"]}, allow_nan=False))
                    sys.stdout.flush()
                    next_report = now + 1.0
            else:
                source.break_epoch()
                with gate_lock:
                    gate.invalidate()
                if now >= next_report:
                    print(json.dumps({"mode": "observe_only", "reason": "waiting_for_aligned_sensors"}))
                    sys.stdout.flush()
                    next_report = now + 1.0
            time.sleep(0.1)
    finally:
        shutdown.set()
        rospy.signal_shutdown("observation complete")


def main():
    parser = argparse.ArgumentParser(description="Standalone LIMO sensor and Decisions diagnostic")
    parser.add_argument("--seconds", type=float, default=12.0)
    arguments = parser.parse_args()
    if not _finite(arguments.seconds) or not 1 <= arguments.seconds <= 60:
        parser.error("--seconds must be between 1 and 60")
    observe(arguments.seconds)


if __name__ == "__main__":
    main()
