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
import json
import math
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKER = os.path.join(ROOT, "worker")
if not os.path.isdir(WORKER):
    WORKER = os.path.join(ROOT, "worktrees", "limo-mcp", "worker")
if WORKER not in sys.path:
    sys.path.insert(0, WORKER)

from limo_mapless_worker import run_decision_process, summarize_forward_view
from limo_scan_fusion import RollingScanFusion
from observation_guards import ClockGuard, OdomGuard, sensors_aligned
from policy import ForwardPolicy, assess_front
from projected_depth_gate import project_depth_frame
from relay import _monotonic

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - ROS Melodic Python 2
except NameError:
    INTEGER_TYPES = (int,)
NUMBER_TYPES = INTEGER_TYPES + (float,)


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


def newest_scan_veto(scan, extrinsics):
    """Reject newly measured front hits; non-returns remain unknown.

    This permits an already assessed snapshot to survive arrival of a newer
    scan only when that scan has enough measured front rays and no measured
    hit in the padded corridor. It does not assert clearance between rays.
    """
    try:
        if (scan.header.frame_id.lstrip("/") != "laser_link"
                or not transform_matches(extrinsics, extrinsics)):
            return False
        angle_min, increment = scan.angle_min, scan.angle_increment
        lower, upper, ranges = scan.range_min, scan.range_max, scan.ranges
        if (not all(_finite(item) for item in (angle_min, increment, lower, upper))
                or increment == 0 or not 0 < lower < upper <= 100
                or not isinstance(ranges, (tuple, list)) or not 1 <= len(ranges) <= 8192):
            return False
        angle_end = angle_min + (len(ranges) - 1) * increment
        if (not _finite(angle_end) or not -math.pi - 1e-5 <= angle_min <= math.pi + 1e-5
                or not -math.pi - 1e-5 <= angle_end <= math.pi + 1e-5):
            return False
        valid_front = 0
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
            valid_front += 1
            x = extrinsics[0] + value * math.cos(angle)
            y = extrinsics[1] + value * math.sin(angle)
            if -0.20 <= x <= 0.65 and abs(y) <= 0.17:
                return False
        return valid_front >= 20
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

    def __init__(self, depth_projection_config=None):
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
        self.epoch = 0
        self.fusion_epoch = None
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
                    return
                stamp = source_time(ros_now, received, message.header.stamp.to_sec(), self.limits[name])
                if stamp is None:
                    self.latest.pop(name, None)
                    self.epoch += 1
                    return
                entry = {"message": message, "stamp": stamp}
                if name == "odom":
                    if (message.header.frame_id.lstrip("/") != "odom"
                            or message.child_frame_id.lstrip("/") != "base_link"):
                        self.latest.pop(name, None)
                        self.epoch += 1
                        return
                    p = message.pose.pose.position
                    q = message.pose.pose.orientation
                    quaternion = (q.x, q.y, q.z, q.w)
                    if (not all(_finite(x) for x in (p.x, p.y) + quaternion)
                            or not 0.99 <= sum(x * x for x in quaternion) <= 1.01):
                        self.latest.pop(name, None)
                        self.epoch += 1
                        return
                    roll, pitch, yaw = self.tf.transformations.euler_from_quaternion(quaternion)
                    if abs(roll) > 0.1 or abs(pitch) > 0.1:
                        self.latest.pop(name, None)
                        self.epoch += 1
                        return
                    entry["pose"] = (p.x, p.y, yaw)
                    if not self.odom_guard.observe(message.header.seq, stamp, entry["pose"]):
                        self.latest.clear()
                        self.odom_history.clear()
                        self.epoch += 1
                        return
                previous = self.latest.get(name)
                if previous is not None and stamp <= previous["stamp"]:
                    self.latest.pop(name, None)
                    self.epoch += 1
                    return
                self.latest[name] = entry
                if name == "odom":
                    entry["epoch"] = self.epoch
                    self.odom_history.append(entry)
        except (AttributeError, TypeError, ValueError, OverflowError):
            with self.lock:
                self.latest.pop(name, None)
                self.epoch += 1
            return

    def snapshot(self):
        """Take aligned observations, independently of model/network activity."""
        with self.lock:
            if self.clock_guard.faulted or self.odom_guard.faulted:
                return None
            current = dict(self.latest)
            odom_history = list(self.odom_history)[-150:]
            epoch = self.epoch
        now = _monotonic()
        if any(name not in current for name in self.limits):
            return None
        if any(not 0 <= now - current[name]["stamp"] <= self.limits[name]
               for name in self.limits):
            return None
        if not sensors_aligned({name: current[name]["stamp"] for name in ("scan", "color", "depth")},
                               odom_history, now, max_sensor_age_sec=0.5,
                               max_sensor_skew_sec=0.25, max_odom_skew_sec=0.05,
                               max_translation_m=0.02, max_rotation_rad=0.02,
                               max_linear_mps=0.12, max_angular_radps=0.1):
            return None
        scan = current["scan"]["message"]
        if scan.header.frame_id.lstrip("/") != "laser_link":
            return None
        status = current["status"]["message"]
        depth = current["depth"]["message"]
        info = current["info"]["message"]
        color = current["color"]["message"]
        if (depth.header.frame_id.lstrip("/") != "camera_depth_optical_frame"
                or color.header.frame_id.lstrip("/") != "camera_color_optical_frame"
                or depth.header.frame_id != info.header.frame_id
                or depth.width != info.width or depth.height != info.height
                or abs(current["depth"]["stamp"] - current["info"]["stamp"]) > 0.2):
            return None
        try:
            trans, rot = self.listener.lookupTransform("base_link", scan.header.frame_id,
                                                       scan.header.stamp)
            roll, pitch, yaw = self.tf.transformations.euler_from_quaternion(rot)
            extrinsics = (trans[0], trans[1], yaw)
            signature = (trans[0], trans[1], trans[2], roll, pitch, yaw)
            if (not all(_finite(x) for x in signature)
                    or abs(roll) > 0.1 or abs(pitch) > 0.1):
                return None
            if self.fusion is None or self.fusion_epoch != epoch:
                self.extrinsics = signature
                self.fusion = RollingScanFusion(extrinsics)
                self.fusion_epoch = epoch
                self.scan_stamp = None
            elif not transform_matches(self.extrinsics, signature):
                # Never continue across an extrinsic change without a restart.
                return None
            if current["scan"]["stamp"] != self.scan_stamp:
                aligned = nearest_odometry(odom_history, current["scan"]["stamp"])
                if aligned is None or aligned.get("epoch") != epoch:
                    return None
                packet = {"stamp": current["scan"]["stamp"], "ranges": list(scan.ranges),
                          "angle_min": scan.angle_min, "angle_increment": scan.angle_increment,
                          "range_min": scan.range_min, "range_max": scan.range_max}
                if not self.fusion.add_scan(packet, aligned["pose"], aligned["stamp"], now):
                    return None
                self.scan_stamp = current["scan"]["stamp"]
            depth_array = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
            depth_summary = depth_grid(depth_array, depth.encoding, current["depth"]["stamp"])
            if depth_summary is None:
                return None
            if self.depth_projection_config is not None:
                depth_trans, depth_rot = self.listener.lookupTransform(
                    "base_link", depth.header.frame_id, depth.header.stamp)
                depth_roll, depth_pitch, depth_yaw = self.tf.transformations.euler_from_quaternion(depth_rot)
                depth_signature = tuple(depth_trans) + (depth_roll, depth_pitch, depth_yaw)
                if self.depth_extrinsics is None:
                    if not transform_matches(depth_signature, depth_signature):
                        return None
                    self.depth_extrinsics = depth_signature
                elif not transform_matches(self.depth_extrinsics, depth_signature):
                    return None
                matrix = self.tf.transformations.quaternion_matrix(depth_rot)
                matrix[:3, 3] = depth_trans
                k = info.K
                intrinsics = {"width": depth.width, "height": depth.height,
                              "fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5],
                              "rectified": True}
                depth_summary["corridor"] = project_depth_frame(
                    depth_array, depth.encoding, intrinsics, matrix, current["depth"]["stamp"],
                    self.depth_projection_config)
            image_pose = nearest_odometry(odom_history, current["color"]["stamp"])
            depth_pose = nearest_odometry(odom_history, current["depth"]["stamp"])
            if (image_pose is None or image_pose.get("epoch") != epoch
                    or depth_pose is None or depth_pose.get("epoch") != epoch):
                return None
            with self.lock:
                if self.epoch != epoch or self.clock_guard.faulted or self.odom_guard.faulted:
                    return None
                snapshot_now = _monotonic()
                latest_odom = refreshed_odometry(current["odom"], self.latest.get("odom"), snapshot_now)
            if latest_odom is None:
                return None
            pose = latest_odom["pose"]
            scene = self.fusion.snapshot(pose, latest_odom["stamp"], snapshot_now)
            odom = latest_odom["message"]
            state = base_state(odom, status, min(latest_odom["stamp"], current["status"]["stamp"]))
            if state is None:
                return None
            with self.lock:
                if self.epoch != epoch or self.clock_guard.faulted or self.odom_guard.faulted:
                    return None
            return {"scene": scene, "state": state, "depth": depth_summary, "snapshot_at": snapshot_now, "epoch": epoch,
                    "pose": pose, "image_pose": image_pose["pose"], "depth_pose": depth_pose["pose"],
                    "color": current["color"]["message"], "scan": scan,
                    "laser_extrinsics": extrinsics,
                    "scan_at": current["scan"]["stamp"], "odom_at": latest_odom["stamp"],
                    "status_at": current["status"]["stamp"], "depth_at": current["depth"]["stamp"],
                    "color_at": current["color"]["stamp"]}
        except Exception:  # noqa: BLE001 - missing/invalid observation cannot authorize motion
            return None

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

    def is_current_epoch(self, epoch):
        with self.lock:
            return (_valid_epoch(epoch) and epoch == self.epoch and not self.clock_guard.faulted
                    and not self.odom_guard.faulted)

    def is_current_sample(self, sample):
        """Recheck a candidate against the latest local scan, base and pose."""
        try:
            with self.lock:
                now = _monotonic()
                if (not isinstance(sample, dict) or not _valid_epoch(sample.get("epoch"))
                        or sample.get("epoch") != self.epoch
                        or self.clock_guard.faulted or self.odom_guard.faulted):
                    return False
                odom, status, scan = self.latest["odom"], self.latest["status"], self.latest["scan"]
                if (not all(_finite(sample.get(name)) for name in
                            ("snapshot_at", "scan_at", "status_at", "depth_at", "color_at"))
                        or not _finite(scan.get("stamp"))
                        or not 0 <= now - scan["stamp"] <= 0.3
                        or scan["stamp"] < sample["scan_at"]
                        or status["stamp"] < sample["status_at"]
                        or not 0 <= now - odom["stamp"] <= 0.1
                        or not 0 <= now - status["stamp"] <= 1.0
                        or not 0 <= now - sample["snapshot_at"] <= 0.25
                        or not 0 <= now - sample["scan_at"] <= 0.5
                        or not 0 <= now - sample["depth_at"] <= 0.5
                        or not 0 <= now - sample["color_at"] <= 0.5):
                    return False
                if (scan["stamp"] != sample["scan_at"]
                        and not newest_scan_veto(scan.get("message"), self.extrinsics)):
                    return False
                distance, rotation = _pose_delta(sample["pose"], odom["pose"])
                state = base_state(odom["message"], status["message"], now)
                return (distance <= 0.005 and rotation <= 0.01 and state is not None
                        and state["motion_mode"] == 0 and state["error_code"] == 0
                        and state["vehicle_state"] == 0 and state["control_mode"] == 1
                        and state["battery_voltage"] >= 10
                        and 0 <= state["linear_x_mps"] <= 0.02
                        and abs(state["linear_y_mps"]) <= 0.005
                        and abs(state["angular_z_radps"]) <= 0.02)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return False

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
                    "odom_fault": self.odom_guard.fault_reason, "sensor_ages_sec": ages}


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
