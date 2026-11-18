#!/usr/bin/env python2
"""Candidate daemon-owned continuous, mapless LIMO exploration worker.

The running robot pack does not register this worker. It requires an exact
daemon action, ROS observations, a configured Decisions key file and its
dedicated velocity relay. Physical sensing and stop behavior remain unverified.
"""

import base64
import json
import math
import os
import pwd
import re
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time

try:
    STRING_TYPES = (basestring,)  # noqa: F821
    INTEGER_TYPES = (int, long)  # noqa: F821
except NameError:
    STRING_TYPES = (str,)
    INTEGER_TYPES = (int,)
NUMBER_TYPES = INTEGER_TYPES + (float,)

PROTOCOL = "rosclaw.limo.worker.v1"
_cancel = False


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, NUMBER_TYPES):
        return False
    try:
        return not math.isnan(value) and not math.isinf(value)
    except (ValueError, OverflowError):
        return False


def validate_request(value):
    fields = set(
        [
            "protocol",
            "operation",
            "schema_version",
            "action_id",
            "body_id",
            "body_snapshot_hash",
            "duration_sec",
            "daemon_pid",
        ]
    )
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("unexpected request fields")
    duration = value["duration_sec"]
    if (
        value["protocol"] != PROTOCOL
        or value["operation"] != "MAPLESS_EXPLORE"
        or value["schema_version"] != "limo.explore.v1"
        or value["body_id"] != "limo"
        or not isinstance(value["action_id"], STRING_TYPES)
        or not re.match(r"^[A-Za-z0-9_-]{1,128}\Z", value["action_id"])
        or not isinstance(value["body_snapshot_hash"], STRING_TYPES)
        or not re.match(r"^[0-9a-f]{64}\Z", value["body_snapshot_hash"])
        or not _finite(duration)
        or not 1 <= duration <= 30
        or isinstance(value["daemon_pid"], bool)
        or not isinstance(value["daemon_pid"], INTEGER_TYPES)
        or value["daemon_pid"] <= 1
    ):
        raise ValueError("invalid exploration request")
    return dict(value)


def depth_metadata_matches(image, info):
    try:
        return (
            bool(image["frame_id"])
            and image["frame_id"] == info["frame_id"]
            and image["width"] == info["width"]
            and image["height"] == info["height"]
            and _finite(image["stamp"])
            and _finite(info["stamp"])
            and abs(image["stamp"] - info["stamp"]) <= 0.2
        )
    except (KeyError, TypeError):
        return False


def summarize_depth(array, encoding):
    import numpy as np

    if len(array.shape) != 2 or not array.size:
        raise ValueError("invalid depth image")
    region = array.astype("float32")
    if encoding == "16UC1":
        region = region * 0.001
    elif encoding != "32FC1":
        raise ValueError("unsupported depth encoding")
    valid = np.isfinite(region) & (region > 0.05) & (region <= 20.0)
    ratio = float(np.count_nonzero(valid)) / float(region.size)
    minimum = float(np.min(region[valid])) if np.any(valid) else None
    return minimum, ratio


def sensor_stamps_current(stamps, ros_now):
    limits = {"scan": 0.5, "odom": 0.5, "status": 1.2, "depth": 0.8, "color": 0.8}
    if not isinstance(stamps, dict) or not _finite(ros_now):
        return False
    for name, max_age in limits.items():
        stamp = stamps.get(name)
        if not _finite(stamp) or not -0.1 <= ros_now - stamp <= max_age:
            return False
    values = [stamps[name] for name in limits]
    return max(values) - min(values) <= 0.75


class StoppedOdomVerifier(object):  # noqa: UP004 - Python 2 ROS worker
    """Require repeated new low-speed odometry after the stop request."""

    def __init__(self, stop_requested_at, stop_requested_ros):
        if not _finite(stop_requested_at) or not _finite(stop_requested_ros):
            raise ValueError("invalid stop request time")
        self.last_at = stop_requested_at
        self.last_source = stop_requested_ros
        self.samples = 0
        self.first_at = None
        self.first_source = None
        self.linear = None
        self.angular = None

    def observe(self, x, y, angular, observed_at, now, source_stamp, source_now):
        if not all(
            _finite(value) for value in (x, y, angular, observed_at, now, source_stamp, source_now)
        ):
            self.invalidate()
            return False
        if (
            observed_at == self.last_at
            and source_stamp == self.last_source
            and 0 <= now - observed_at <= 0.3
            and 0 <= source_now - source_stamp <= 0.3
        ):
            return False
        if (
            observed_at <= self.last_at
            or source_stamp <= self.last_source
            or not 0 <= now - observed_at <= 0.3
            or not 0 <= source_now - source_stamp <= 0.3
        ):
            self.invalidate()
            return False
        if self.samples and (observed_at - self.last_at > 0.3 or source_stamp - self.last_source > 0.3):
            self.invalidate()
        self.last_at = observed_at
        self.last_source = source_stamp
        self.linear = math.hypot(x, y)
        self.angular = abs(angular)
        if self.linear > 0.01 or self.angular > 0.03:
            self.invalidate()
            return False
        if self.samples == 0:
            self.first_at = observed_at
            self.first_source = source_stamp
        self.samples += 1
        return (self.samples >= 3 and observed_at - self.first_at >= 0.2 - 1e-9
                and source_stamp - self.first_source >= 0.2 - 1e-9)

    def invalidate(self):
        self.samples = 0
        self.first_at = None
        self.first_source = None


def summarize_scan(ranges, angle_min, angle_increment, range_min, range_max, tx, ty, yaw):
    if not all(
        _finite(value) for value in (angle_min, angle_increment, range_min, range_max, tx, ty, yaw)
    ):
        raise ValueError("invalid laser metadata")
    if (
        not 0 < range_min < range_max <= 100
        or angle_increment == 0
        or not 1 <= len(ranges) <= 10000
    ):
        raise ValueError("invalid laser ranges")
    sectors = {"front": [], "left": [], "right": [], "rear": []}
    coverage = {name: [[0, 0] for _ in range(4)] for name in sectors}

    def sector_and_bin(angle):
        angle = math.atan2(math.sin(angle), math.cos(angle))
        if -0.55 <= angle <= 0.55:
            name, offset, width = "front", angle + 0.55, 1.1
        elif 0.55 < angle <= 2.1:
            name, offset, width = "left", angle - 0.55, 1.55
        elif -2.1 <= angle < -0.55:
            name, offset, width = "right", angle + 2.1, 1.55
        else:
            name, offset, width = "rear", (angle if angle > 0 else angle + 2 * math.pi) - 2.1, 2 * math.pi - 4.2
        return name, min(3, max(0, int(4 * offset / width)))

    for index, distance in enumerate(ranges):
        angle = angle_min + index * angle_increment + yaw
        ray_sector, ray_bin = sector_and_bin(angle)
        coverage[ray_sector][ray_bin][0] += 1
        if not _finite(distance) or not range_min <= distance <= range_max:
            continue
        coverage[ray_sector][ray_bin][1] += 1
        x = tx + distance * math.cos(angle)
        y = ty + distance * math.sin(angle)
        bearing = math.atan2(y, x)
        measured = math.hypot(x, y)
        endpoint_sector, _ = sector_and_bin(bearing)
        sectors[endpoint_sector].append(measured)
    return {
        name: min(values) if values and all(total >= 2 and valid * 2 >= total
                                             for total, valid in coverage[name]) else None
        for name, values in sectors.items()
    }


def summarize_forward_view(ranges, angle_min, angle_increment, range_min, range_max, tx, ty, yaw):
    """Describe measured forward rays and missing angular spans without inferring free space."""
    if not all(
        _finite(value) for value in (angle_min, angle_increment, range_min, range_max, tx, ty, yaw)
    ) or not (0 < range_min < range_max <= 100 and angle_increment != 0 and 1 <= len(ranges) <= 10000):
        raise ValueError("invalid laser scan")
    bounds = {
        "right": (-math.pi / 2, -math.pi / 6),
        "center": (-math.pi / 6, math.pi / 6),
        "left": (math.pi / 6, math.pi / 2),
    }
    sectors = {
        name: {"total_rays": 0, "valid_rays": 0, "nearest_to_base_m": None,
               "nearest_bearing_from_base_rad": None, "observed_angle_min_rad": None,
               "observed_angle_max_rad": None, "max_invalid_return_run_rad": 0.0,
               "_missing": 0, "_valid_angles": []}
        for name in bounds
    }
    base_hits = {name: {"nearest_m": None, "bearing_rad": None}
                 for name in ("right", "center", "left", "other")}
    rear_total = 0
    rear_valid = 0
    rear_nearest = None
    for index, distance in enumerate(ranges):
        angle = angle_min + index * angle_increment + yaw
        angle = math.atan2(math.sin(angle), math.cos(angle))
        if -math.pi / 2 <= angle < -math.pi / 6:
            name = "right"
        elif -math.pi / 6 <= angle <= math.pi / 6:
            name = "center"
        elif math.pi / 6 < angle <= math.pi / 2:
            name = "left"
        else:
            name = None
        valid = _finite(distance) and range_min <= distance <= range_max
        if valid:
            x, y = tx + distance * math.cos(angle), ty + distance * math.sin(angle)
            measured = math.hypot(x, y)
            bearing = math.atan2(y, x) if measured > 1e-9 else None
            if bearing is None:
                base_name = "other"
            elif -math.pi / 2 <= bearing < -math.pi / 6:
                base_name = "right"
            elif -math.pi / 6 <= bearing <= math.pi / 6:
                base_name = "center"
            elif math.pi / 6 < bearing <= math.pi / 2:
                base_name = "left"
            else:
                base_name = "other"
            hit = base_hits[base_name]
            if hit["nearest_m"] is None or measured < hit["nearest_m"]:
                hit["nearest_m"] = measured
                hit["bearing_rad"] = bearing
        if name is None:
            if abs(angle) >= 2.1:
                rear_total += 1
                if valid:
                    rear_nearest = measured if rear_nearest is None else min(rear_nearest, measured)
                    rear_valid += 1
            continue
        sector = sectors[name]
        sector["total_rays"] += 1
        low, high = sector["observed_angle_min_rad"], sector["observed_angle_max_rad"]
        sector["observed_angle_min_rad"] = angle if low is None else min(low, angle)
        sector["observed_angle_max_rad"] = angle if high is None else max(high, angle)
        if valid:
            nearest = sector["nearest_to_base_m"]
            if nearest is None or measured < nearest:
                sector["nearest_to_base_m"] = measured
                sector["nearest_bearing_from_base_rad"] = bearing
            sector["valid_rays"] += 1
            sector["_valid_angles"].append(angle)
            sector["_missing"] = 0
        else:
            sector["_missing"] += 1
            sector["max_invalid_return_run_rad"] = max(
                sector["max_invalid_return_run_rad"], sector["_missing"] * abs(angle_increment)
            )
    for name, sector in sectors.items():
        del sector["_missing"]
        angles = sorted(sector.pop("_valid_angles"))
        low, high = bounds[name]
        if angles:
            gaps = [angles[0] - low, high - angles[-1]]
            gaps.extend(angles[index] - angles[index - 1] for index in range(1, len(angles)))
            sector["max_unobserved_angle_rad"] = max(gaps)
        else:
            sector["max_unobserved_angle_rad"] = high - low
        if not sector["total_rays"]:
            sector["max_invalid_return_run_rad"] = None
    sectors["rear_total_rays"] = rear_total
    sectors["rear_valid_rays"] = rear_valid
    sectors["rear_nearest_to_base_m"] = rear_nearest
    sectors["base_hits"] = base_hits
    return sectors


def motion_delta(initial, current):
    if len(initial) != 3 or len(current) != 3 or not all(_finite(v) for v in initial + current):
        raise ValueError("invalid odometry pose")
    translation = math.hypot(current[0] - initial[0], current[1] - initial[1])
    yaw_difference = current[2] - initial[2]
    rotation = abs(math.atan2(math.sin(yaw_difference), math.cos(yaw_difference)))
    return translation, rotation


class MotionEvidence(object):  # noqa: UP004 - Python 2 ROS worker
    def __init__(self):
        self.baseline = None
        self.max_translation = 0.0
        self.max_rotation = 0.0

    def acknowledge(self, pose):
        motion_delta(pose, pose)
        if self.baseline is None:
            self.baseline = pose

    def observe(self, pose):
        if self.baseline is not None:
            translation, rotation = motion_delta(self.baseline, pose)
            self.max_translation = max(self.max_translation, translation)
            self.max_rotation = max(self.max_rotation, rotation)


def decision_pose_current(observed_pose, current_pose):
    try:
        translation, rotation = motion_delta(observed_pose, current_pose)
        return translation <= 0.15 and rotation <= 0.35
    except (ValueError, TypeError):
        return False


def run_decision_process(command, evidence, cancelled, clock, timeout=6.0):
    """Bound helper wall time and reap it on shutdown or timeout."""
    payload = json.dumps(evidence, allow_nan=False).encode("utf-8")
    if len(payload) > 3 * 1024 * 1024 or cancelled():
        return None
    with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as sink, open(os.devnull, "w") as errors:
        source.write(payload)
        source.seek(0)
        process = subprocess.Popen(
            command, stdin=source, stdout=sink, stderr=errors, close_fds=True
        )
        deadline = clock() + timeout
        while process.poll() is None and not cancelled() and clock() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            process.terminate()
            grace = clock() + 0.3
            while process.poll() is None and clock() < grace:
                time.sleep(0.05)
            if process.poll() is None:
                process.kill()
            process.wait()
            return None
        if cancelled() or clock() >= deadline or process.returncode != 0:
            return None
        sink.seek(0)
        output = sink.read(65537)
        if len(output) > 65536:
            return None
        return json.loads(output)


def _metadata(message):
    return {
        "frame_id": message.header.frame_id,
        "width": message.width,
        "height": message.height,
        "stamp": message.header.stamp.to_sec(),
    }


def _signal_cancel(_signum, _frame):
    global _cancel
    _cancel = True


def execute(request):
    if os.getppid() != request["daemon_pid"]:
        raise ValueError("daemon parent is no longer present")
    import cv2
    import rospy
    import tf
    from cv_bridge import CvBridge
    from limo_base.msg import LimoStatus
    from limo_mapless_controller import DECISION_TTL, MaplessController
    from limo_mapless_relay import _monotonic
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo, Image, LaserScan

    if os.geteuid() != pwd.getpwnam("rosclaw-hw").pw_uid:
        raise ValueError("worker must be launched by rosclawd")
    key_file = os.environ.get("ROSCLAW_LIMO_DECISIONS_KEY_FILE")
    if not key_file or not os.path.isfile(key_file):
        raise ValueError("Decisions API key file is not configured")
    rospy.init_node("rosclaw_limo_mapless", anonymous=True, disable_signals=True)
    bridge = CvBridge()
    listener = tf.TransformListener()
    lock = threading.Lock()
    controller_lock = threading.Lock()
    records = {}
    decision = {"running": False, "count": 0, "pose": None, "expired": False}
    decision_shutdown = threading.Event()
    decision_thread = None

    def callback(message, name):
        with lock:
            records[name] = (message, _monotonic())

    subscriptions = [
        rospy.Subscriber("/scan", LaserScan, callback, callback_args="scan", queue_size=1),
        rospy.Subscriber("/odom", Odometry, callback, callback_args="odom", queue_size=1),
        rospy.Subscriber(
            "/limo_status", LimoStatus, callback, callback_args="status", queue_size=1
        ),
        rospy.Subscriber(
            "/camera/color/image_raw", Image, callback, callback_args="color", queue_size=1
        ),
        rospy.Subscriber(
            "/camera/depth/image_raw", Image, callback, callback_args="depth", queue_size=1
        ),
        rospy.Subscriber(
            "/camera/depth/camera_info", CameraInfo, callback, callback_args="info", queue_size=1
        ),
    ]
    started = _monotonic()
    controller = MaplessController(started, started + request["duration_sec"])
    parent_pid = request["daemon_pid"]
    directory = os.path.dirname(os.path.abspath(__file__))
    if os.getppid() != parent_pid:
        raise ValueError("daemon parent is no longer present")
    relay = subprocess.Popen(
        [sys.executable, os.path.join(directory, "limo_mapless_relay.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        close_fds=True,
    )
    commanded = False
    relay_acknowledged = False
    acknowledged_at = None
    saw_observation = False
    reason = "duration_elapsed"
    stopped_odometry = None
    movement = MotionEvidence()

    def snapshot():
        with lock:
            current = dict(records)
        if any(
            name not in current for name in ("scan", "odom", "status", "color", "depth", "info")
        ):
            return None
        scan, scan_at = current["scan"]
        odom, odom_at = current["odom"]
        status, status_at = current["status"]
        depth, depth_at = current["depth"]
        info, _ = current["info"]
        color, color_at = current["color"]
        if not depth_metadata_matches(_metadata(depth), _metadata(info)):
            return None
        if (
            odom.header.frame_id.lstrip("/") != "odom"
            or odom.child_frame_id.lstrip("/") != "base_link"
        ):
            return None
        ros_now = rospy.Time.now().to_sec()
        mono_now = _monotonic()
        source_stamps = {name: message.header.stamp.to_sec() for name, message in
                         (("scan", scan), ("odom", odom), ("status", status), ("depth", depth), ("color", color))}
        if not sensor_stamps_current(source_stamps, ros_now):
            return None
        scan_at = min(scan_at, mono_now - (ros_now - source_stamps["scan"]))
        odom_at = min(odom_at, mono_now - (ros_now - source_stamps["odom"]))
        status_at = min(status_at, mono_now - (ros_now - source_stamps["status"]))
        depth_at = min(depth_at, mono_now - (ros_now - source_stamps["depth"]))
        color_at = min(color_at, mono_now - (ros_now - source_stamps["color"]))
        try:
            trans, rot = listener.lookupTransform(
                "base_link", scan.header.frame_id, scan.header.stamp
            )
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(rot)
            if abs(roll) > 0.1 or abs(pitch) > 0.1:
                return None
            sectors = summarize_scan(
                scan.ranges,
                scan.angle_min,
                scan.angle_increment,
                scan.range_min,
                scan.range_max,
                trans[0],
                trans[1],
                yaw,
            )
            forward_view = summarize_forward_view(
                scan.ranges,
                scan.angle_min,
                scan.angle_increment,
                scan.range_min,
                scan.range_max,
                trans[0],
                trans[1],
                yaw,
            )
            depth_array = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
            minimum, ratio = summarize_depth(depth_array, depth.encoding)
            pose = odom.pose.pose
            orientation = pose.orientation
            odom_yaw = tf.transformations.euler_from_quaternion(
                (orientation.x, orientation.y, orientation.z, orientation.w)
            )[2]
            return {
                "scan_at": scan_at,
                "odom_at": odom_at,
                "status_at": status_at,
                "depth_at": depth_at,
                "front": sectors["front"],
                "left": sectors["left"],
                "right": sectors["right"],
                "rear": sectors["rear"],
                "forward_view": forward_view,
                "depth_min": minimum,
                "depth_valid_ratio": ratio,
                "error_code": int(status.error_code),
                "motion_mode": int(status.motion_mode),
                "battery_voltage": float(status.battery_voltage),
                "linear_x_mps": float(odom.twist.twist.linear.x),
                "angular_z_radps": float(odom.twist.twist.angular.z),
                "pose": (float(pose.position.x), float(pose.position.y), odom_yaw),
                "color": color,
                "color_at": color_at,
            }
        except Exception:  # noqa: BLE001 - transient sensor/TF failures produce no command
            return None

    def query_model(sample):
        try:
            image = bridge.imgmsg_to_cv2(sample["color"], desired_encoding="bgr8")
            ok, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                return
            encoded = base64.b64encode(jpeg.tobytes()).decode("ascii")
            captured = min(
                sample[name] for name in ("scan_at", "odom_at", "status_at", "depth_at", "color_at")
            )
            evidence = {
                "image_b64": encoded,
                "captured_at": captured,
                "laser": {
                    "front_min_m": sample["front"],
                    "left_min_m": sample["left"],
                    "right_min_m": sample["right"],
                    "rear_min_m": sample["rear"],
                    "forward_view": sample["forward_view"],
                },
                "base": {
                    "linear_x_mps": sample["linear_x_mps"],
                    "angular_z_radps": sample["angular_z_radps"],
                    "battery_voltage": sample["battery_voltage"],
                },
            }
            response = run_decision_process(
                ["/usr/bin/python3", os.path.join(directory, "limo_mapless_decision.py")],
                evidence,
                cancelled=lambda: _cancel or decision_shutdown.is_set(),
                clock=_monotonic,
            )
            if (isinstance(response, dict) and not decision_shutdown.is_set()
                    and response.get("ok") is True and response.get("captured_at") == captured):
                with lock:
                    latest = records.get("odom")
                if latest is None:
                    return
                latest_odom, latest_at = latest
                now = _monotonic()
                ros_now = rospy.Time.now().to_sec()
                if (not 0 <= now - latest_at <= 0.5
                        or not 0 <= ros_now - latest_odom.header.stamp.to_sec() <= 0.5
                        or latest_odom.header.frame_id.lstrip("/") != "odom"
                        or latest_odom.child_frame_id.lstrip("/") != "base_link"):
                    return
                pose = latest_odom.pose.pose
                q = pose.orientation
                yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
                current_pose = (float(pose.position.x), float(pose.position.y), yaw)
                if not decision_pose_current(sample["pose"], current_pose):
                    return
                with controller_lock:
                    if controller.set_decision(
                        response.get("choice"), response.get("captured_at"), _monotonic()
                    ):
                        decision["count"] += 1
                        decision["pose"] = sample["pose"]
        except Exception:  # noqa: BLE001 - network/model errors leave the old decision to expire
            pass
        finally:
            with lock:
                decision["running"] = False

    try:
        readable, _, _ = select.select([relay.stdout], [], [], 5.0)
        if not readable or relay.stdout.readline().strip() != b"READY":
            raise ValueError("velocity relay did not become ready")
        sequence = 0
        next_query = started
        while not _cancel and not rospy.is_shutdown() and _monotonic() < controller.deadline:
            now = _monotonic()
            if os.getppid() != parent_pid or relay.poll() is not None:
                reason = "controller_parent_or_relay_lost"
                break
            sample = snapshot()
            now = _monotonic()
            if sample is not None:
                saw_observation = True
                if relay_acknowledged and movement.baseline is None:
                    if sample["odom_at"] > acknowledged_at:
                        movement.acknowledge(sample["pose"])
                else:
                    movement.observe(sample["pose"])
            with controller_lock:
                if decision["count"] and now >= controller.observed_at + DECISION_TTL:
                    decision["expired"] = True
                current_sample = sample
                if sample is not None and decision["pose"] is not None and not decision_pose_current(decision["pose"], sample["pose"]):
                    current_sample = None
                linear, angular = controller.tick(current_sample, now)
            frame = {"sequence": sequence, "stamp": now, "linear": linear, "angular": angular}
            relay.stdin.write((json.dumps(frame, allow_nan=False) + "\n").encode("utf-8"))
            relay.stdin.flush()
            if linear or angular:
                commanded = True
            if not relay_acknowledged:
                readable, _, _ = select.select([relay.stdout], [], [], 0.0)
                if readable:
                    acknowledgement = relay.stdout.readline().strip()
                    match = re.match(br"^MOTION_ACK ([0-9]+)$", acknowledgement)
                    if not match or int(match.group(1)) > sequence or not commanded:
                        reason = "invalid_relay_acknowledgement"
                        break
                    relay_acknowledged = True
                    acknowledged_at = _monotonic()
            sequence += 1
            if sample is not None and now >= next_query:
                with lock:
                    launch = not decision["running"]
                    if launch:
                        decision["running"] = True
                if launch:
                    decision_thread = threading.Thread(target=query_model, args=(sample,))
                    decision_thread.daemon = True
                    decision_thread.start()
                    next_query = now + 1.0
            time.sleep(0.05)
        if _cancel:
            reason = "cancelled"
        elif rospy.is_shutdown():
            reason = "ros_shutdown"
    finally:
        decision_shutdown.set()
        stop_requested_at = _monotonic()
        stop_requested_ros = rospy.Time.now().to_sec()
        try:  # noqa: SIM105 - contextlib.suppress is unavailable on Python 2
            relay.stdin.close()
        except (IOError, OSError):
            pass
        deadline = _monotonic() + 2.0
        while relay.poll() is None and _monotonic() < deadline:
            time.sleep(0.05)
        if relay.poll() is None:
            relay.terminate()
        terminate_deadline = _monotonic() + 1.0
        while relay.poll() is None and _monotonic() < terminate_deadline:
            time.sleep(0.05)
        if relay.poll() is None:
            relay.kill()
        relay.wait()
        if decision_thread is not None:
            decision_thread.join(1.0)
        verifier = StoppedOdomVerifier(stop_requested_at, stop_requested_ros)
        feedback_deadline = _monotonic() + 2.0
        while _monotonic() < feedback_deadline:
            with lock:
                latest_odom = records.get("odom")
            if latest_odom is not None:
                odom, observed_at = latest_odom
                if odom.header.frame_id.lstrip("/") != "odom" or odom.child_frame_id.lstrip("/") != "base_link":
                    verifier.invalidate()
                    time.sleep(0.05)
                    continue
                twist = odom.twist.twist
                if verifier.observe(
                    twist.linear.x,
                    twist.linear.y,
                    twist.angular.z,
                    observed_at,
                    _monotonic(),
                    odom.header.stamp.to_sec(),
                    rospy.Time.now().to_sec(),
                ):
                    stopped_odometry = {
                        "linear_speed_mps": verifier.linear,
                        "angular_speed_radps": verifier.angular,
                        "samples": verifier.samples,
                    }
                    break
            time.sleep(0.05)
        for subscription in subscriptions:
            subscription.unregister()

    if reason == "duration_elapsed":
        if not saw_observation:
            reason = "no_compatible_observation"
        elif not decision["count"]:
            reason = "no_fresh_model_decision"
        elif decision["expired"]:
            reason = "model_decision_expired"
        elif not commanded:
            reason = "no_motion_commanded"
        elif not relay_acknowledged:
            reason = "no_relay_acknowledgement"
        elif not stopped_odometry:
            reason = "stopped_odometry_unverified"
        elif movement.max_translation < 0.01 and movement.max_rotation < 0.03:
            reason = "odometry_motion_not_observed"
    if _cancel:
        reason = "cancelled"

    # Requesting zero is not evidence that the physical base stopped.
    return {
        "ok": bool(
            reason == "duration_elapsed"
            and commanded
            and relay_acknowledged
            and decision["count"]
            and stopped_odometry
            and (movement.max_translation >= 0.01 or movement.max_rotation >= 0.03)
        ),
        "protocol": PROTOCOL,
        "operation": "MAPLESS_EXPLORE",
        "action_id": request["action_id"],
        "reason": reason,
        "decisions_accepted": decision["count"],
        "motion_commanded": commanded,
        "relay_acknowledged": relay_acknowledged,
        "stopped_odometry": stopped_odometry,
        "motion_evidence": {
            "odom_translation_m": movement.max_translation,
            "odom_rotation_rad": movement.max_rotation,
            "physical_motion_independently_observed": False,
        },
        "physical_stop_verified": False,
    }


def main():
    signal.signal(signal.SIGTERM, _signal_cancel)
    signal.signal(signal.SIGINT, _signal_cancel)
    try:
        raw = sys.stdin.read(65537)
        if len(raw) > 65536:
            raise ValueError("request too large")
        result = execute(validate_request(json.loads(raw)))
    except Exception:  # noqa: BLE001 - don't expose credentials or ROS internals
        result = {"ok": False, "error": "mapless worker unavailable"}
    sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
    sys.stdout.flush()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
