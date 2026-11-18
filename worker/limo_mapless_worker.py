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
import threading
import time

try:
    STRING_TYPES = (basestring,)  # noqa: F821
    NUMBER_TYPES = (int, long, float)  # noqa: F821
except NameError:
    STRING_TYPES = (str,)
    NUMBER_TYPES = (int, float)

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


class StoppedOdomVerifier(object):  # noqa: UP004 - Python 2 ROS worker
    """Require repeated new low-speed odometry after the stop request."""

    def __init__(self, stop_requested_at, stop_requested_ros):
        if not _finite(stop_requested_at) or not _finite(stop_requested_ros):
            raise ValueError("invalid stop request time")
        self.last_at = stop_requested_at
        self.last_source = stop_requested_ros
        self.samples = 0
        self.linear = None
        self.angular = None

    def observe(self, x, y, angular, observed_at, now, source_stamp, source_now):
        if not all(
            _finite(value) for value in (x, y, angular, observed_at, now, source_stamp, source_now)
        ):
            self.samples = 0
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
            self.samples = 0
            return False
        self.last_at = observed_at
        self.last_source = source_stamp
        self.linear = math.hypot(x, y)
        self.angular = abs(angular)
        if self.linear > 0.01 or self.angular > 0.03:
            self.samples = 0
            return False
        self.samples += 1
        return self.samples >= 3


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
    sectors = {"front": [], "left": [], "right": []}
    for index, distance in enumerate(ranges):
        if not _finite(distance) or not range_min <= distance <= range_max:
            continue
        angle = angle_min + index * angle_increment + yaw
        x = tx + distance * math.cos(angle)
        y = ty + distance * math.sin(angle)
        bearing = math.atan2(y, x)
        measured = math.hypot(x, y)
        if -0.55 <= bearing <= 0.55:
            sectors["front"].append(measured)
        elif 0.55 < bearing <= 2.1:
            sectors["left"].append(measured)
        elif -2.1 <= bearing < -0.55:
            sectors["right"].append(measured)
    return {name: min(values) if values else None for name, values in sectors.items()}


def motion_delta(initial, current):
    if len(initial) != 3 or len(current) != 3 or not all(_finite(v) for v in initial + current):
        raise ValueError("invalid odometry pose")
    translation = math.hypot(current[0] - initial[0], current[1] - initial[1])
    yaw_difference = current[2] - initial[2]
    rotation = abs(math.atan2(math.sin(yaw_difference), math.cos(yaw_difference)))
    return translation, rotation


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
    import cv2
    import numpy as np
    import rospy
    import tf
    from cv_bridge import CvBridge
    from limo_base.msg import LimoStatus
    from limo_mapless_controller import MaplessController
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
    decision = {"running": False, "count": 0}

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
    parent_pid = os.getppid()
    directory = os.path.dirname(os.path.abspath(__file__))
    relay = subprocess.Popen(
        [sys.executable, os.path.join(directory, "limo_mapless_relay.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        close_fds=True,
    )
    commanded = False
    saw_observation = False
    reason = "duration_elapsed"
    stopped_odometry = None
    start_pose = None
    max_translation = 0.0
    max_rotation = 0.0

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
        for message in (scan, odom, status, depth, color):
            age = ros_now - message.header.stamp.to_sec()
            if not -0.2 <= age <= 1.5:
                return None
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
            depth_array = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
            width = depth_array.shape[1]
            region = depth_array[:, width // 3 : 2 * width // 3].astype("float32")
            if depth.encoding == "16UC1":
                region = region * 0.001
            elif depth.encoding != "32FC1":
                return None
            valid = np.isfinite(region) & (region > 0.05) & (region <= 20.0)
            ratio = float(np.count_nonzero(valid)) / float(region.size)
            minimum = float(np.min(region[valid])) if np.any(valid) else None
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
                },
                "base": {
                    "linear_x_mps": sample["linear_x_mps"],
                    "angular_z_radps": sample["angular_z_radps"],
                    "battery_voltage": sample["battery_voltage"],
                },
            }
            process = subprocess.Popen(
                ["/usr/bin/python3", os.path.join(directory, "limo_mapless_decision.py")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
            )
            output, _ = process.communicate(json.dumps(evidence, allow_nan=False).encode("utf-8"))
            response = json.loads(output)
            if process.returncode == 0 and response.get("ok") is True:
                with controller_lock:
                    if controller.set_decision(
                        response.get("choice"), response.get("captured_at"), _monotonic()
                    ):
                        decision["count"] += 1
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
                if start_pose is None:
                    start_pose = sample["pose"]
                translation, rotation = motion_delta(start_pose, sample["pose"])
                max_translation = max(max_translation, translation)
                max_rotation = max(max_rotation, rotation)
            with controller_lock:
                linear, angular = controller.tick(sample, now)
            if linear or angular:
                commanded = True
            frame = {"sequence": sequence, "stamp": now, "linear": linear, "angular": angular}
            relay.stdin.write((json.dumps(frame, allow_nan=False) + "\n").encode("utf-8"))
            relay.stdin.flush()
            sequence += 1
            if sample is not None and now >= next_query:
                with lock:
                    launch = not decision["running"]
                    if launch:
                        decision["running"] = True
                if launch:
                    thread = threading.Thread(target=query_model, args=(sample,))
                    thread.daemon = True
                    thread.start()
                    next_query = now + 1.0
            time.sleep(0.05)
        if _cancel:
            reason = "cancelled"
    finally:
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
        verifier = StoppedOdomVerifier(stop_requested_at, stop_requested_ros)
        feedback_deadline = _monotonic() + 2.0
        while _monotonic() < feedback_deadline:
            with lock:
                latest_odom = records.get("odom")
            if latest_odom is not None:
                odom, observed_at = latest_odom
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
        elif not commanded:
            reason = "no_motion_commanded"
        elif not stopped_odometry:
            reason = "stopped_odometry_unverified"
        elif max_translation < 0.01 and max_rotation < 0.03:
            reason = "odometry_motion_not_observed"

    # Requesting zero is not evidence that the physical base stopped.
    return {
        "ok": bool(
            reason == "duration_elapsed"
            and commanded
            and decision["count"]
            and stopped_odometry
            and (max_translation >= 0.01 or max_rotation >= 0.03)
        ),
        "protocol": PROTOCOL,
        "operation": "MAPLESS_EXPLORE",
        "action_id": request["action_id"],
        "reason": reason,
        "decisions_accepted": decision["count"],
        "motion_commanded": commanded,
        "stopped_odometry": stopped_odometry,
        "motion_evidence": {
            "odom_translation_m": max_translation,
            "odom_rotation_rad": max_rotation,
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
