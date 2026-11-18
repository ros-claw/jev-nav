#!/usr/bin/env python2
"""Fixed-topic read-only synchronized scene capture for the Decisions observer."""

import base64
import hashlib
import json
import math
import sys
import threading
import time

try:
    from .scene_observation_core import make_snapshot, scene_digest
except (ImportError, ValueError):
    from scene_observation_core import make_snapshot, scene_digest

try:
    NUMBER_TYPES = (int, long, float)  # noqa: F821 - Python 2 ROS worker
except NameError:
    NUMBER_TYPES = (int, float)

PROTOCOL = "rosclaw.limo.scene-observer.v1"


def validate_request(request):
    if (
        not isinstance(request, dict)
        or set(request) != {"protocol", "timeout_sec"}
        or request.get("protocol") != PROTOCOL
    ):
        raise ValueError("invalid scene request")
    timeout = request["timeout_sec"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, NUMBER_TYPES)
        or math.isnan(timeout)
        or math.isinf(timeout)
        or not 0.1 <= timeout <= 10
    ):
        raise ValueError("invalid scene timeout")
    return float(timeout)


def depth_sectors(array, encoding):
    import numpy as np

    if (
        len(array.shape) != 2
        or array.shape[1] < 3
        or array.shape[0] < 1
        or array.size > 1920 * 1080
        or encoding not in ("16UC1", "32FC1")
    ):
        raise ValueError("invalid depth image")
    values = array.astype(np.float32)
    if encoding == "16UC1":
        values *= 0.001
    result = {}
    width = values.shape[1]
    for index, name in enumerate(("left", "center", "right")):
        columns = values[:, index * width // 3 : (index + 1) * width // 3]
        valid = np.isfinite(columns) & (columns > 0) & (columns <= 20)
        count = int(np.count_nonzero(valid))
        result[name] = {
            "minimum_m": float(np.min(columns[valid])) if count else None,
            "valid_ratio": float(count) / columns.size,
        }
    return result


def depth_bands(array, encoding):
    if len(array.shape) != 2 or array.shape[0] < 3:
        raise ValueError("depth image is too small for bands")
    height = array.shape[0]
    return {
        name: depth_sectors(array[index * height // 3 : (index + 1) * height // 3, :], encoding)
        for index, name in enumerate(("top", "middle", "bottom"))
    }


def laser_transform(listener, ros, tf_module, scan, timeout):
    frame = str(scan.header.frame_id)
    if not frame or len(frame) > 128:
        raise ValueError("invalid laser frame")
    listener.waitForTransform("base_link", frame, scan.header.stamp, ros.Duration(timeout))
    translation, quaternion = listener.lookupTransform("base_link", frame, scan.header.stamp)
    roll, pitch, yaw = tf_module.transformations.euler_from_quaternion(quaternion)
    values = [float(translation[0]), float(translation[1]), float(roll), float(pitch), float(yaw)]
    if (
        any(math.isnan(value) or math.isinf(value) for value in values)
        or abs(roll) > 0.1
        or abs(pitch) > 0.1
    ):
        raise ValueError("invalid or tilted laser transform")
    return {
        "source": frame,
        "target": "base_link",
        "stamp": scan.header.stamp.to_sec(),
        "x": values[0],
        "y": values[1],
        "yaw": values[4],
        "verified": False,
    }


def collect(ros, filters, image_type, scan_type, status_type, odom_type, timeout):
    """Collect one approximate ROS-header-time round from fixed topics."""
    ready = threading.Event()
    received = []
    subscriptions = []

    def callback(scan, color, depth, status, odom):
        if not ready.is_set():
            received.append((scan, color, depth, status, odom))
            ready.set()

    try:
        for topic, message_type in (
            ("/scan", scan_type),
            ("/camera/color/image_raw", image_type),
            ("/camera/depth/image_raw", image_type),
            ("/limo_status", status_type),
            ("/odom", odom_type),
        ):
            subscriptions.append(filters.Subscriber(topic, message_type))
        sync = filters.ApproximateTimeSynchronizer(subscriptions, queue_size=15, slop=0.3)
        sync.registerCallback(callback)
        if not ready.wait(timeout):
            raise ValueError("synchronized scene capture timed out")
        return received[0]
    finally:
        for subscription in subscriptions:
            subscription.sub.unregister()


def main():
    try:
        timeout = validate_request(json.load(sys.stdin))
        import cv2
        import message_filters
        import rospy
        import tf
        from cv_bridge import CvBridge
        from limo_base.msg import LimoStatus
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image, LaserScan

        rospy.init_node("rosclaw_limo_scene_observer", anonymous=True, disable_signals=True)
        listener = tf.TransformListener()
        deadline = time.time() + timeout
        for attempt in range(3):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise ValueError("scene capture timed out")
            scan, color, depth, status, odom = collect(
                rospy, message_filters, Image, LaserScan, LimoStatus, Odometry, remaining
            )
            try:
                transform = laser_transform(
                    listener, rospy, tf, scan, max(0.01, min(0.6, deadline - time.time()))
                )
                break
            except Exception:
                if attempt == 2 or time.time() >= deadline:
                    raise
        if (
            color.width <= 0
            or color.height <= 0
            or color.width * color.height > 1920 * 1080
            or odom.header.frame_id.lstrip("/") != "odom"
            or odom.child_frame_id.lstrip("/") != "base_link"
        ):
            raise ValueError("invalid color image or odometry frame")
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(color, desired_encoding="bgr8")
        scale = max(1, int((max(color.width, color.height) + 639) / 640))
        if scale > 1:
            frame = cv2.resize(
                frame,
                ((color.width + scale - 1) // scale, (color.height + scale - 1) // scale),
                interpolation=cv2.INTER_NEAREST,
            )
        ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise ValueError("color image encoding failed")
        png = encoded.tostring()
        if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) > 4000000:
            raise ValueError("invalid encoded color image")
        depth_data = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        stamps = {
            "scan": scan.header.stamp.to_sec(),
            "color": color.header.stamp.to_sec(),
            "depth": depth.header.stamp.to_sec(),
            "status": status.header.stamp.to_sec(),
            "odom": odom.header.stamp.to_sec(),
        }
        data = {
            "stamps": stamps,
            "frames": {
                "laser": str(scan.header.frame_id),
                "color": str(color.header.frame_id),
                "depth": str(depth.header.frame_id),
            },
            "laser_tf": transform,
            "scan": {
                "angle_min": float(scan.angle_min),
                "angle_increment": float(scan.angle_increment),
                "range_min": float(scan.range_min),
                "range_max": float(scan.range_max),
                "time_increment": float(scan.time_increment),
                "scan_time": float(scan.scan_time),
                "ranges": [float(value) for value in scan.ranges],
            },
            "base": {
                "error_code": int(status.error_code),
                "motion_mode": int(status.motion_mode),
                "battery_voltage": float(status.battery_voltage),
                "linear_x_mps": float(odom.twist.twist.linear.x),
                "angular_z_radps": float(odom.twist.twist.angular.z),
            },
            "depth": depth_sectors(depth_data, str(depth.encoding)),
            "depth_bands": depth_bands(depth_data, str(depth.encoding)),
        }
        snapshot = make_snapshot(data, rospy.Time.now().to_sec(), time.time())
        response = {
            "ok": True,
            "protocol": PROTOCOL,
            "snapshot": snapshot,
            "diagnostic_only": True,
            "command_dispatched": False,
            "usable_for_real_execution": False,
            "sync": "ros_header_approximate",
            "png_base64": base64.b64encode(png).decode("ascii"),
            "png_sha256": "sha256:" + hashlib.sha256(png).hexdigest(),
            "scene_sha256": scene_digest(png, snapshot),
        }
    except Exception as exc:
        response = {"ok": False, "protocol": PROTOCOL, "error": str(exc)[:300]}
    print(json.dumps(response, separators=(",", ":"), allow_nan=False))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
