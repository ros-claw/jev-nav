#!/usr/bin/env python2
# mypy: ignore-errors
"""Fixed-topic read-only ROS observation bundle; no actuation interfaces."""

import json
import math
import sys
import threading

from local_observation_core import make_bundle

PROTOCOL = "rosclaw.limo.local-observation.v1"


def validate_request(request):
    if (
        not isinstance(request, dict)
        or set(request) != {"protocol", "timeout_sec"}
        or request.get("protocol") != PROTOCOL
    ):
        raise ValueError("invalid observation request")
    timeout = request["timeout_sec"]
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)):
        raise ValueError("invalid timeout")
    timeout = float(timeout)
    if math.isnan(timeout) or math.isinf(timeout) or not 0.1 <= timeout <= 10:
        raise ValueError("invalid timeout")
    return timeout


def _transform(listener, tf_module, ros, frame, stamp, timeout):
    listener.waitForTransform("base_link", frame, stamp, ros.Duration(min(timeout, 2.0)))
    translation, quaternion = listener.lookupTransform("base_link", frame, stamp)
    matrix = tf_module.transformations.quaternion_matrix(quaternion)
    matrix = [[float(matrix[i][j]) for j in range(4)] for i in range(4)]
    for i in range(3):
        matrix[i][3] = float(translation[i])
    return matrix


def _calibration_diagnostic(info):
    """Summarize numerical metadata only; do not validate calibration."""
    try:
        rotation = [float(value) for value in info.R]
        intrinsics = [float(value) for value in info.K]
        projection = [float(value) for value in info.P]
        if (
            len(rotation) != 9
            or len(intrinsics) != 9
            or len(projection) != 12
            or any(
                math.isnan(value) or math.isinf(value)
                for value in rotation + intrinsics + projection
            )
        ):
            return "calibration_diagnostic=unavailable"
        error = max(
            abs(value - (1.0 if index in (0, 4, 8) else 0.0))
            for index, value in enumerate(rotation)
        )
        return (  # noqa: UP032 - ROS Melodic worker runs on Python 2
            "rectification_max_error={:.6g}; K(fx,fy,cx,cy)=[{:.6g},{:.6g},{:.6g},{:.6g}]; "
            "P(fx,fy,cx,cy)=[{:.6g},{:.6g},{:.6g},{:.6g}]"
        ).format(
            error,
            intrinsics[0],
            intrinsics[4],
            intrinsics[2],
            intrinsics[5],
            projection[0],
            projection[5],
            projection[2],
            projection[6],
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "calibration_diagnostic=unavailable"


def _synchronized_messages(filters, laser_scan_type, image_type, camera_info_type, timeout):
    """Acquire only fixed-topic, approximately simultaneous ROS messages."""
    subscriptions = []
    result = []
    ready = threading.Event()

    def on_messages(scan, image, info):
        if not ready.is_set():
            result.append((scan, image, info))
            ready.set()

    try:
        for topic, message_type in (
            ("/scan", laser_scan_type),
            ("/camera/depth/image_raw", image_type),
            ("/camera/depth/camera_info", camera_info_type),
        ):
            subscriptions.append(filters.Subscriber(topic, message_type))
        synchronizer = filters.ApproximateTimeSynchronizer(subscriptions, queue_size=10, slop=0.1)
        synchronizer.registerCallback(on_messages)
        if not ready.wait(timeout):
            raise ValueError("synchronized sensor capture timed out")
        return result[0]
    finally:
        for subscription in subscriptions:
            subscription.sub.unregister()


def collect(ros, tf_module, filters, laser_scan_type, image_type, camera_info_type, timeout):
    listener = tf_module.TransformListener()
    scan, image, info = _synchronized_messages(
        filters, laser_scan_type, image_type, camera_info_type, timeout
    )
    if (
        image.header.frame_id != info.header.frame_id
        or image.width != info.width
        or image.height != info.height
    ):
        raise ValueError(
            (  # noqa: UP032 - ROS Melodic worker runs on Python 2
                "depth image and CameraInfo frame or dimensions mismatch: "
                "image frame={!r} size={}x{}; info frame={!r} size={}x{}"
            ).format(
                str(image.header.frame_id)[:128],
                image.width,
                image.height,
                str(info.header.frame_id)[:128],
                info.width,
                info.height,
            )
            + "; "
            + _calibration_diagnostic(info)
        )
    if image.is_bigendian not in (0, 1):
        raise ValueError("invalid depth endian flag")
    scan_matrix = _transform(
        listener, tf_module, ros, scan.header.frame_id, scan.header.stamp, timeout
    )
    depth_matrix = _transform(
        listener, tf_module, ros, image.header.frame_id, image.header.stamp, timeout
    )
    scan_data = {
        "stamp": scan.header.stamp.to_sec(),
        "frame_id": str(scan.header.frame_id),
        "angle_min": scan.angle_min,
        "angle_increment": scan.angle_increment,
        "range_min": scan.range_min,
        "range_max": scan.range_max,
        "ranges": list(scan.ranges),
    }
    depth_data = {
        "stamp": image.header.stamp.to_sec(),
        "frame_id": str(image.header.frame_id),
        "width": int(image.width),
        "height": int(image.height),
        "step": int(image.step),
        "encoding": str(image.encoding),
        "is_bigendian": bool(image.is_bigendian),
        "data": image.data,
    }
    info_data = {
        "stamp": info.header.stamp.to_sec(),
        "frame_id": str(info.header.frame_id),
        "width": int(info.width),
        "height": int(info.height),
        "K": list(info.K),
        "D": list(info.D),
        "R": list(info.R),
        "P": list(info.P),
        "binning_x": int(info.binning_x),
        "binning_y": int(info.binning_y),
        "roi": {
            "x_offset": int(info.roi.x_offset),
            "y_offset": int(info.roi.y_offset),
            "width": int(info.roi.width),
            "height": int(info.roi.height),
            "do_rectify": bool(info.roi.do_rectify),
        },
    }
    return make_bundle(
        scan_data, depth_data, info_data, scan_matrix, depth_matrix, now=ros.Time.now().to_sec()
    )


def main():
    try:
        timeout = validate_request(json.load(sys.stdin))
        import message_filters
        import rospy
        import tf
        from sensor_msgs.msg import CameraInfo, Image, LaserScan

        rospy.init_node("rosclaw_limo_local_observation", anonymous=True, disable_signals=True)
        result = collect(rospy, tf, message_filters, LaserScan, Image, CameraInfo, timeout)
        result.update(ok=True, protocol=PROTOCOL)
    except Exception as exc:
        result = {"ok": False, "protocol": PROTOCOL, "error": str(exc)}
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
