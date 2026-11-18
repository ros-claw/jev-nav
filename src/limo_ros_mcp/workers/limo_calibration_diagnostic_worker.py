#!/usr/bin/env python2
# mypy: ignore-errors
"""Fixed-topic read-only calibration diagnostics; never robot control."""

import json
import math
import sys
import time

from calibration_bundle_core import CAMERA_FRAMES, make_diagnostic
from limo_local_observation_worker import _synchronized_messages

PROTOCOL = "rosclaw.limo.calibration-diagnostic.v1"


def validate_request(request):
    if (
        not isinstance(request, dict)
        or set(request) != {"protocol", "timeout_sec"}
        or request.get("protocol") != PROTOCOL
    ):
        raise ValueError("invalid diagnostic request")
    timeout = request["timeout_sec"]
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)):
        raise ValueError("invalid timeout")
    timeout = float(timeout)
    if math.isnan(timeout) or math.isinf(timeout) or not 0.1 <= timeout <= 10:
        raise ValueError("invalid timeout")
    return timeout


def _lookup_matrix(listener, tf_module, ros, target, source, stamp, timeout):
    listener.waitForTransform(target, source, stamp, ros.Duration(timeout))
    translation, quaternion = listener.lookupTransform(target, source, stamp)
    transform = tf_module.transformations.quaternion_matrix(quaternion)
    transform = [[float(transform[i][j]) for j in range(4)] for i in range(4)]
    for i in range(3):
        transform[i][3] = float(translation[i])
    return transform


def collect(ros, tf_module, filters, laser_type, image_type, info_type, timeout):
    listener = tf_module.TransformListener()
    deadline = time.time() + timeout
    for attempt in range(3):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise ValueError("synchronized capture timed out")
        scan, image, info = _synchronized_messages(
            filters, laser_type, image_type, info_type, remaining
        )
        if image.header.frame_id not in CAMERA_FRAMES:
            raise ValueError("unexpected claimed camera optical frame")
        if image.is_bigendian not in (0, 1):
            raise ValueError("invalid depth endian flag")
        target, source = image.header.frame_id, scan.header.frame_id
        try:
            transform = _lookup_matrix(
                listener, tf_module, ros, target, source, scan.header.stamp,
                max(0.01, min(deadline - time.time(), 2.0)),
            )
            break
        except Exception:  # noqa: BLE001 - the first scan may predate this TF listener
            if attempt == 2 or time.time() >= deadline:
                raise
            time.sleep(min(0.2, max(0.0, deadline - time.time())))
    try:
        base_transform = _lookup_matrix(
            listener, tf_module, ros, "base_link", source, scan.header.stamp, min(timeout, 1.0)
        )
    except Exception:  # noqa: BLE001 - optional TF must not fabricate a fallback
        base_transform = None
    scan_data = {
        "stamp": scan.header.stamp.to_sec(),
        "frame_id": str(scan.header.frame_id),
        "angle_min": scan.angle_min,
        "angle_increment": scan.angle_increment,
        "range_min": scan.range_min,
        "range_max": scan.range_max,
        "ranges": list(scan.ranges),
        "time_increment": scan.time_increment,
        "scan_time": scan.scan_time,
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
        "distortion_model": str(info.distortion_model),
        "K": list(info.K),
        "D": list(info.D),
        "R": list(info.R),
        "P": list(info.P),
    }
    return make_diagnostic(
        scan_data,
        depth_data,
        info_data,
        transform,
        now=ros.Time.now().to_sec(),
        scan_to_base=base_transform,
    )


def main():
    try:
        timeout = validate_request(json.load(sys.stdin))
        import message_filters
        import rospy
        import tf
        from sensor_msgs.msg import CameraInfo, Image, LaserScan

        rospy.init_node("rosclaw_limo_calibration_diagnostic", anonymous=True, disable_signals=True)
        result = collect(rospy, tf, message_filters, LaserScan, Image, CameraInfo, timeout)
        result.update(ok=True, protocol=PROTOCOL)
    except Exception as exc:
        result = {"ok": False, "protocol": PROTOCOL, "error": str(exc)}
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
