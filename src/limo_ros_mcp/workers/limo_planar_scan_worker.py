#!/usr/bin/env python2
# mypy: ignore-errors
"""Fixed /scan and base_link TF read-only diagnostic; no motor interfaces."""

import json
import math
import sys
import time

from limo_calibration_diagnostic_worker import _lookup_matrix
from local_observation_core import _frame
from planar_scan_core import make_planar_scan

PROTOCOL = "rosclaw.limo.planar-scan-diagnostic.v1"


def validate_request(request):
    if (
        not isinstance(request, dict)
        or set(request) != {"protocol", "timeout_sec"}
        or request.get("protocol") != PROTOCOL
    ):
        raise ValueError("invalid planar scan request")
    timeout = request["timeout_sec"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("invalid timeout")
    timeout = float(timeout)
    if math.isnan(timeout) or math.isinf(timeout) or not 0.1 <= timeout <= 10:
        raise ValueError("invalid timeout")
    return timeout


def collect(ros, tf_module, laser_type, timeout, clock=time.time):
    listener = tf_module.TransformListener()
    started = clock()
    transform = None
    last_error = None
    for _ in range(3):
        current = clock()
        remaining = timeout - (current - started)
        if current < started or remaining <= 0:
            break
        scan = ros.wait_for_message("/scan", laser_type, timeout=remaining)
        source = _frame(str(scan.header.frame_id))
        try:
            current = clock()
            remaining = timeout - (current - started)
            if current < started or remaining <= 0:
                break
            transform = _lookup_matrix(
                listener,
                tf_module,
                ros,
                "base_link",
                source,
                scan.header.stamp,
                min(remaining, 1.0),
            )
            break
        except Exception as exc:  # noqa: BLE001 - retry a fresh scan, never substitute a transform
            last_error = exc
    if transform is None:
        raise ValueError("timestamped laser-to-base TF unavailable: " + str(last_error)[:180])
    values = {
        "stamp": scan.header.stamp.to_sec(),
        "frame_id": source,
        "angle_min": scan.angle_min,
        "angle_increment": scan.angle_increment,
        "range_min": scan.range_min,
        "range_max": scan.range_max,
        "time_increment": scan.time_increment,
        "scan_time": scan.scan_time,
        "ranges": list(scan.ranges),
    }
    return make_planar_scan(values, transform, now=ros.Time.now().to_sec())


def main():
    try:
        timeout = validate_request(json.load(sys.stdin))
        import rospy
        import tf
        from sensor_msgs.msg import LaserScan

        rospy.init_node("rosclaw_limo_planar_scan_diagnostic", anonymous=True, disable_signals=True)
        result = collect(rospy, tf, LaserScan, timeout)
        result.update(ok=True, protocol=PROTOCOL)
    except Exception as exc:
        result = {"ok": False, "protocol": PROTOCOL, "error": str(exc)}
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
