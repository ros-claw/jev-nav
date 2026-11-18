#!/usr/bin/env python2
# mypy: ignore-errors
"""Read one fixed ROS depth topic; return bounded optical-depth statistics."""

import json
import math
import sys

import rospy
from depth_summary_core import summarize
from sensor_msgs.msg import Image

PROTOCOL = "rosclaw.limo.depth-summary-worker.v1"
TOPIC = "/camera/depth/image_raw"


def main():
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict) or request.get("protocol") != PROTOCOL:
            raise ValueError("invalid worker protocol")
        timeout = float(request.get("timeout_sec", 5))
        if math.isnan(timeout) or math.isinf(timeout) or not 0.1 <= timeout <= 10:
            raise ValueError("invalid timeout")
        rospy.init_node("rosclaw_limo_depth_summary", anonymous=True, disable_signals=True)
        msg = rospy.wait_for_message(TOPIC, Image, timeout=timeout)
        summary = summarize(
            msg.data,
            int(msg.width),
            int(msg.height),
            int(msg.step),
            str(msg.encoding),
            bool(msg.is_bigendian),
        )
        summary.update(
            ok=True,
            protocol=PROTOCOL,
            topic=TOPIC,
            encoding=str(msg.encoding),
            frame_id=str(msg.header.frame_id),
            stamp_sec=float(msg.header.stamp.to_sec()),
            width=int(msg.width),
            height=int(msg.height),
        )
    except Exception as exc:
        summary = {"ok": False, "protocol": PROTOCOL, "error": str(exc)}
    print(json.dumps(summary, separators=(",", ":"), allow_nan=False))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
