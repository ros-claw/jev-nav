#!/usr/bin/env python2
"""Read-only depth point diagnostic using explicitly assumed calibration."""
from __future__ import print_function

import argparse
import json

from depth_corridor import diagnose_depth_corridor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assume-rectified", action="store_true", required=True)
    parser.add_argument("--ground-z", type=float, required=True)
    parser.add_argument("--ground-tolerance", type=float, default=0.025)
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.frames <= 30 or not 0 <= args.ground_tolerance < 0.03:
        parser.error("invalid frame count or ground tolerance")
    import rospy
    import tf
    from cv_bridge import CvBridge
    from sensor_msgs.msg import CameraInfo, Image

    rospy.init_node("limo_depth_corridor_probe", anonymous=True, disable_signals=True)
    listener = tf.TransformListener()
    bridge = CvBridge()
    first = rospy.wait_for_message("/camera/depth/image_raw", Image, timeout=5)
    listener.waitForTransform("base_link", first.header.frame_id, rospy.Time(0), rospy.Duration(4))
    rospy.sleep(0.2)
    config = {"forward_min_m": 0.18, "forward_max_m": 1.0, "half_width_m": 0.15,
              "ground_z_m": args.ground_z, "ground_tolerance_m": args.ground_tolerance,
              "obstacle_min_height_m": 0.03, "obstacle_max_height_m": 0.4,
              "min_depth_m": 0.3, "max_depth_m": 3.0, "sample_stride": 2}
    for _ in range(args.frames):
        depth = rospy.wait_for_message("/camera/depth/image_raw", Image, timeout=5)
        info = rospy.wait_for_message("/camera/depth/camera_info", CameraInfo, timeout=5)
        if (depth.header.frame_id != info.header.frame_id or depth.width != info.width
                or depth.height != info.height
                or abs((depth.header.stamp - info.header.stamp).to_sec()) > 0.2):
            raise ValueError("depth image and camera info do not align")
        listener.waitForTransform("base_link", depth.header.frame_id, depth.header.stamp,
                                  rospy.Duration(4))
        translation, rotation = listener.lookupTransform("base_link", depth.header.frame_id,
                                                          depth.header.stamp)
        matrix = tf.transformations.quaternion_matrix(rotation)
        matrix[:3, 3] = translation
        k = info.K
        intrinsics = {"width": depth.width, "height": depth.height,
                      "fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5],
                      "rectified": args.assume_rectified}
        result = diagnose_depth_corridor(bridge.imgmsg_to_cv2(depth, "passthrough"),
                                         depth.encoding, intrinsics, matrix, config)
        result["calibration_verified"] = False
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
