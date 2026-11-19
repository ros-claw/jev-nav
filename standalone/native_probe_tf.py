#!/usr/bin/env python2
"""Populate an isolated probe TF topic; never publish to /tf or /cmd_vel."""
from __future__ import division

import argparse
import math
import re

_NAMESPACE = re.compile(r'^limo_native_probe_[0-9a-f]{12}\Z')
_ALLOWED = frozenset((('base_link', 'laser_link'), ('base_link', 'camera_link'),
                      ('camera_link', 'camera_depth_frame'),
                      ('camera_depth_frame', 'camera_depth_optical_frame')))


def valid_namespace(value):
    return isinstance(value, str) and _NAMESPACE.match(value) is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', required=True)
    args = parser.parse_args()
    if not valid_namespace(args.namespace):
        parser.error('invalid private probe namespace')
    import rospy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from tf2_msgs.msg import TFMessage

    rospy.init_node('limo_native_probe_tf', anonymous=True)
    publisher = rospy.Publisher('/' + args.namespace + '/tf', TFMessage, queue_size=20)

    def publish(message):
        try:
            if not rospy.is_shutdown():
                publisher.publish(message)
        except rospy.ROSException:
            if not rospy.is_shutdown():
                raise

    def receive(message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        age = (rospy.Time.now() - message.header.stamp).to_sec()
        if (message.header.frame_id.lstrip('/') != 'odom'
                or message.child_frame_id.lstrip('/') != 'base_link'
                or not 0 <= age <= .2
                or not all(not math.isnan(value) and not math.isinf(value) for value in values)
                or not .99 <= q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w <= 1.01):
            return
        transform = TransformStamped()
        transform.header = message.header
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = p.x
        transform.transform.translation.y = p.y
        transform.transform.translation.z = p.z
        transform.transform.rotation = q
        publish(TFMessage([transform]))

    def receive_tf(message):
        selected = [entry for entry in message.transforms
                    if (entry.header.frame_id.lstrip('/'), entry.child_frame_id.lstrip('/')) in _ALLOWED]
        if selected:
            publish(TFMessage(selected))

    rospy.Subscriber('/odom', Odometry, receive, queue_size=1)
    rospy.Subscriber('/tf', TFMessage, receive_tf, queue_size=20)
    rospy.spin()


if __name__ == '__main__':
    main()
