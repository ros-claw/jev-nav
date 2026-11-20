#!/usr/bin/env python2
"""Replay real recorded frames offline through separate diagnostic processors.

TF is preloaded from the whole bag so lookups can use transforms that had not
yet arrived at the frame's original receipt time. This is intentionally
optimistic offline interpolation, not live TF latency or a motion permit.
The supplied ground height and rectification are assumptions, not calibration.
No ROS publisher, model request, serial device, or physical motion is used.
"""
from __future__ import division, print_function

import argparse
import collections
import json
import math
import os
import sys
import time

from frame_replay import replay_frames
from bag_tf import cache_duration, populate_transforms
from independent_perception import DepthProcessor, ScanProcessor
from nav_timeline import Timeline
from relay import _monotonic
from sensing_timeline import NativeMeter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag')
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--ground-z', type=float, default=-.1)
    args = parser.parse_args(argv)
    if not -.5 <= args.ground_z <= 0:
        parser.error('invalid ground-height assumption')
    import rosbag
    import rospy
    import tf
    import limo_scan_fusion
    from cv_bridge import CvBridge
    from explore import lookup_rigid_transform, nearest_odometry, transform_matches

    bridge = CvBridge()
    with rosbag.Bag(args.bag) as bag:
        transform_tree = tf.TransformerROS(True, rospy.Duration(cache_duration(bag.get_start_time(),
                                                                               bag.get_end_time())))
        tf_failures = populate_transforms(transform_tree, bag.read_messages(topics=['/tf', '/tf_static']))
    config = {'forward_min_m': .18, 'forward_max_m': 1., 'half_width_m': .15,
              'ground_z_m': args.ground_z, 'ground_tolerance_m': .025,
              'obstacle_min_height_m': .03, 'obstacle_max_height_m': .4,
              'min_depth_m': .3, 'max_depth_m': 3., 'sample_stride': 4}
    native = limo_scan_fusion._native_scan
    meter = NativeMeter(native) if native is not None else None
    if meter is not None:
        limo_scan_fusion._native_scan = meter
    depth_processor, scan_processor = DepthProcessor(config), None
    odometry = collections.deque(maxlen=300)
    info = None
    depth_signature = laser_signature = None
    available = {'scan': 0., 'depth': 0.}
    counts, frames = collections.Counter(), []
    timeline = Timeline(args.output_prefix + '.jsonl', 'bag-' + str(int(time.time())))
    complete = False
    try:
        with rosbag.Bag(args.bag) as bag:
            for topic, message, bag_time in bag.read_messages():
                received = bag_time.to_sec()
                if topic == '/odom':
                    position = message.pose.pose.position
                    q = message.pose.pose.orientation
                    yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
                    stamp = message.header.stamp.to_sec()
                    pose = (position.x, position.y, yaw)
                    odometry.append({'stamp': stamp, 'pose': pose, 'epoch': 0})
                    timeline.record('odom', received, {'clock_domain': 'ros_bag_time', 'odom_at': stamp,
                                    'sensor_received_at': received, 'x': pose[0], 'y': pose[1], 'yaw': yaw,
                                    'linear_velocity': message.twist.twist.linear.x,
                                    'angular_velocity': message.twist.twist.angular.z})
                    continue
                if topic == '/camera/depth/camera_info':
                    info = message
                    continue
                if topic not in ('/scan', '/camera/depth/image_raw'):
                    continue
                stream = 'scan' if topic == '/scan' else 'depth'
                stamp = message.header.stamp.to_sec()
                processing_at = max(received, available[stream])
                started = _monotonic()
                result = {'clear': None, 'reason': 'unavailable', 'valid_until': None}
                error_class = None
                if meter is not None:
                    meter.used, meter.reason = False, 'not_called'
                try:
                    aligned = nearest_odometry(list(odometry), stamp)
                    if aligned is None or not odometry:
                        raise ValueError('odom_alignment')
                    current = odometry[-1]
                    if stream == 'scan':
                        trans, rotation = transform_tree.lookupTransform('base_link', message.header.frame_id,
                                                                          message.header.stamp)
                        roll, pitch, yaw = tf.transformations.euler_from_quaternion(rotation)
                        signature = tuple(trans) + (roll, pitch, yaw)
                        if abs(roll) > .1 or abs(pitch) > .1:
                            raise ValueError('laser_tilt')
                        if scan_processor is None:
                            laser_signature = signature
                            scan_processor = ScanProcessor((trans[0], trans[1], yaw))
                        if not transform_matches(laser_signature, signature):
                            raise ValueError('laser_transform_changed')
                        raw = {'stamp': stamp, 'frame_id': message.header.frame_id,
                               'ranges': list(message.ranges), 'angle_min': message.angle_min,
                               'angle_increment': message.angle_increment, 'range_min': message.range_min,
                               'range_max': message.range_max, 'time_increment': message.time_increment}
                        result = scan_processor.process(raw, aligned['pose'], aligned['stamp'],
                                                        current['pose'], current['stamp'], processing_at)
                    else:
                        if (info is None or info.header.frame_id != message.header.frame_id
                                or info.width != message.width or info.height != message.height
                                or abs(info.header.stamp.to_sec() - stamp) > .2):
                            raise ValueError('camera_info')
                        dx = aligned['pose'][0] - current['pose'][0]
                        dy = aligned['pose'][1] - current['pose'][1]
                        da = aligned['pose'][2] - current['pose'][2]
                        if math.hypot(dx, dy) > .01 or abs(math.atan2(math.sin(da), math.cos(da))) > .02:
                            raise ValueError('depth_pose_changed')
                        trans, rotation = lookup_rigid_transform(transform_tree, tf, rospy,
                                                                 message.header.frame_id, message.header.stamp,
                                                                 depth_signature)
                        signature = tuple(trans) + tuple(tf.transformations.euler_from_quaternion(rotation))
                        if depth_signature is None:
                            depth_signature = signature
                        if not transform_matches(depth_signature, signature):
                            raise ValueError('depth_transform_changed')
                        matrix = tf.transformations.quaternion_matrix(rotation)
                        matrix[:3, 3] = trans
                        k = info.K
                        intrinsic = {'width': message.width, 'height': message.height,
                                     'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5], 'rectified': True}
                        array = bridge.imgmsg_to_cv2(message, desired_encoding='passthrough')
                        result = depth_processor.process(array, message.encoding, intrinsic, matrix,
                                                         stamp, processing_at)
                except Exception as exc:  # diagnostic: retain type, never raw message
                    error_class = type(exc).__name__
                    result['reason'] = 'processing_exception'
                elapsed = _monotonic() - started
                completed = processing_at + elapsed
                available[stream] = completed
                deadline = result.get('valid_until')
                frames.append({'stream': stream, 'acquired': stamp, 'received': received,
                               'completed': completed, 'valid_until': deadline,
                               'clear': result['clear'], 'reason': result['reason']})
                status = 'candidate' if result['clear'] is True else 'obstacle' if result['clear'] is False else 'unknown'
                counts[stream + ':' + result['reason'] + ':' + status] += 1
                fields = {'clock_domain': 'ros_bag_time', 'stage': stream, 'status': status,
                          'sensor_acquired_at': stamp, 'sensor_received_at': received,
                          'perception_start': processing_at, 'perception_end': completed,
                          'processing_sec': elapsed, 'queue_delay_sec': processing_at - received,
                          'valid_until': deadline, 'reason': result['reason']}
                for name in ('ray_count', 'retained_scan_count', 'invalid_count', 'unobserved_space',
                             'sampled_candidate'):
                    if name in result:
                        fields[name] = result[name]
                if error_class is not None:
                    fields['error_class'] = error_class
                if stream == 'scan':
                    fields['accel_used'] = meter.used if meter is not None else False
                    fields['fallback_reason'] = meter.reason if meter is not None else 'native_unavailable'
                timeline.record('sensor', completed, fields)
    finally:
        complete = timeline.close(5.)
    gate_result = replay_frames(frames)
    output = {'mode': 'offline_recorded_frames', 'motion_requested': False,
              'tf_preloaded_from_future': True, 'tf_load_failures': tf_failures,
              'odom_continuity_checked': False, 'base_status_checked': False, 'calibration_verified': False,
              'timeline_complete': complete, 'dropped': timeline.dropped,
              'counts': dict(counts), 'replay': gate_result}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(args.output_prefix + '.json', flags, 0o600)
    with os.fdopen(fd, 'w') as output_file:
        json.dump(output, output_file, sort_keys=True, allow_nan=False)
    print(json.dumps({'motion_requested': False, 'counts': dict(counts), 'timeline_complete': complete,
                      'accepted_hypothetical_commands': gate_result['accepted_commands'],
                      'longest_hypothetical_segment_sec': gate_result['longest_segment_sec']}, sort_keys=True))
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
