#!/usr/bin/env python2
"""Read-only parallel ROS frame processing; never creates a velocity publisher.

Assessments are sampled diagnostic evidence, not full clearance or motion
permission. Camera calibration and floor assumptions remain unverified.
"""
from __future__ import division, print_function

import argparse
import collections
import json
import math
import sys
import time

from independent_perception import DepthProcessor, ScanProcessor
from independent_worker import IndependentWorkers
from nav_timeline import Timeline
from relay import _monotonic


class Clock(object):
    def now(self):
        return _monotonic()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeline', required=True)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--ground-z', type=float, default=-.1)
    parser.add_argument('--stream', choices=('both', 'scan', 'depth'), default='both')
    parser.add_argument('--events-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 300 or not -.5 <= args.ground_z <= 0:
        parser.error('invalid duration or ground-height assumption')
    if args.events_fd is not None and (args.events_fd < 3 or args.stream == 'both'):
        parser.error('private event pipe requires one sensor stream and fd >= 3')
    return args


def poll_for(seconds, clock, is_shutdown, poll, sleep):
    started = clock.now()
    try:
        while clock.now() - started < seconds and not is_shutdown():
            if poll() is not True:
                return 'fault'
            sleep(.05)
    except KeyboardInterrupt:
        return 'interrupted'
    return 'complete'


def aligned_odometry(observation, sensor_stamp, epoch, clock):
    """Capture one current odometry view without reprocessing sensor frames."""
    from explore import nearest_odometry
    with observation.lock:
        now = clock.now()
        if (observation.epoch != epoch or observation.clock_guard.faulted or observation.odom_guard.faulted):
            return None, None, now
        odom = observation.latest.get('odom')
        history = list(observation.odom_history)[-150:]
    aligned = nearest_odometry(history, sensor_stamp)
    if (odom is None or aligned is None or aligned.get('epoch') != epoch or odom.get('epoch') != epoch
            or not 0 <= now - odom['stamp'] <= .15):
        return None, None, now
    return aligned, odom, now


def main(argv=None):
    args = parse_args(argv)
    import rospy
    import tf
    from explore import RosObservation, lookup_rigid_transform, transform_matches

    rospy.init_node('jev_independent_sensing_probe', anonymous=True, disable_signals=True)
    if rospy.get_param('/use_sim_time', False):
        raise ValueError('wall clock is required')
    clock = Clock()
    timeline = Timeline(args.timeline, 'independent-' + str(int(time.time())))
    if args.events_fd is not None:
        from frame_pipe import FrameWriter, PipeTimeline
        timeline = PipeTimeline(timeline, FrameWriter(args.events_fd, args.stream), args.stream)
    config = {'forward_min_m': .18, 'forward_max_m': 1., 'half_width_m': .15,
              'ground_z_m': args.ground_z, 'ground_tolerance_m': .025,
              'obstacle_min_height_m': .03, 'obstacle_max_height_m': .4,
              'min_depth_m': .3, 'max_depth_m': 3., 'sample_stride': 4}
    depth_processor = DepthProcessor(config)
    contexts = {'scan': {'processor': None, 'signature': None}, 'depth': {'signature': None}}

    class StreamingObservation(RosObservation):
        def __init__(self):
            self.nav_workers = None
            self.recording = False
            RosObservation.__init__(self, depth_projection_config=config)

        def _receive(self, message, name):
            callback_at = _monotonic()
            with self.lock:
                before_epoch = self.epoch
            RosObservation._receive(self, message, name)
            if not self.recording:
                return
            with self.lock:
                epoch_after = self.epoch
                issue = self.epoch_issue
            if epoch_after != before_epoch:
                fields = {'clock_domain': 'monotonic', 'epoch': epoch_after,
                          'stage': issue['sensor'] if issue else 'unknown',
                          'reason': issue['reason'] if issue else 'epoch_changed'}
                if issue is not None and 'sensor_age_sec' in issue:
                    fields['sensor_age_sec'] = issue['sensor_age_sec']
                timeline.record('stop', _monotonic(), fields)
                self.nav_workers.fail('epoch_changed')
                return
            if name == 'odom':
                with self.lock:
                    entry = self.latest.get('odom')
                    if entry is None or entry.get('message') is not message:
                        return
                    stamp, received, pose, epoch = entry['stamp'], entry['received_at'], entry['pose'], self.epoch
                timeline.record('odom', received, {'clock_domain': 'monotonic', 'odom_at': stamp,
                                'sensor_received_at': received, 'callback_at': callback_at, 'epoch': epoch,
                                'x': pose[0], 'y': pose[1], 'yaw': pose[2],
                                'linear_velocity': message.twist.twist.linear.x,
                                'angular_velocity': message.twist.twist.angular.z})
                return
            if name not in self.nav_workers.queues:
                return
            with self.lock:
                entry = self.latest.get(name)
                if (entry is None or entry.get('message') is not message or
                        self.clock_guard.faulted or self.odom_guard.faulted):
                    self.nav_workers.fail('input_invalid')
                    return
                stamp, received, epoch = entry['stamp'], entry['received_at'], self.epoch
                payload = {'message': message, 'stamp': stamp, 'epoch': epoch,
                           'callback_at': callback_at,
                           'info': self.latest.get('info')}
            self.nav_workers.submit(name, stamp, received, epoch, payload)

    observation = StreamingObservation()
    def scan_frame(payload, now):
        message, stamp = payload['message'], payload['stamp']
        aligned, odom, now = aligned_odometry(observation, stamp, payload['epoch'], clock)
        if aligned is None:
            return {'clear': None, 'reason': 'scan_odom_alignment', 'valid_until': None}
        try:
            trans, rotation = observation.listener.lookupTransform('base_link', message.header.frame_id,
                                                                   message.header.stamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return {'clear': None, 'reason': 'scan_transform_unavailable', 'valid_until': None}
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(rotation)
        signature = tuple(trans) + (roll, pitch, yaw)
        if abs(roll) > .1 or abs(pitch) > .1:
            return {'clear': None, 'reason': 'scan_transform_invalid', 'valid_until': None}
        context = contexts['scan']
        if context['processor'] is None:
            context['signature'] = signature
            context['processor'] = ScanProcessor((trans[0], trans[1], yaw))
        if not transform_matches(context['signature'], signature):
            return {'clear': None, 'reason': 'scan_transform_changed', 'valid_until': None}
        raw = {'stamp': stamp, 'frame_id': message.header.frame_id, 'ranges': list(message.ranges),
               'angle_min': message.angle_min, 'angle_increment': message.angle_increment,
               'range_min': message.range_min, 'range_max': message.range_max,
               'time_increment': message.time_increment}
        return context['processor'].process(raw, aligned['pose'], aligned['stamp'],
                                            odom['pose'], odom['stamp'], now)

    def depth_frame(payload, now):
        message, stamp, info_entry = payload['message'], payload['stamp'], payload['info']
        aligned, odom, now = aligned_odometry(observation, stamp, payload['epoch'], clock)
        if aligned is None or info_entry is None:
            return {'clear': None, 'reason': 'depth_odom_alignment', 'valid_until': None}
        info = info_entry['message']
        if (message.header.frame_id.lstrip('/') != 'camera_depth_optical_frame'
                or message.header.frame_id != info.header.frame_id or message.width != info.width
                or message.height != info.height or abs(stamp - info_entry['stamp']) > .2):
            return {'clear': None, 'reason': 'camera_alignment', 'valid_until': None}
        distance = math.hypot(aligned['pose'][0] - odom['pose'][0], aligned['pose'][1] - odom['pose'][1])
        delta = aligned['pose'][2] - odom['pose'][2]
        if distance > .01 or abs(math.atan2(math.sin(delta), math.cos(delta))) > .02:
            return {'clear': None, 'reason': 'depth_pose_changed', 'valid_until': None}
        context = contexts['depth']
        try:
            trans, rotation = lookup_rigid_transform(observation.listener, tf, rospy, message.header.frame_id,
                                                     message.header.stamp, context['signature'])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, ValueError):
            return {'clear': None, 'reason': 'depth_transform_unavailable', 'valid_until': None}
        signature = tuple(trans) + tuple(tf.transformations.euler_from_quaternion(rotation))
        if context['signature'] is None:
            context['signature'] = signature
        if not transform_matches(context['signature'], signature):
            return {'clear': None, 'reason': 'depth_transform_changed', 'valid_until': None}
        matrix = tf.transformations.quaternion_matrix(rotation)
        matrix[:3, 3] = trans
        k = info.K
        intrinsic = {'width': message.width, 'height': message.height, 'fx': k[0], 'fy': k[4],
                     'cx': k[2], 'cy': k[5], 'rectified': True}
        array = observation.bridge.imgmsg_to_cv2(message, desired_encoding='passthrough')
        return depth_processor.process(array, message.encoding, intrinsic, matrix, stamp, now)

    processors = {'scan': scan_frame, 'depth': depth_frame}
    ages = {'scan': .3, 'depth': .15}
    if args.stream != 'both':
        processors = {args.stream: processors[args.stream]}
        ages = {args.stream: ages[args.stream]}
    workers = IndependentWorkers(ages, .2, processors, clock, timeline)
    observation.nav_workers = workers
    counts = collections.Counter()
    complete = False
    outcome = 'not_started'
    try:
        # Allow subscriptions and the TF buffer to fill without processing
        # frames or creating velocity output.
        rospy.rostime.wallsleep(2.)
        workers.start()
        observation.recording = True
        def poll_once():
            with observation.lock:
                epoch = observation.epoch
            state = workers.read_current(epoch)
            now = state['checked_at']
            counts[state['state'] + ':' + state['reason']] += 1
            if not timeline.record('control', now, {'clock_domain': 'monotonic', 'epoch': epoch,
                                   'status': state['state'], 'reason': state['reason'],
                                   'pending': state['pending'], 'valid_until': state['valid_until']}):
                workers.fail('timeline_incomplete')
                return False
            if state['state'] == 'fault':
                return False
            return True
        outcome = poll_for(args.seconds, clock, rospy.is_shutdown, poll_once, rospy.rostime.wallsleep)
    except KeyboardInterrupt:
        outcome = 'interrupted'
    finally:
        observation.recording = False
        closed = workers.close(2.)
        complete = timeline.close(5.)
        rospy.signal_shutdown('read-only independent sensing probe finished')
    print(json.dumps({'mode': 'read_only_independent_sensing', 'motion_requested': False,
                      'counts': dict(counts), 'worker_closed': closed, 'timeline_complete': complete,
                      'unprocessed_frames': workers.unprocessed_at_close,
                      'outcome': outcome, 'dropped': timeline.dropped, 'timeline': args.timeline}, sort_keys=True))
    return 0 if closed and complete and outcome != 'fault' and not any(key.startswith('fault:') for key in counts) else 1


if __name__ == '__main__':
    sys.exit(main())
