#!/usr/bin/env python2
"""One supervised, forward-only candidate trial; default is observation-only.

The Decisions API selects the direction. A separate relay enforces a short
motion deadline, while local sensor and process monitors can stop the trial.
Unknown sensor regions, site conditions, chassis braking, and an independent
hardware stop are not established by this program.
"""
from __future__ import division, print_function

import argparse
import collections
import json
import math
import os
import threading
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='enable a single bounded motion trial')
    parser.add_argument('--longer-forward', action='store_true',
                        help='allow up to two seconds of forward motion in this trial')
    parser.add_argument('--driver-pid', type=int)
    parser.add_argument('--driver-exe')
    parser.add_argument('--driver-sha256')
    parser.add_argument('--operator-site-reported', action='store_true')
    parser.add_argument('--ground-z', type=float)
    parser.add_argument('--assume-rectified', action='store_true')
    args = parser.parse_args(argv)
    if args.execute:
        digest = args.driver_sha256
        if (args.driver_pid is None or args.driver_pid <= 1
                or not args.driver_exe or not args.driver_exe.startswith('/')
                or not digest or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or not args.operator_site_reported or not args.assume_rectified
                or args.ground_z is None or math.isnan(args.ground_z) or math.isinf(args.ground_z)
                or not -0.5 <= args.ground_z <= 0.0):
            parser.error('execute requires driver identity, site report and explicit depth assumptions')
    return args


def summarize_observations(odometry, commands):
    """Keep reported odometry separate from observed commands and physical proof."""
    result = {'odom_samples': 0, 'odom_displacement_m': None, 'max_abs_odom_speed_mps': None,
              'last_odom_speed_mps': None, 'nonzero_cmd_vel_messages': 0,
              'physical_stop_verified': False}
    def finite(value):
        try:
            return not isinstance(value, bool) and not math.isnan(value) and not math.isinf(value)
        except (TypeError, ValueError, OverflowError):
            return False
    valid = [item for item in odometry if isinstance(item, (tuple, list)) and len(item) == 4
             and all(finite(value) for value in item)]
    if valid:
        result.update({'odom_samples': len(valid), 'last_odom_speed_mps': valid[-1][3],
                       'max_abs_odom_speed_mps': max(abs(item[3]) for item in valid)})
        if len(valid) >= 2:
            result['odom_displacement_m'] = math.hypot(valid[-1][1] - valid[0][1],
                                                        valid[-1][2] - valid[0][2])
    result['nonzero_cmd_vel_messages'] = sum(
        1 for item in commands if isinstance(item, (tuple, list)) and len(item) == 3
        and all(finite(value) for value in item) and (item[1] != 0 or item[2] != 0))
    return result


class RealClock(object):  # noqa: UP004 - ROS Melodic Python 2
    def now(self):
        from relay import _monotonic
        return _monotonic()

    def sleep(self, delay):
        time.sleep(delay)


def make_relay(clock, motion_limit_sec=0.3):
    from live_trial_adapter import RelayProcessTransport
    from threaded_relay_adapter import ThreadedRelayAdapter
    return ThreadedRelayAdapter(RelayProcessTransport(clock), clock, motion_limit_sec=motion_limit_sec)


def make_ros_monitor(clock, driver_pid, master_uri):
    from live_monitors import Ros1ReadOnlyBackend, RosCallerMonitor
    backend = Ros1ReadOnlyBackend(master_uri, clock, rpc_timeout=0.2)
    return RosCallerMonitor(backend, clock, driver_pid, poll_interval=0.1, max_age=0.3,
                            acquire_timeout=1.0, bind_timeout=0.35)


class ObservationRecorder(object):  # noqa: UP004 - ROS Melodic Python 2
    """Subscribe only; never publish a command."""
    def __init__(self, clock):
        import rospy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        self.clock, self.lock = clock, threading.Lock()
        self.odometry, self.commands = collections.deque(maxlen=2000), collections.deque(maxlen=2000)
        self.subscribers = [rospy.Subscriber('/odom', Odometry, self._odom, queue_size=10),
                            rospy.Subscriber('/cmd_vel', Twist, self._command, queue_size=10)]

    def _odom(self, message):
        try:
            point, twist = message.pose.pose.position, message.twist.twist
            item = (self.clock.now(), float(point.x), float(point.y), float(twist.linear.x))
            with self.lock:
                self.odometry.append(item)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return

    def _command(self, message):
        try:
            item = (self.clock.now(), float(message.linear.x), float(message.angular.z))
            with self.lock:
                self.commands.append(item)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return

    def summary(self):
        with self.lock:
            return summarize_observations(list(self.odometry), list(self.commands))


def main(argv=None):
    args = parse_args(argv)
    import rospy
    from explore import RosObservation
    from live_monitors import (
        ProcSerialBackend,
        SerialOwnershipMonitor,
    )
    from live_trial_adapter import (
        CvJpegEncoder,
        DecisionsAdapter,
        ExistingDecisionHelper,
        LiveTrial,
        LongerForwardTrial,
        ObservationAdapter,
    )
    from projected_depth_gate import ProjectedDepthGate

    rospy.init_node('limo_supervised_trial', anonymous=True, disable_signals=True)
    if rospy.get_param('/use_sim_time', False):
        raise ValueError('wall clock is required')
    os.environ.setdefault('ROSCLAW_LIMO_DECISIONS_KEY_FILE', os.path.expanduser('~/rosclaw/jev/.env'))
    clock = RealClock()
    projection = None
    if args.assume_rectified and args.ground_z is not None:
        projection = {'forward_min_m': 0.18, 'forward_max_m': 1.0, 'half_width_m': 0.15,
                      'ground_z_m': args.ground_z, 'ground_tolerance_m': 0.025,
                      'obstacle_min_height_m': 0.03, 'obstacle_max_height_m': 0.4,
                      'min_depth_m': 0.3, 'max_depth_m': 3.0, 'sample_stride': 4}
    observation = RosObservation(depth_projection_config=projection)
    recorder = ObservationRecorder(clock)
    source = ObservationAdapter(observation, CvJpegEncoder(observation.bridge), take_scene_ownership=True,
                                projection_only_after_model=True, recheck_snapshot=False,
                                predecision_warmup=True, serialize_processing=args.longer_forward)
    # Populate subscriptions and scan fusion before the bounded session starts.
    warm_until = clock.now() + 2.0
    while clock.now() < warm_until and not rospy.is_shutdown():
        observation.snapshot()
        time.sleep(0.05)
    if not args.execute:
        print(json.dumps({'mode': 'observation_only', 'observation': observation.diagnostic_status(),
                          'recorder': recorder.summary()}, allow_nan=False))
        return 0
    ros_guard = make_ros_monitor(clock, args.driver_pid,
                                os.environ.get('ROS_MASTER_URI', 'http://localhost:11311'))
    serial_backend = ProcSerialBackend(clock, args.driver_pid, '/dev/ttyTHS1', fuser_timeout=0.4)
    serial_guard = SerialOwnershipMonitor(serial_backend, clock, args.driver_pid, args.driver_exe,
                                          args.driver_sha256, max_age=0.6, acquire_timeout=1.2,
                                          ros_monitor=ros_guard)
    model = DecisionsAdapter(ExistingDecisionHelper(clock, cancelled=rospy.is_shutdown))
    relay = make_relay(clock, motion_limit_sec=2.0 if args.longer_forward else 0.3)
    report = {'clear_room': True, 'operator_present': True, 'emergency_stop_available': True,
              'source': 'operator_report', 'independently_verified': False}
    trial_class = LongerForwardTrial if args.longer_forward else LiveTrial
    trial = trial_class(source, model, relay, ros_guard, clock, enable_motion=True,
                        duration_sec=2.0 if args.longer_forward else 0.6,
                        serial_guard=serial_guard, depth_gate=ProjectedDepthGate(),
                        operator_site_report=report)
    result = trial.run()
    # Continue read-only observation briefly after the relay has been closed.
    time.sleep(0.7)
    print(json.dumps({'mode': 'supervised_single_trial', 'result': result,
                      'observed': recorder.summary(), 'observation': observation.diagnostic_status()},
                     allow_nan=False))
    return 0


if __name__ == '__main__':
    import sys
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({'mode': 'trial_error', 'error_type': type(exc).__name__,
                          'physical_stop_verified': False}))
        sys.exit(1)
