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

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - Python 2.7 ROS
except NameError:
    INTEGER_TYPES = (int,)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='enable a single bounded motion trial')
    parser.add_argument('--longer-forward', action='store_true',
                        help='allow up to two seconds of forward motion in this trial')
    parser.add_argument('--single-threaded', action='store_true',
                        help='experiment with synchronous sensing in the longer trial')
    parser.add_argument('--prepared-forward', action='store_true',
                        help='run final sensor processing in the longer trial worker')
    parser.add_argument('--preproject-depth', action='store_true',
                        help='use an independent depth projection worker in prepared mode')
    parser.add_argument('--split-sensor-checks', action='store_true',
                        help='prepare scan and depth proofs in separate workers')
    parser.add_argument('--driver-pid', type=int)
    parser.add_argument('--driver-exe')
    parser.add_argument('--driver-sha256')
    parser.add_argument('--operator-site-reported', action='store_true')
    parser.add_argument('--ground-z', type=float)
    parser.add_argument('--assume-rectified', action='store_true')
    args = parser.parse_args(argv)
    if args.single_threaded and not args.longer_forward:
        parser.error('--single-threaded requires --longer-forward')
    if args.prepared_forward and (not args.longer_forward or args.single_threaded):
        parser.error('--prepared-forward requires --longer-forward without --single-threaded')
    if args.preproject_depth and not args.prepared_forward:
        parser.error('--preproject-depth requires --prepared-forward')
    if args.split_sensor_checks and not args.prepared_forward:
        parser.error('--split-sensor-checks requires --prepared-forward')
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


def trial_period(args):
    # Refresh sooner in the synchronous experiment, without extending any TTL.
    return 0.025 if args.single_threaded or args.prepared_forward else 0.05


def trial_exit_code(result, observed):
    """Report diagnostic success only for a completed trial with odom movement.

    Five millimetres is a diagnostic threshold, not independently verified
    physical displacement or a navigation acceptance threshold.
    """
    try:
        displacement = observed['odom_displacement_m']
        commands = observed['nonzero_cmd_vel_messages']
        if (result.get('reason') in ('time_limit', 'motion_limit', 'distance_limit')
                and result.get('motion_requested') is True
                and result.get('stop_requested') is True
                and ('sampler_closed' not in result or result['sampler_closed'] is True)
                and ('depth_worker_closed' not in result or result['depth_worker_closed'] is True)
                and ('split_workers_closed' not in result or result['split_workers_closed'] is True)
                and result.get('source_restore_failed', False) is False
                and observed.get('measurement_scope') == 'relay_callback_window'
                and observed.get('relay_caller_observed') is True
                and not isinstance(displacement, bool) and not math.isnan(displacement)
                and not math.isinf(displacement) and displacement >= .005
                and not isinstance(commands, bool) and isinstance(commands, INTEGER_TYPES) and commands > 0):
            return 0
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        pass
    return 2


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


def summarize_trial_observations(odometry, commands, expected_caller, started, ended):
    """Use relay-attributed command and odometry callback timestamps.

    ROS caller IDs are not authenticated; the result is not independent
    physical motion or stopping verification. ROS topic callbacks can be
    delayed or reordered, so this is not the relay's actual motion interval.
    """
    summary = summarize_observations([], [])
    summary.update({'measurement_scope': 'relay_callback_window', 'relay_caller_observed': False})
    try:
        if (not isinstance(expected_caller, str) or not expected_caller.startswith('/')
                or math.isnan(started) or math.isinf(started) or math.isnan(ended) or math.isinf(ended)
                or not 0 <= started <= ended):
            return summary
        relevant = sorted(item for item in commands if isinstance(item, (tuple, list)) and len(item) == 4
                          and item[3] == expected_caller and started <= item[0] <= ended)
        selected = [item for item in relevant if item[1] != 0 or item[2] != 0]
        if not selected:
            return summary
        first = selected[0][0]
        zeros = [item[0] for item in relevant if item[0] > first and item[1] == 0 and item[2] == 0]
        stop = min(zeros) if zeros else selected[-1][0]
        if any(item[0] > stop for item in selected):
            return summary
        # Exclude pre-command movement and post-session motion. This may
        # undercount the first short movement but avoids attributing unrelated
        # movement to the trial.
        scoped_odom = sorted(item for item in odometry if isinstance(item, (tuple, list)) and len(item) == 4
                             and first <= item[0] <= stop)
        summary = summarize_observations(scoped_odom, [item[:3] for item in selected])
        summary.update({'measurement_scope': 'relay_callback_window', 'relay_caller_observed': True})
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        pass
    return summary


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
    # Read the most recent watchdog packet more frequently between graph RPCs.
    return RosCallerMonitor(backend, clock, driver_pid, poll_interval=0.05, max_age=0.3,
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
            caller = getattr(message, '_connection_header', {}).get('callerid')
            item = (self.clock.now(), float(message.linear.x), float(message.angular.z), caller)
            with self.lock:
                self.commands.append(item)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return

    def summary(self):
        with self.lock:
            return summarize_observations(list(self.odometry), [item[:3] for item in self.commands])

    def trial_summary(self, caller, started, ended):
        with self.lock:
            return summarize_trial_observations(list(self.odometry), list(self.commands), caller, started, ended)


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
        PreparedLongerForwardTrial,
        SynchronousLongerForwardTrial,
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
    trial_class = (PreparedLongerForwardTrial if args.prepared_forward else
                   SynchronousLongerForwardTrial if args.single_threaded else
                   LongerForwardTrial if args.longer_forward else LiveTrial)
    trial = trial_class(source, model, relay, ros_guard, clock, enable_motion=True,
                        duration_sec=2.0 if args.longer_forward else 0.6,
                        period_sec=trial_period(args),
                        serial_guard=serial_guard, depth_gate=ProjectedDepthGate(),
                        operator_site_report=report)
    trial_started = clock.now()
    if args.preproject_depth:
        observation.start_depth_preprojection()
    if args.split_sensor_checks:
        observation.start_split_validation()
    try:
        result = trial.run()
    finally:
        split_workers_closed = observation.close_split_validation() if args.split_sensor_checks else None
        depth_worker_closed = observation.close_depth_preprojection() if args.preproject_depth else None
    if args.split_sensor_checks:
        result['split_workers_closed'] = split_workers_closed
        result['split_check_counts'] = observation._split_validation.counts()
    if args.preproject_depth:
        result['depth_worker_closed'] = depth_worker_closed
        result['depth_preprojection_counts'] = dict(observation.depth_preprojection_counts)
    trial_ended = clock.now()
    # Continue read-only observation briefly after the relay has been closed.
    time.sleep(0.7)
    caller = result.get('ros_guard_evidence', {}).get('relay_node')
    observed = recorder.trial_summary(caller, trial_started, trial_ended)
    exit_code = trial_exit_code(result, observed)
    print(json.dumps({'mode': 'supervised_single_trial', 'result': result,
                      'observed': observed, 'exit_code': exit_code,
                      'observation': observation.diagnostic_status()},
                     allow_nan=False))
    return exit_code


if __name__ == '__main__':
    import sys
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({'mode': 'trial_error', 'error_type': type(exc).__name__,
                          'physical_stop_verified': False}))
        sys.exit(1)
