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
    parser.add_argument('--depth-projection-process', action='store_true',
                        help='project depth in a child process with a private bounded socket')
    parser.add_argument('--split-sensor-checks', action='store_true',
                        help='prepare scan and depth proofs in separate workers')
    parser.add_argument('--scan-validation-process', action='store_true',
                        help='validate rolling laser coverage in a separate process')
    parser.add_argument('--separate-model-camera', action='store_true',
                        help='use color for model captures and pause it during local control')
    parser.add_argument('--warm-split-start', action='store_true',
                        help='prepare a new split sensor proof at rest before the first nonzero intent')
    parser.add_argument('--direct-relay-writes', action='store_true',
                        help='submit checked commands directly to the independent relay process')
    parser.add_argument('--speed-mps', type=float, default=.02,
                        help='set the requested forward speed up to 0.02 m/s')
    parser.add_argument('--depth-sample-stride', type=int, choices=(2, 4), default=4,
                        help='sample every second or fourth depth pixel in each dimension')
    parser.add_argument('--lidar-margin-filter', action='store_true',
                        help='mark isolated extra-margin lidar returns unknown in a forward trial')
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
    if args.depth_projection_process and (not args.preproject_depth or not args.split_sensor_checks
                                          or not args.assume_rectified or args.ground_z is None):
        parser.error('--depth-projection-process requires split checks, preprojection and explicit depth assumptions')
    if args.split_sensor_checks and not args.prepared_forward:
        parser.error('--split-sensor-checks requires --prepared-forward')
    if args.scan_validation_process and not args.split_sensor_checks:
        parser.error('--scan-validation-process requires --split-sensor-checks')
    if args.separate_model_camera and not args.split_sensor_checks:
        parser.error('--separate-model-camera requires --split-sensor-checks')
    if args.warm_split_start and not args.split_sensor_checks:
        parser.error('--warm-split-start requires --split-sensor-checks')
    if args.direct_relay_writes and not args.warm_split_start:
        parser.error('--direct-relay-writes requires --warm-split-start')
    if not 0 < args.speed_mps <= .02:
        parser.error('--speed-mps must be between 0 and 0.02 m/s')
    if args.execute:
        if args.lidar_margin_filter and not args.split_sensor_checks:
            parser.error('live lidar margin filtering requires split sensor checks')
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


def make_ros_monitor(clock, driver_pid, master_uri, retry_late_samples=False):
    from live_monitors import Ros1ReadOnlyBackend, RosCallerMonitor
    backend = Ros1ReadOnlyBackend(master_uri, clock, rpc_timeout=0.2)
    # Read the most recent watchdog packet more frequently between graph RPCs.
    return RosCallerMonitor(backend, clock, driver_pid, poll_interval=0.05, max_age=0.3,
                            acquire_timeout=1.0, bind_timeout=0.35, retry_late_samples=retry_late_samples)


class ObservationRecorder(object):  # noqa: UP004 - ROS Melodic Python 2
    """Subscribe only; never publish a command."""
    def __init__(self, clock, capacity=2000):
        import rospy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        self.clock, self.lock = clock, threading.Lock()
        self.odometry, self.commands = collections.deque(maxlen=capacity), collections.deque(maxlen=capacity)
        self.subscribers = [rospy.Subscriber('/odom', Odometry, self._odom, queue_size=10),
                            rospy.Subscriber('/cmd_vel', Twist, self._command, queue_size=10)]

    def _odom(self, message):
        try:
            received = self.clock.now()
            pose, twist = message.pose.pose, message.twist.twist
            point, orientation = pose.position, pose.orientation
            x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            item = (received, point.x, point.y, twist.linear.x, yaw, twist.angular.z,
                    message.header.stamp.to_sec(), message.header.seq)
            if any(math.isnan(value) or math.isinf(value) for value in item + (x, y, z, w)):
                return
            with self.lock:
                self.odometry.append(item)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return

    def _command(self, message):
        try:
            caller = getattr(message, '_connection_header', {}).get('callerid')
            item = (self.clock.now(), float(message.linear.x), float(message.angular.z), caller)
            if any(math.isnan(value) or math.isinf(value) for value in item[:3]):
                return
            with self.lock:
                self.commands.append(item)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return

    def summary(self):
        with self.lock:
            return summarize_observations([item[:4] for item in self.odometry],
                                          [item[:3] for item in self.commands])

    def commands_after(self, caller, after):
        selected = []
        with self.lock:
            for at, linear, angular, source in reversed(self.commands):
                if at <= after:
                    break
                if caller is not None and caller == source:
                    selected.append((at, linear, angular))
        return list(reversed(selected))

    def trial_summary(self, caller, started, ended):
        with self.lock:
            return summarize_trial_observations([item[:4] for item in self.odometry], list(self.commands),
                                                caller, started, ended)

    def trial_trajectory(self, caller, started, ended):
        with self.lock:
            return {
                'clock': 'host_monotonic_receipt',
                'odom_columns': ['at', 'x_m', 'y_m', 'forward_mps', 'yaw_rad', 'yaw_radps',
                                 'ros_source_stamp_sec', 'ros_source_seq'],
                'odom': [item for item in self.odometry if started <= item[0] <= ended],
                'command_columns': ['at', 'forward_mps', 'yaw_radps'],
                'commands': [item[:3] for item in self.commands
                             if caller is not None and caller == item[3] and started <= item[0] <= ended],
            }


def main(argv=None):
    args = parse_args(argv)
    if args.scan_validation_process:
        from scan_validation_process import ScanValidationProcess
        with ScanValidationProcess() as scan_validator:
            return run_with_depth(args, scan_validator)
    return run_with_depth(args)


def run_with_depth(args, scan_validator=None):
    if args.depth_projection_process:
        from depth_projection_process import DepthProjectionProcess
        with DepthProjectionProcess() as projector:
            return run(args, projector, scan_validator)
    return run(args, scan_validator=scan_validator)


def run(args, projector=None, scan_validator=None):
    import rospy
    from explore import RosObservation
    from live_monitors import (
        ProcSerialBackend,
        SerialOwnershipMonitor,
    )
    from live_trial_adapter import (
        CvJpegEncoder,
        DecisionsAdapter,
        DirectPreparedLongerForwardTrial,
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
                      'min_depth_m': 0.3, 'max_depth_m': 3.0, 'sample_stride': args.depth_sample_stride}
    observation = RosObservation(depth_projection_config=projection, depth_projector=projector,
                                 scan_validator=scan_validator, separate_model_camera=args.separate_model_camera,
                                 filter_lidar_margin=args.lidar_margin_filter,
                                 use_source_deadlines=args.direct_relay_writes)
    recorder = ObservationRecorder(clock)
    source = ObservationAdapter(observation, CvJpegEncoder(observation.bridge), take_scene_ownership=True,
                                projection_only_after_model=True, recheck_snapshot=False,
                                predecision_warmup=True, serialize_processing=args.longer_forward)
    source.warm_split_start = args.warm_split_start
    # Populate subscriptions and scan fusion before the bounded session starts.
    warm_until = clock.now() + 2.0
    while clock.now() < warm_until and not rospy.is_shutdown():
        observation.snapshot()
        time.sleep(0.05)
    if not args.execute:
        print(json.dumps({'mode': 'observation_only', 'observation': observation.diagnostic_status(),
                          'recorder': recorder.summary(),
                          'projection_process': projector.summary() if projector is not None else None},
                         allow_nan=False))
        return 0
    ros_guard = make_ros_monitor(clock, args.driver_pid,
                                os.environ.get('ROS_MASTER_URI', 'http://localhost:11311'))
    serial_backend = ProcSerialBackend(clock, args.driver_pid, '/dev/ttyTHS1', fuser_timeout=0.4)
    serial_guard = SerialOwnershipMonitor(serial_backend, clock, args.driver_pid, args.driver_exe,
                                          args.driver_sha256, max_age=0.6, acquire_timeout=1.2,
                                          ros_monitor=ros_guard)
    model = DecisionsAdapter(ExistingDecisionHelper(clock, cancelled=rospy.is_shutdown), forward_only=True,
                             forward_limit_m=.04 if args.longer_forward else .006,
                             forward_speed_mps=args.speed_mps)
    relay = make_relay(clock, motion_limit_sec=2.0 if args.longer_forward else 0.3)
    report = {'clear_room': True, 'operator_present': True, 'emergency_stop_available': True,
              'source': 'operator_report', 'independently_verified': False}
    trial_class = (DirectPreparedLongerForwardTrial if args.direct_relay_writes else
                   PreparedLongerForwardTrial if args.prepared_forward else
                   SynchronousLongerForwardTrial if args.single_threaded else
                   LongerForwardTrial if args.longer_forward else LiveTrial)
    trial = trial_class(source, model, relay, ros_guard, clock, enable_motion=True,
                        duration_sec=2.0 if args.longer_forward else 0.6,
                        speed_mps=args.speed_mps,
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
        result['split_check_timeline'] = observation._split_validation.timeline()
    if args.preproject_depth:
        result['depth_worker_closed'] = depth_worker_closed
        result['depth_preprojection_counts'] = dict(observation.depth_preprojection_counts)
    if projector is not None:
        result['depth_projection_process'] = projector.summary()
    if scan_validator is not None:
        result['scan_validation_process'] = scan_validator.summary()
    trial_ended = clock.now()
    # Continue read-only observation briefly after the relay has been closed.
    time.sleep(0.7)
    caller = result.get('ros_guard_evidence', {}).get('relay_node')
    observed = recorder.trial_summary(caller, trial_started, trial_ended)
    trajectory = recorder.trial_trajectory(caller, trial_started, clock.now())
    exit_code = trial_exit_code(result, observed)
    print(json.dumps({'mode': 'supervised_single_trial', 'result': result,
                      'observed': observed, 'trajectory': trajectory, 'exit_code': exit_code,
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
