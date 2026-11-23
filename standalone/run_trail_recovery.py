#!/usr/bin/env python2
"""Retrace verified odometry in an attended static room."""
from __future__ import division, print_function

import argparse
import json
import math
import os
import re
import sys
import time

from navigation import Command, Pose, _relative
from trail_recovery import (MAX_LATERAL_M, TrailFollower, continued_offset, front_obstacle, front_registration,
                            measured_trail, predicted_front, prior_retrace_offset, recorded_scan_matches,
                            scan_matches, straight_trail)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True)
    parser.add_argument('--front-reference', required=True)
    parser.add_argument('--distance-m', type=float, default=.32)
    parser.add_argument('--resume', help='recording of a partial retrace on this strip')
    parser.add_argument('--recorded-curve', action='store_true', help='follow recorded turns and reverse segments')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--static-room-clear', action='store_true')
    parser.add_argument('--driver-pid', type=int)
    parser.add_argument('--driver-exe')
    parser.add_argument('--driver-sha256')
    args = parser.parse_args(argv)
    if args.recorded_curve and args.resume:
        parser.error('partial straight-trail registration is only available for a straight retreat')
    if args.execute and (not args.static_room_clear or not args.driver_pid or args.driver_pid <= 1
                         or not args.driver_exe or not os.path.isabs(args.driver_exe)
                         or not re.match(r'\A[a-f0-9]{64}\Z', args.driver_sha256 or '')):
        parser.error('execution requires a clear supervised static room and the running driver identity')
    with open(args.trace) as stream:
        trace = json.load(stream)
    with open(args.front_reference) as stream:
        reference = json.load(stream)
    follower = None
    if args.recorded_curve:
        points, report = measured_trail(trace, reference, args.distance_m)
        follower = TrailFollower(points)
    else:
        report = straight_trail(trace, reference, args.distance_m)
    offset = Pose(0., 0., 0.)
    if args.resume:
        with open(args.resume) as stream:
            offset = prior_retrace_offset(json.load(stream), args.distance_m)
        reference = predicted_front(reference, offset)
    if not args.execute:
        print(json.dumps(report, allow_nan=False))
        return 0
    return execute(args, reference, report, offset, follower)


def execute(args, reference, report, origin, follower):
    import rospy
    from explore import RosObservation, base_state, sample_clocks, source_time
    from live_monitors import ProcFdSerialBackend, SerialOwnershipMonitor
    from live_trial_adapter import RelayProcessTransport
    from navigation_actuator import ActuationError, NavigationActuator
    from sensor_msgs.msg import LaserScan
    from supervised_trial import ObservationRecorder, RealClock, make_ros_monitor

    rospy.init_node('jev_trail_recovery', anonymous=True)
    clock = RealClock()
    if rospy.get_param('/use_sim_time', False):
        raise ActuationError('wall_time_required')
    observation = RosObservation(sensor_names=('odom', 'status'))
    scan = [None]

    def receive(message):
        pair = sample_clocks(clock.now, lambda: rospy.Time.now().to_sec())
        stamp = source_time(pair[0], pair[1], message.header.stamp.to_sec(), .30) if pair else None
        if stamp is not None and message.header.frame_id.lstrip('/') == 'laser_link':
            scan[0] = (message, stamp)

    subscription = rospy.Subscriber('/scan', LaserScan, receive, queue_size=1)

    def snapshot(now):
        record = scan[0]
        with observation.lock:
            odom, status = observation.latest.get('odom'), observation.latest.get('status')
            epoch = observation.epoch
            guarded = observation.clock_guard.faulted or observation.odom_guard.faulted
        if (guarded or not record or not odom or not status or odom['epoch'] != epoch
                or not (0 <= now - record[1] < .30 and 0 <= now - odom['stamp'] < .15
                        and 0 <= now - status['stamp'] < 1.)):
            return None
        base = base_state(odom['message'], status['message'], status['stamp'])
        if (base is None or base['control_mode'] != 1 or base['motion_mode'] != 0
                or base['vehicle_state'] != 0 or base['error_code'] != 0 or base['battery_voltage'] < 10.):
            return None
        return Pose(*odom['pose']), epoch, record[0]

    ros = make_ros_monitor(clock, args.driver_pid, os.environ.get('ROS_MASTER_URI', 'http://localhost:11311'))
    backend = ProcFdSerialBackend(clock, args.driver_pid, '/dev/ttyTHS1')
    serial = SerialOwnershipMonitor(backend, clock, args.driver_pid, args.driver_exe, args.driver_sha256,
                                    max_age=.6, acquire_timeout=1.2, ros_monitor=ros)
    actuator = NavigationActuator(RelayProcessTransport(clock), ros, serial, clock, reverse_navigation=True,
                                  trail_pivot=follower is not None)
    recorder = ObservationRecorder(clock)
    start = clock.now()
    offset = origin
    distance = -offset.x
    reason = 'observation_unavailable'
    zero = Command(0., 0., 'stop', 'retrace_wait')
    matches_reference = recorded_scan_matches if follower is not None else scan_matches

    def check_anchor(seconds, armed=False):
        until = clock.now() + seconds
        seen = None
        matches = []
        while not rospy.is_shutdown() and clock.now() < until:
            sample = snapshot(clock.now())
            if sample is not None and sample[2].header.stamp != seen:
                seen = sample[2].header.stamp
                match = (front_registration(sample[2], reference) if args.resume else
                         0. if matches_reference(sample[2], reference) else None)
                if match is None:
                    matches = []
                else:
                    matches = matches[-1:] + [match]
                    if len(matches) == 2 and abs(matches[0] - matches[1]) <= .012:
                        return sample, sum(matches) / 2.
            if armed:
                actuator.publish(zero)
            time.sleep(.04)
        return None

    try:
        if check_anchor(5.) is None:
            raise ActuationError('recorded_front_reference_unconfirmed')
        actuator.open()
        checked = check_anchor(2., armed=True)
        if checked is None:
            raise ActuationError('recorded_front_reference_changed')
        initial, shift = checked
        origin = origin._replace(x=origin.x - shift)
        report['front_registration_delta_m'] = shift
        pose, epoch = initial[:2]
        start = last_ready = clock.now()
        reason = 'session_ended'
        duration = 54.8 if follower is not None else 29.8
        while not rospy.is_shutdown() and clock.now() - start < duration:
            now = clock.now()
            sample = snapshot(now)
            if sample is None and now - last_ready < .3:
                actuator.publish(zero)
                time.sleep(.04)
                continue
            if sample is None or sample[1] != epoch:
                reason = 'observation_unavailable'
                break
            last_ready = now
            offset = continued_offset(origin, _relative(pose, sample[0]))
            if follower is not None:
                distance = math.hypot(offset.x, offset.y)
                requested = follower.step(offset)
                if not (requested.linear or requested.angular):
                    reason = requested.reason
                    break
                if front_obstacle(sample[2], requested.angular):
                    reason = 'retrace_live_obstacle_or_missing_returns'
                    break
                command = actuator.publish(requested)
                if not (command.linear or command.angular):
                    reason = command.reason
                    break
                time.sleep(.04)
                continue
            distance = -offset.x
            if distance < -.005 or abs(offset.y) > MAX_LATERAL_M or abs(offset.yaw) > .04:
                reason = 'retrace_path_departure'
                break
            if distance >= args.distance_m:
                reason = 'retrace_distance_reached'
                break
            correction = max(-.02, min(.02, 1.6 * (4. * offset.y - offset.yaw)))
            command = actuator.publish(Command(-.018, correction, 'reverse', 'retrace'))
            if not command.linear:
                reason = command.reason
                break
            time.sleep(.04)
    except (ActuationError, IOError, OSError) as error:
        reason = str(error) if isinstance(error, ActuationError) else type(error).__name__
    finally:
        actuator.close()
        subscription.unregister()
    data = recorder.trial_trajectory(actuator.caller, start, clock.now())
    report.update(reason=reason, displacement_m=distance, final_offset=offset, trajectory=data,
                  relay_fault=actuator.stop_line, relay_exited=actuator.transport.exited)
    if follower is not None:
        report['last_completed_keyframe'] = follower.index
    print(json.dumps(report, allow_nan=False))
    return 0 if reason == 'retrace_distance_reached' and actuator.transport.exited else 2


if __name__ == '__main__':
    sys.exit(main())
