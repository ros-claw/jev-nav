#!/usr/bin/env python2
"""Run odometry-frame navigation using live ROS perception."""
from __future__ import division, print_function

import argparse
import base64
import collections
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid

from navigation import Grid, LENGTH, Navigation, Pose, View, ready
from navigation_actuator import ActuationError, NavigationActuator, NavigationHeartbeat
from navigation_runtime import NavigationRuntime
from native_sensing_probe import _stop_group, _yaw
from relay import _monotonic


def brighten_low_light(image):
    """Raise measured intensity only. Keep the original pixels and geometry."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark, bright = np.percentile(gray, [10, 99])
    if bright > 60 or bright - dark < 5:
        return image
    gain = min(6., 180. / max(20., bright))
    table = np.minimum(255, np.arange(256, dtype=np.float32) * gain).astype(np.uint8)
    return cv2.LUT(image, table)


class RosNavigationInput(object):
    def __init__(self, rospy, grid_topic, decisions, operator_clear_front=False, static_room=False):
        from explore import RosObservation
        from limo_fresh_layer.msg import FreshGrid
        from sensor_msgs.msg import Image

        self.rospy = rospy
        self.operator_clear_front = operator_clear_front
        self.static_room = static_room
        self.observation = RosObservation(sensor_names=('odom', 'status'))
        self.latest_grid = None
        self.last_grid_times = None
        self.latest_color = None
        self.grid_updates = collections.Counter()
        self.decision_errors = collections.Counter()
        self.subscriptions = [rospy.Subscriber(grid_topic, FreshGrid, self._grid, queue_size=1)]
        if decisions:
            self.subscriptions.append(rospy.Subscriber('/camera/color/image_raw', Image, self._color,
                                                        queue_size=1, buff_size=1024 * 1024))

    def _color(self, message):
        self.latest_color = message

    def _grid(self, message):
        from explore import sample_clocks, source_time

        try:
            pair = sample_clocks(_monotonic, lambda: self.rospy.Time.now().to_sec())
            if pair is None:
                return
            grid = message.grid
            if (message.header.frame_id.lstrip('/') != 'odom' or grid.header.frame_id != message.header.frame_id
                    or grid.header.stamp != message.header.stamp):
                self.grid_updates['grid_header_mismatch'] += 1
                self.latest_grid = None
                return
            if not message.scan_fresh or not message.depth_fresh:
                self.grid_updates[('scan_fresh' if message.scan_fresh else 'scan_unavailable') + '.' +
                                  ('depth_fresh' if message.depth_fresh else 'depth_unavailable')] += 1
                self.latest_grid = None
                return
            raw_times = (message.header.stamp.to_sec(), message.scan_stamp.to_sec(), message.depth_stamp.to_sec())
            if max(raw_times[1:]) > raw_times[0]:
                self.latest_grid = None
                return
            times = tuple(source_time(pair[0], pair[1], stamp, age)
                          for stamp, age in zip(raw_times, (.25, .30, .20)))
            info = grid.info
            x, y = info.origin.position.x, info.origin.position.y
            if (None in times or not 0 < info.width <= 256 or not 0 < info.height <= 256
                    or not .01 <= info.resolution <= .10 or abs(_yaw(info.origin.orientation)) > 1e-8
                    or any(math.isnan(value) or math.isinf(value) for value in (x, y))
                    or len(grid.data) != info.width * info.height):
                self.grid_updates['invalid_grid'] += 1
                self.latest_grid = None
                return
            previous = self.last_grid_times
            if previous is not None and any(stamp < old for stamp, old in zip(raw_times, previous)):
                self.grid_updates['grid_regression'] += 1
                self.latest_grid = None
                return
            self.last_grid_times = raw_times
            self.latest_grid = times, Grid(x, y, info.resolution, info.width, info.height, tuple(grid.data))
            self.grid_updates['ready'] += 1
        except (AttributeError, ArithmeticError, ValueError, TypeError):
            self.grid_updates['invalid_grid'] += 1
            self.latest_grid = None

    def view(self):
        from explore import base_state

        observation = self.observation
        with observation.lock:
            odom, status = observation.latest.get('odom'), observation.latest.get('status')
            epoch = observation.epoch
            guarded = observation.clock_guard.faulted or observation.odom_guard.faulted
        snapshot = self.latest_grid
        if snapshot is None or odom is None or status is None:
            return None
        state = base_state(odom['message'], status['message'], status['stamp'])
        ready = (not guarded and odom['epoch'] == epoch and state is not None
                 and state['control_mode'] == 1 and state['motion_mode'] == 0
                 and state['error_code'] == 0 and state['vehicle_state'] == 0
                 and state['battery_voltage'] >= 10.)
        times, grid = snapshot
        return View(times[0], times[1], times[2], odom['stamp'], status['stamp'], epoch,
                    Pose(*odom['pose']), grid, ready)

    def choose(self, request, cancelled):
        from explore import WORKER, base_state, run_decision_process, sample_clocks, source_time
        from live_trial_adapter import CvJpegEncoder

        image = self.latest_color
        if image is None or not request.choices or cancelled():
            return None
        clocks = sample_clocks(_monotonic, lambda: self.rospy.Time.now().to_sec())
        if clocks is None or image.header.frame_id.lstrip('/') != 'camera_color_optical_frame':
            return None
        stamp = source_time(clocks[0], clocks[1], image.header.stamp.to_sec(), .5)
        if stamp is None or abs(stamp - request.at) > .25:
            return None
        observation = self.observation
        with observation.lock:
            if observation.epoch != request.epoch:
                return None
            odom, status = observation.latest.get('odom'), observation.latest.get('status')
        if odom is None or status is None:
            return None
        state = base_state(odom['message'], status['message'], status['stamp'])
        if state is None:
            return None
        jpeg = CvJpegEncoder(observation.bridge, transform=brighten_low_light)(image)
        evidence = {'image_b64': base64.b64encode(jpeg).decode('ascii'), 'captured_at': stamp,
                    'laser': {'locally_checked_segment_length_m': LENGTH,
                              'planning_successor': request.predecessor is not None},
                    'base': state, 'navigation_choices': list(request.choices),
                    'operator_clear_front': self.operator_clear_front}
        if self.static_room:
            evidence['exploration_goal'] = 'circulate_room'
        result = run_decision_process(['/usr/bin/python3', os.path.join(WORKER, 'limo_mapless_decision.py')],
                                      evidence, cancelled, _monotonic, diagnostics=True)
        if result and result.get('ok') is True and result.get('captured_at') == stamp:
            return result.get('choice')
        error = result.get('error_class') if result else None
        self.decision_errors[error or 'DecisionsUnavailable'] += 1
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeline', required=True)
    parser.add_argument('--seconds', type=float, help='end a recorded validation run after this many seconds')
    parser.add_argument('--decisions', action='store_true', help='enable Decisions API calls for checked candidates')
    output = parser.add_mutually_exclusive_group()
    output.add_argument('--drive-forward', action='store_true', help='use the supervised forward hardware output')
    output.add_argument('--drive-arcs', action='store_true', help='execute currently observed forward and arc paths')
    parser.add_argument('--static-room', action='store_true', help='use nearby rear history in a supervised static room')
    parser.add_argument('--operator-clear-front', action='store_true', help='record the current supervised site report')
    parser.add_argument('--driver-pid', type=int)
    parser.add_argument('--driver-exe')
    parser.add_argument('--driver-sha256')
    parser.add_argument('--speed-mps', type=float)
    parser.add_argument('--max-travel-m', type=float, default=.40)
    args = parser.parse_args(argv)
    args.drive = args.drive_forward or args.drive_arcs
    if args.speed_mps is None:
        args.speed_mps = .014 if args.drive else .02
    if args.seconds is not None and not 0. < args.seconds < float('inf'):
        parser.error('seconds must be positive and finite')
    if not 0 < args.speed_mps <= .02 or not .05 <= args.max_travel_m <= (12. if args.static_room else .50):
        parser.error('invalid supervised motion bounds')
    if args.static_room and args.drive_forward:
        parser.error('static room exploration uses the arc or private preview output')
    if args.drive and (not args.decisions or args.seconds is None or args.driver_pid is None or args.driver_pid <= 1
                       or not args.driver_exe or not os.path.isabs(args.driver_exe)
                       or not re.match(r'\A[a-f0-9]{64}\Z', args.driver_sha256 or '')):
        parser.error('hardware operation requires Decisions, a finite run and driver identity')
    if args.operator_clear_front != args.drive_forward:
        parser.error('the operator clear-front report is required for --drive-forward only')
    return args


def summarize_navigation_trajectory(trajectory):
    """Measure a navigation run that can stop temporarily and resume."""
    from supervised_trial import summarize_observations

    commands = sorted(trajectory['commands'], key=lambda row: row[0])
    active = [row for row in commands if row[1] or row[2]]
    summary = summarize_observations([], commands)
    summary.update({'measurement_scope': 'relay_callback_window', 'relay_caller_observed': bool(commands),
                    'longest_nonzero_sec': 0., 'zero_after_last_motion': None,
                    'nonzero_angular_cmd_vel_messages': sum(row[2] != 0. for row in commands),
                    'odom_heading_span_rad': None})
    if not active:
        return summary
    first = active[0][0]
    stop = next((row[0] for row in commands if row[0] > active[-1][0] and not (row[1] or row[2])), None)
    last = active[-1][0] if stop is None else stop
    odometry = sorted((row for row in trajectory['odom'] if first <= row[0] <= last), key=lambda row: row[0])
    summary.update(summarize_observations([row[:4] for row in odometry], commands))
    headings = [row[4] for row in odometry if len(row) >= 5]
    if headings:
        current = minimum = maximum = 0.
        for before, after in zip(headings, headings[1:]):
            change = after - before
            current += math.atan2(math.sin(change), math.cos(change))
            minimum, maximum = min(minimum, current), max(maximum, current)
        summary['odom_heading_span_rad'] = maximum - minimum
    summary['longest_nonzero_sec'] = max([end - start for start, end in nonzero_intervals(commands)] or [0.])
    summary['zero_after_last_motion'] = stop
    return summary


def nonzero_intervals(commands):
    start = previous = None
    for at, linear, angular in commands:
        # A gap at least as long as the driver watchdog cannot establish an
        # uninterrupted interval, even if the recorder saw no zero message.
        if start is not None and at - previous >= .30:
            yield start, previous
            start = None
        if linear or angular:
            start = at if start is None else start
        elif start is not None:
            yield start, at
            start = None
        previous = at
    if start is not None:
        yield start, previous


def summarize_room_loop(trajectory, arc_traces):
    """Check wheel-odometry return and heading change after the first actual turn."""
    report = {'closed': False, 'continuous_commands': False, 'heading_change_rad': 0.,
              'return_distance_m': None, 'duration_sec': None}
    arcs = [arc for arc in arc_traces.values() if abs(arc['heading_change_rad']) >= .05]
    if not arcs:
        return report
    first_arc = min(arcs, key=lambda arc: arc['started_at'])
    samples = sorted(row for row in trajectory['odom'] if row[0] >= first_arc['started_at'])
    if len(samples) < 2:
        return report
    first = samples[0]
    direction = 1. if first_arc['action'] == 'left' else -1.
    heading = 0.
    for previous, current in zip(samples, samples[1:]):
        change = current[4] - previous[4]
        heading += math.atan2(math.sin(change), math.cos(change))
        distance = math.hypot(current[1] - first[1], current[2] - first[2])
        report.update(heading_change_rad=heading, return_distance_m=distance, duration_sec=current[0] - first[0])
        if direction * heading >= 2. * math.pi - .18 and distance <= .25:
            commands = sorted(trajectory['commands'], key=lambda row: row[0])
            report['closed'] = True
            report['continuous_commands'] = any(start <= first[0] + .10 and end >= current[0] - .10
                                                for start, end in nonzero_intervals(commands))
            break
    return report


def main(argv=None):
    args = parse_args(argv)

    import rospy
    from geometry_msgs.msg import Twist
    from limo_fresh_layer.msg import FreshGrid
    from nav_timeline import Timeline
    if 'depth_stamp' not in FreshGrid.__slots__:
        parser.error('build and source the repository version of limo_fresh_layer')
    if args.decisions:
        os.environ.setdefault('ROSCLAW_LIMO_DECISIONS_KEY_FILE', os.path.expanduser('~/rosclaw/jev/.env'))
    namespace = 'limo_native_probe_' + uuid.uuid4().hex[:12]
    directory = os.path.dirname(os.path.abspath(__file__))
    costmap = '/' + namespace + '/rolling_costmap/costmap/snapshot/fresh_grid'
    preview = '/' + namespace + '/navigation_command_preview'
    commands = ([sys.executable, os.path.join(directory, 'native_probe_tf.py'), '--namespace', namespace],
                ['roslaunch', os.path.join(directory, 'native_costmap_probe.launch'),
                 'namespace:=' + namespace, 'use_fresh:=true', 'grid_rate:=20.0', 'decimate_depth:=true',
                 'project_image:=camera_out/image_raw', 'project_info:=camera_out/camera_info',
                 'depth_points_args:=load depth_image_proc/point_cloud_xyz depth_pipeline',
                 'inflate:=false',
                 'depth_layer_type:=limo_fresh_layer/FreshDepthLayer'])
    children = []
    runtime = timeline = source = actuator = recorder = None
    heartbeat = view = None
    counts = collections.Counter()
    ready_ticks = 0
    room = None
    if args.static_room:
        from room_perception import RoomPerception
        room = RoomPerception()
    first_pose = last_pose = None
    unavailable_guards = ()
    authority_diagnostics = {}
    arc_traces = {}
    travelled = 0.
    reason = 'session_ended'
    started = ended = _monotonic()
    try:
        for command in commands:
            children.append(subprocess.Popen(command, preexec_fn=os.setsid, stdout=sys.stderr,
                                             stderr=sys.stderr, close_fds=True))
        rospy.init_node('jev_navigation', anonymous=True)
        if rospy.get_param('/use_sim_time', False):
            raise RuntimeError('ROS wall time required for live acquisition')
        source = RosNavigationInput(rospy, costmap, args.decisions, args.operator_clear_front, args.static_room)
        publisher = rospy.Publisher(preview, Twist, queue_size=1)
        timeline = Timeline(args.timeline, namespace)

        def publish(command):
            packet = Twist()
            packet.linear.x, packet.angular.z = command.linear, command.angular
            if not rospy.is_shutdown():
                publisher.publish(packet)

        choose = source.choose if args.decisions else lambda request, cancelled: None
        if args.drive:
            from live_monitors import ProcFdSerialBackend, SerialOwnershipMonitor
            from live_trial_adapter import RelayProcessTransport
            from supervised_trial import ObservationRecorder, RealClock, make_ros_monitor

            # Warm the live sensor pipeline before claiming the chassis. This
            # also keeps depth-nodelet initialization out of relay startup.
            warm_until = _monotonic() + 5.
            while not ready(source.view(), _monotonic()):
                if rospy.is_shutdown() or _monotonic() >= warm_until:
                    raise ActuationError('navigation_perception_not_ready')
                time.sleep(.05)
            clock = RealClock()
            ros_guard = make_ros_monitor(clock, args.driver_pid, os.environ.get('ROS_MASTER_URI',
                                                                                 'http://localhost:11311'))
            serial_backend = ProcFdSerialBackend(clock, args.driver_pid, '/dev/ttyTHS1')
            serial_guard = SerialOwnershipMonitor(serial_backend, clock, args.driver_pid, args.driver_exe,
                                                  args.driver_sha256, max_age=.6, acquire_timeout=1.2,
                                                  ros_monitor=ros_guard)
            actuator = NavigationActuator(RelayProcessTransport(clock), ros_guard, serial_guard, clock,
                                          arc_navigation=args.drive_arcs)
            recorder = ObservationRecorder(clock, capacity=60000 if room is not None else 12000)
            actuator.open()
            if room is not None:
                heartbeat = NavigationHeartbeat(actuator, clock)
                heartbeat.start()

        def publish_room(command):
            return heartbeat.submit(command, view, _monotonic())

        publish_command = actuator.publish if actuator is not None else publish
        if heartbeat is not None:
            publish_command = publish_room
        navigation = Navigation(forward_only=args.drive_forward, speed=args.speed_mps,
                                successor_lead=.15 if room is not None else None)
        runtime = NavigationRuntime(choose, publish_command, navigation,
                                    static_room=args.static_room)
        started = slot = last_process_check = last_block_report = _monotonic()
        while not rospy.is_shutdown():
            now = _monotonic()
            if args.seconds is not None and now - started >= args.seconds:
                break
            if now - last_process_check >= .5:
                last_process_check = now
                if any(child.poll() is not None for child in children):
                    raise ActuationError('navigation_perception_process_exited')
            view = source.view()
            now = _monotonic()
            if room is not None:
                view = room.view(view, now)
                now = _monotonic()
            if args.drive and view is not None and last_pose is not None:
                travelled += math.hypot(view.pose.x - last_pose.x, view.pose.y - last_pose.y)
                if travelled >= args.max_travel_m:
                    reason = 'travel_limit'
                    break
            command, request, decision = runtime.cycle(view, now)
            counts[command.reason] += 1
            fields = {'reason': command.reason, 'linear': command.linear, 'angular': command.angular,
                      'final_action': command.action, 'clock_domain': 'monotonic'}
            if room is not None:
                fields['rear_memory_cells'] = room.reused_cells
                fields['rear_memory_age_sec'] = room.oldest_age
            if command.action != 'stop':
                fields['decision_id'] = runtime.navigation.goal.sequence
            if view is not None:
                ready_ticks += int(ready(view, now))
                fields.update(scan_at=view.scan_at, depth_at=view.depth_at, odom_at=view.odom_at,
                              snapshot_at=view.grid_at, status_at=view.status_at,
                              x=view.pose.x, y=view.pose.y, yaw=view.pose.yaw, epoch=view.epoch)
                first_pose = view.pose if first_pose is None else first_pose
                last_pose = view.pose
                if command.action in ('left', 'right'):
                    trace = arc_traces.setdefault(fields['decision_id'], {
                        'action': command.action, 'started_at': now, 'start_yaw_rad': view.pose.yaw})
                    trace.update(last_at=now, last_yaw_rad=view.pose.yaw)
            timeline.record('control', now, fields)
            if actuator is not None and actuator.unavailable_guards != unavailable_guards:
                unavailable_guards = actuator.unavailable_guards
                for stage, guard in (('ros_authority', ros_guard), ('serial_authority', serial_guard)):
                    if stage in unavailable_guards:
                        report = guard.diagnostic()
                        reason_code = report['reason'] if report['reason'] != 'ok' else 'command_deadline_unavailable'
                        timeline.record('run', now, {'stage': stage, 'reason': reason_code,
                                                     'status': 'latched' if report['fault_latched'] else 'sampled'})
            if request is not None:
                timeline.record('decision', now, {'decision_id': request.sequence, 'stage': 'requested',
                                                  'candidate_id': '.'.join(request.choices)})
            if decision is not None:
                timeline.record('decision', now, {'decision_id': decision[0], 'selected_action': decision[1],
                                                  'status': 'accepted' if decision[2] else 'rejected'})
            if runtime.room_continuation is not None:
                segment, action, origin_decision = runtime.room_continuation
                timeline.record('decision', now, {'decision_id': segment, 'selected_action': action,
                                                  'scene_id': origin_decision, 'stage': 'local_continuation'})
            if now - last_block_report >= 1.:
                last_block_report = now
                blocks = dict(runtime.navigation.candidate_blocks)
                if runtime.navigation.blocked_cell is not None:
                    blocks['tracking'] = runtime.navigation.blocked_cell
                for action, (block_reason, x, y) in blocks.items():
                    timeline.record('perception', now, {'stage': action, 'reason': block_reason, 'x': x, 'y': y})
            slot += .05
            remaining = slot - _monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                slot += (1 + int(-remaining / .05)) * .05
    except KeyboardInterrupt:
        reason = 'interrupted'
    except (ActuationError, IOError, OSError) as exc:
        reason = str(exc) if isinstance(exc, ActuationError) else type(exc).__name__
    finally:
        ended = _monotonic()
        if timeline is not None:
            timeline.record('stop', ended, {'reason': reason})
        try:
            if actuator is not None:
                try:
                    if heartbeat is not None:
                        heartbeat.close()
                    authority_diagnostics.update(ros=ros_guard.diagnostic(), serial=serial_guard.diagnostic())
                finally:
                    actuator.close()
        finally:
            try:
                model_joined = runtime.close(publish_stop=actuator is None) if runtime is not None else None
            finally:
                timeline_closed = timeline.close() if timeline is not None else None
                for child in reversed(children):
                    _stop_group(child)
    mode = 'private_ros_navigation'
    if args.drive_forward:
        mode = 'supervised_forward_navigation'
    elif args.drive_arcs:
        mode = 'supervised_static_room' if room is not None else 'supervised_arc_navigation'
    for trace in arc_traces.values():
        change = trace['last_yaw_rad'] - trace['start_yaw_rad']
        trace['heading_change_rad'] = math.atan2(math.sin(change), math.cos(change))
    result = {'mode': mode,
              'preview_topic': None if args.drive else preview, 'control': counts, 'reason': reason,
              'ready_ticks': ready_ticks, 'model_joined': model_joined, 'timeline_complete': timeline_closed,
              'grid_updates': source.grid_updates if source is not None else {},
              'decision_errors': source.decision_errors if source is not None else {},
              'arc_traces': arc_traces,
              'odometry_displacement_m': math.hypot(last_pose.x - first_pose.x, last_pose.y - first_pose.y)
              if first_pose is not None else None}
    if actuator is not None:
        trajectory = recorder.trial_trajectory(actuator.caller, started, _monotonic())
        result.update(relay_caller=actuator.caller, relay_fault=actuator.stop_line,
                      relay_exited=actuator.transport.exited, path_m=travelled,
                      authority_diagnostics=authority_diagnostics,
                      observed=summarize_navigation_trajectory(trajectory), trajectory=trajectory)
        if room is not None:
            result['room_loop'] = summarize_room_loop(trajectory, arc_traces)
    print(json.dumps(result, allow_nan=False))
    complete = ready_ticks and timeline_closed and model_joined
    if args.drive:
        complete = complete and actuator is not None and actuator.transport.exited and travelled >= .20
    if args.drive_arcs and room is not None and complete:
        complete = result['room_loop']['closed'] and result['room_loop']['continuous_commands']
    elif args.drive_arcs and complete:
        observed = result['observed']
        complete = observed['nonzero_angular_cmd_vel_messages'] and any(
            (1 if trace['action'] == 'left' else -1) * trace['heading_change_rad'] >= .05
            for trace in arc_traces.values())
    return 0 if complete else 2


if __name__ == '__main__':
    sys.exit(main())
