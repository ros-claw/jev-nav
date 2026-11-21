#!/usr/bin/env python2
"""Run odometry-frame navigation using live ROS perception and private output."""
from __future__ import division, print_function

import argparse
import base64
import collections
import json
import math
import os
import subprocess
import sys
import time
import uuid

from navigation import Grid, Pose, View, ready
from navigation_runtime import NavigationRuntime
from native_sensing_probe import _stop_group, _yaw
from relay import _monotonic


class RosNavigationInput(object):
    def __init__(self, rospy, grid_topic, decisions):
        from explore import RosObservation
        from limo_fresh_layer.msg import FreshGrid
        from sensor_msgs.msg import Image

        self.rospy = rospy
        self.observation = RosObservation(sensor_names=('odom', 'status'))
        self.latest_grid = None
        self.last_grid_times = None
        self.latest_color = None
        self.grid_updates = collections.Counter()
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
        jpeg = CvJpegEncoder(observation.bridge)(image)
        evidence = {'image_b64': base64.b64encode(jpeg).decode('ascii'), 'captured_at': stamp,
                    'laser': {}, 'base': state, 'navigation_choices': list(request.choices)}
        result = run_decision_process(['/usr/bin/python3', os.path.join(WORKER, 'limo_mapless_decision.py')],
                                      evidence, cancelled, _monotonic, diagnostics=True)
        if result and result.get('ok') is True and result.get('captured_at') == stamp:
            return result.get('choice')
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeline', required=True)
    parser.add_argument('--seconds', type=float, help='end a recorded validation run after this many seconds')
    parser.add_argument('--decisions', action='store_true', help='enable Decisions API calls for checked candidates')
    args = parser.parse_args(argv)
    if args.seconds is not None and not 0. < args.seconds < float('inf'):
        parser.error('seconds must be positive and finite')

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
                 'inflate:=false',
                 'depth_layer_type:=limo_fresh_layer/FreshDepthLayer'])
    children = []
    runtime = timeline = None
    counts = collections.Counter()
    ready_ticks = 0
    first_pose = last_pose = None
    try:
        for command in commands:
            children.append(subprocess.Popen(command, preexec_fn=os.setsid, stdout=sys.stderr,
                                             stderr=sys.stderr, close_fds=True))
        rospy.init_node('jev_navigation', anonymous=True)
        if rospy.get_param('/use_sim_time', False):
            raise RuntimeError('ROS wall time required for live acquisition')
        source = RosNavigationInput(rospy, costmap, args.decisions)
        publisher = rospy.Publisher(preview, Twist, queue_size=1)
        timeline = Timeline(args.timeline, namespace)

        def publish(command):
            packet = Twist()
            packet.linear.x, packet.angular.z = command.linear, command.angular
            if not rospy.is_shutdown():
                publisher.publish(packet)

        choose = source.choose if args.decisions else lambda request, cancelled: None
        runtime = NavigationRuntime(choose, publish)
        started = slot = last_process_check = last_block_report = _monotonic()
        while not rospy.is_shutdown():
            now = _monotonic()
            if args.seconds is not None and now - started >= args.seconds:
                break
            if now - last_process_check >= .5:
                last_process_check = now
                if any(child.poll() is not None for child in children):
                    raise RuntimeError('navigation perception process exited')
            view = source.view()
            now = _monotonic()
            command, request, decision = runtime.cycle(view, now)
            counts[command.reason] += 1
            fields = {'reason': command.reason, 'linear': command.linear, 'angular': command.angular,
                      'final_action': command.action, 'clock_domain': 'monotonic'}
            if view is not None:
                ready_ticks += int(ready(view, now))
                fields.update(scan_at=view.scan_at, depth_at=view.depth_at, odom_at=view.odom_at,
                              snapshot_at=view.grid_at, status_at=view.status_at,
                              x=view.pose.x, y=view.pose.y, yaw=view.pose.yaw, epoch=view.epoch)
                first_pose = view.pose if first_pose is None else first_pose
                last_pose = view.pose
            timeline.record('control', now, fields)
            if request is not None:
                timeline.record('decision', now, {'decision_id': request.sequence, 'stage': 'requested',
                                                  'candidate_id': '.'.join(request.choices)})
            if decision is not None:
                timeline.record('decision', now, {'decision_id': decision[0], 'selected_action': decision[1],
                                                  'status': 'accepted' if decision[2] else 'rejected'})
            if now - last_block_report >= 1.:
                last_block_report = now
                blocks = dict(runtime.navigation.candidate_blocks)
                if runtime.navigation.blocked_cell is not None:
                    blocks['tracking'] = runtime.navigation.blocked_cell
                for action, (reason, x, y) in blocks.items():
                    timeline.record('perception', now, {'stage': action, 'reason': reason, 'x': x, 'y': y})
            slot += .05
            remaining = slot - _monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                slot += (1 + int(-remaining / .05)) * .05
    except KeyboardInterrupt:
        pass
    finally:
        model_joined = runtime.close() if runtime is not None else None
        timeline_closed = timeline.close() if timeline is not None else None
        for child in reversed(children):
            _stop_group(child)
    print(json.dumps({'mode': 'private_ros_navigation', 'preview_topic': preview, 'control': counts,
                      'ready_ticks': ready_ticks, 'model_joined': model_joined, 'timeline_complete': timeline_closed,
                      'grid_updates': source.grid_updates,
                      'odometry_displacement_m': math.hypot(last_pose.x - first_pose.x, last_pose.y - first_pose.y)
                      if first_pose is not None else None}, allow_nan=False))
    return 0 if ready_ticks and timeline_closed and model_joined else 2


if __name__ == '__main__':
    sys.exit(main())
