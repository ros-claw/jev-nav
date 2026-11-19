#!/usr/bin/env python2
"""Bounded native ROS sensing experiment; never publishes a velocity command.

The local grid, depth points, and reconstructed odom TF are private to a new
random namespace. A synthetic point can optionally test obstacle marking in
that private grid. This is not an executor or a safety/clearance certificate.
"""
from __future__ import division, print_function

import argparse
import collections
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from relay import _monotonic


def _yaw(quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    if (not all(not math.isnan(value) and not math.isinf(value) for value in (x, y, z, w))
            or not .99 <= x*x + y*y + z*z + w*w <= 1.01):
        raise ValueError('invalid orientation')
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def _sampled_front_cells(grid, odometry):
    """Summarize a small front region; this is not a clearance decision."""
    try:
        if (grid.header.frame_id.lstrip('/') != 'odom'
                or odometry.header.frame_id.lstrip('/') != 'odom'
                or odometry.child_frame_id.lstrip('/') != 'base_link'
                or abs((grid.header.stamp - odometry.header.stamp).to_sec()) > .5
                or grid.info.resolution <= 0
                or len(grid.data) != grid.info.width * grid.info.height):
            return None
        pose = odometry.pose.pose
        grid_yaw, robot_yaw = _yaw(grid.info.origin.orientation), _yaw(pose.orientation)
        gc, gs = math.cos(grid_yaw), math.sin(grid_yaw)
        rc, rs = math.cos(robot_yaw), math.sin(robot_yaw)
        counts = {'unknown': 0, 'free': 0, 'inflated': 0, 'occupied': 0}
        for index, value in enumerate(grid.data):
            x = (index % grid.info.width + .5) * grid.info.resolution
            y = (index // grid.info.width + .5) * grid.info.resolution
            dx = grid.info.origin.position.x + gc*x - gs*y - pose.position.x
            dy = grid.info.origin.position.y + gs*x + gc*y - pose.position.y
            forward, lateral = rc*dx + rs*dy, -rs*dx + rc*dy
            if .18 <= forward <= .6 and abs(lateral) <= .15:
                category = ('unknown' if value < 0 else 'free' if value == 0
                            else 'occupied' if value >= 100 else 'inflated')
                counts[category] += 1
        return counts
    except (ArithmeticError, AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return None


def _stop_group(process):
    """Stop only the process group started by this probe."""
    if process is None:
        return
    try:
        if os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGINT)
        elif process.poll() is None:
            process.terminate()
    except OSError:
        pass
    deadline = _monotonic() + 3.0
    while process.poll() is None and _monotonic() < deadline:
        time.sleep(.05)
    if process.poll() is None:
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass
    process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-synthetic-obstacle', action='store_true',
                        help='inject a fake point only into the isolated probe grid')
    parser.add_argument('--fresh-layer-workspace',
                        help='catkin workspace containing the experimental fresh laser layer')
    args = parser.parse_args(argv)
    namespace = 'limo_native_probe_' + uuid.uuid4().hex[:12]
    root = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ)
    launch_args = []
    if args.fresh_layer_workspace:
        workspace = os.path.abspath(args.fresh_layer_workspace)
        devel = os.path.join(workspace, 'devel')
        if not os.path.isfile(os.path.join(devel, 'lib', 'liblimo_fresh_layer.so')):
            parser.error('built native layer not found in workspace')
        sys.path.insert(0, os.path.join(devel, 'lib', 'python2.7', 'dist-packages'))
        for name, prefix in (('CMAKE_PREFIX_PATH', devel),
                             ('LD_LIBRARY_PATH', os.path.join(devel, 'lib')),
                             ('ROS_PACKAGE_PATH', os.path.join(workspace, 'src'))):
            env[name] = prefix + (':' + env[name] if env.get(name) else '')
        launch_args.append('use_fresh:=true')
    lifetime = ['/usr/bin/timeout', '--signal=INT', '--kill-after=3s', '35s']
    commands = [lifetime + [sys.executable, os.path.join(root, 'native_probe_tf.py'),
                            '--namespace', namespace],
                lifetime + ['/opt/ros/melodic/bin/roslaunch', os.path.join(root, 'native_costmap_probe.launch'),
                            'namespace:=' + namespace] + launch_args]
    processes, log = [], tempfile.TemporaryFile()  # noqa: SIM115 - closed after child cleanup
    rospy = None
    try:
        for command in commands:
            processes.append(subprocess.Popen(command, stdout=log, stderr=log, env=env,
                                              preexec_fn=os.setsid, close_fds=True))
        import rospy as ros
        from nav_msgs.msg import OccupancyGrid
        from sensor_msgs.msg import LaserScan, PointCloud2
        rospy = ros
        rospy.init_node('limo_native_sensing_probe', anonymous=True, disable_signals=True)
        lock = threading.Lock()
        received = []
        latest_grid = [None]
        target = [None]
        costs = []
        inputs = {'scan': {'count': 0, 'last_at': None, 'last_age': None},
                  'depth_points': {'count': 0, 'last_at': None, 'last_age': None}}

        def sensor(name, message):
            expected = 'laser_link' if name == 'scan' else 'camera_depth_optical_frame'
            if message.header.frame_id.lstrip('/') != expected:
                return
            age = (rospy.Time.now() - message.header.stamp).to_sec()
            present = bool(message.ranges) if name == 'scan' else bool(message.data and message.width)
            if not present or math.isnan(age) or math.isinf(age) or not 0 <= age <= 1.0:
                return
            with lock:
                inputs[name]['count'] += 1
                inputs[name]['last_at'] = _monotonic()
                inputs[name]['last_age'] = age

        def receive(grid, scan_stamp=None, scan_fresh=False, scan_age_at_receipt=None):
            now = _monotonic()
            with lock:
                counts = collections.Counter(grid.data)
                received.append((now, counts[-1], counts[0], counts[100], len(grid.data),
                                 grid.header.stamp.to_sec(), scan_stamp, scan_fresh, scan_age_at_receipt))
                latest_grid[0] = grid
                if target[0] is not None and grid.info.resolution > 0:
                    x = int(math.floor((target[0][0] - grid.info.origin.position.x) / grid.info.resolution))
                    y = int(math.floor((target[0][1] - grid.info.origin.position.y) / grid.info.resolution))
                    if 0 <= x < grid.info.width and 0 <= y < grid.info.height:
                        costs.append(grid.data[y * grid.info.width + x])

        subscribers = [rospy.Subscriber('/scan', LaserScan, lambda message: sensor('scan', message), queue_size=1),
                       rospy.Subscriber('/' + namespace + '/depth_points', PointCloud2,
                                        lambda message: sensor('depth_points', message), queue_size=1)]
        if args.fresh_layer_workspace:
            from limo_fresh_layer.msg import FreshGrid
            def receive_bound(message):
                checked, scan = message.header.stamp.to_sec(), message.scan_stamp.to_sec()
                if (message.grid.header.stamp != message.header.stamp
                        or message.grid.header.frame_id != message.header.frame_id
                        or not all(not math.isnan(value) and not math.isinf(value) for value in (checked, scan))
                        or message.scan_fresh and not 0 < scan <= checked
                        or not message.scan_fresh and scan != 0):
                    return
                age = (rospy.Time.now() - message.scan_stamp).to_sec() if message.scan_fresh else None
                if age is not None and (math.isnan(age) or math.isinf(age) or age < 0):
                    return
                receive(message.grid, scan, message.scan_fresh, age)
            subscribers.append(rospy.Subscriber('/' + namespace + '/rolling_costmap/costmap/snapshot/fresh_grid',
                                                FreshGrid, receive_bound, queue_size=1))
        else:
            subscribers.append(rospy.Subscriber('/' + namespace + '/rolling_costmap/costmap/costmap',
                                                OccupancyGrid, receive, queue_size=1))
        deadline = _monotonic() + 20.0
        ready = False
        while _monotonic() < deadline and not rospy.is_shutdown():
            if any(process.poll() is not None for process in processes):
                raise RuntimeError('probe_process_exited')
            with lock:
                now = _monotonic()
                ready = (len(received) >= 8 and all(
                    item['count'] >= 3 and item['last_at'] is not None and now - item['last_at'] <= .7
                    for item in inputs.values()))
                if args.fresh_layer_workspace:
                    ready = (ready and any(item[7] and 0 <= item[5] - item[6] <= .3
                                           and item[8] is not None and 0 <= item[8] <= .3
                                           and now - item[0] <= .7 for item in received[-10:]))
            if ready:
                break
            time.sleep(.05)
        if not ready:
            with lock:
                detail = {'grid_samples': len(received),
                          'scan_messages': inputs['scan']['count'],
                          'depth_messages': inputs['depth_points']['count'],
                          'bound_fresh_grid_samples': sum(1 for item in received if item[7])}
            raise RuntimeError('grid_timeout:' + json.dumps(detail, sort_keys=True))

        from nav_msgs.msg import Odometry
        odometry = rospy.wait_for_message('/odom', Odometry, timeout=3)
        with lock:
            front_cells = _sampled_front_cells(latest_grid[0], odometry)
        if args.test_synthetic_obstacle:
            from sensor_msgs import point_cloud2
            from std_msgs.msg import Header
            pose = rospy.wait_for_message('/odom', Odometry, timeout=3).pose.pose
            yaw = _yaw(pose.orientation)
            if not all(not math.isnan(value) and not math.isinf(value)
                       for value in (pose.position.x, pose.position.y)):
                raise RuntimeError('invalid_probe_pose')
            with lock:
                target[0] = (pose.position.x + .5 * math.cos(yaw), pose.position.y + .5 * math.sin(yaw))
            publisher = rospy.Publisher('/' + namespace + '/depth_points', PointCloud2, queue_size=1)
            until = _monotonic() + 4.0
            while _monotonic() < until and not rospy.is_shutdown():
                header = Header(stamp=rospy.Time.now(), frame_id='base_link')
                publisher.publish(point_cloud2.create_cloud_xyz32(header, [[.5, 0., .1]]))
                time.sleep(.1)
            with lock:
                marked = sum(value == 100 for value in costs)
            if marked == 0:
                raise RuntimeError('synthetic_obstacle_not_marked')
        else:
            marked = None
        with lock:
            points = list(received)
            checked_at = _monotonic()
            input_summary = {name: {'messages': item['count'], 'last_age_at_receipt_sec': item['last_age'],
                                    'since_last_receipt_sec': checked_at - item['last_at']}
                             for name, item in inputs.items()}
        if any(item['since_last_receipt_sec'] > .7 for item in input_summary.values()):
            raise RuntimeError('probe_input_stale')
        paired = None
        scan_margin_ms = None
        if args.fresh_layer_workspace:
            paired = sum(1 for point in points if point[7] and point[6] is not None
                         and point[8] is not None and 0 <= point[8] <= .3
                         and 0 <= point[5] - point[6] <= .3)
            if paired == 0:
                raise RuntimeError('scan_evidence_unpaired')
            margins = sorted(1000 * (.3 - point[8]) for point in points if point[7]
                             and point[8] is not None and 0 <= point[8] <= .3)
            scan_margin_ms = {'min': round(margins[0], 1), 'median': round(margins[len(margins)//2], 1)}
        for subscriber in subscribers:
            subscriber.unregister()
        elapsed = points[-1][0] - points[0][0]
        result = {'mode': 'isolated_sensing_probe', 'motion_requested': False,
                  'global_tf_published': False, 'physical_clearance_verified': False,
                  'experimental_fresh_laser_layer': bool(args.fresh_layer_workspace),
                  'grid_samples': len(points), 'grid_rate_hz': round((len(points) - 1) / elapsed, 2)
                  if elapsed > 0 else None,
                  'unknown_cells': points[-1][1], 'free_cells': points[-1][2],
                  'occupied_cells': points[-1][3], 'total_cells': points[-1][4],
                  'inputs': input_summary,
                  'sampled_front_cells': front_cells,
                  'grid_samples_with_bound_scan': paired,
                  'scan_validity_remaining_ms_at_receipt': scan_margin_ms,
                  'synthetic_obstacle_lethal_samples': marked}
    finally:
        for process in reversed(processes):
            _stop_group(process)
        if rospy is not None:
            try:
                rospy.delete_param('/' + namespace)
            except (KeyError, rospy.ROSException):
                pass
        log.close()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        known = ('probe_process_exited', 'grid_timeout', 'invalid_probe_pose',
                 'synthetic_obstacle_not_marked', 'probe_input_stale', 'scan_evidence_unpaired')
        reason = str(error) if isinstance(error, RuntimeError) else ''
        detail = None
        if reason.startswith('grid_timeout:'):
            detail, reason = json.loads(reason.split(':', 1)[1]), 'grid_timeout'
        print(json.dumps({'mode': 'isolated_sensing_probe', 'error_type': type(error).__name__,
                          'reason': reason if reason in known else None, 'detail': detail,
                          'motion_requested': False}))
        sys.exit(1)
