#!/usr/bin/env python2
"""Exercise the experimental laser layer using only synthetic private ROS topics."""
from __future__ import division, print_function

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from native_sensing_probe import _stop_group
from relay import _monotonic


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True, help='catkin workspace containing the built plugin')
    invalid = parser.add_mutually_exclusive_group()
    invalid.add_argument('--unsafe-inf-config', action='store_true',
                         help='verify that enabling Inf clearing keeps the grid unknown')
    invalid.add_argument('--unsafe-unknown-config', action='store_true',
                         help='verify that disabling unknown tracking keeps the grid unknown')
    args = parser.parse_args(argv)
    workspace = os.path.abspath(args.workspace)
    devel = os.path.join(workspace, 'devel')
    if not os.path.isfile(os.path.join(devel, 'lib', 'liblimo_fresh_layer.so')):
        parser.error('built native layer not found in workspace')
    sys.path.insert(0, os.path.join(devel, 'lib', 'python2.7', 'dist-packages'))
    namespace = 'limo_native_probe_' + uuid.uuid4().hex[:12]
    root = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ)
    for name, prefix in (('CMAKE_PREFIX_PATH', devel),
                         ('LD_LIBRARY_PATH', os.path.join(devel, 'lib')),
                         ('ROS_PACKAGE_PATH', os.path.join(workspace, 'src'))):
        env[name] = prefix + (':' + env[name] if env.get(name) else '')
    command = ['/usr/bin/timeout', '--signal=INT', '--kill-after=3s', '55s',
               '/opt/ros/melodic/bin/roslaunch', os.path.join(root, 'fresh_laser_probe.launch'),
               'namespace:=' + namespace]
    if args.unsafe_inf_config:
        command.append('allow_inf:=true')
    if args.unsafe_unknown_config:
        command.append('track_unknown:=false')
    invalid_label = ('invalid_inf_config' if args.unsafe_inf_config else
                     'invalid_unknown_config' if args.unsafe_unknown_config else None)
    log = tempfile.TemporaryFile()  # noqa: SIM115 - closed after the process group exits
    process = None
    rospy = None
    try:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env,
                                   preexec_fn=os.setsid, close_fds=True)
        import rospy as ros
        from dynamic_reconfigure.client import Client
        from geometry_msgs.msg import TransformStamped
        from limo_fresh_layer.msg import FreshGrid
        from nav_msgs.msg import OccupancyGrid
        from sensor_msgs import point_cloud2
        from sensor_msgs.msg import LaserScan, PointCloud2, TimeReference
        from std_msgs.msg import Header
        from tf2_msgs.msg import TFMessage
        rospy = ros
        rospy.init_node('limo_fresh_laser_test', anonymous=True, disable_signals=True)
        transforms = rospy.Publisher('/' + namespace + '/tf', TFMessage, queue_size=10)
        scans = rospy.Publisher('/' + namespace + '/scan', LaserScan, queue_size=1)
        depth = rospy.Publisher('/' + namespace + '/depth_points', PointCloud2, queue_size=1)
        lock = threading.Lock()
        phase = [None, 0.0]
        values = []
        evidence = []
        snapshots = []

        def grid_callback(grid):
            x = int(math.floor((.525 - grid.info.origin.position.x) / grid.info.resolution))
            y = int(math.floor((.025 - grid.info.origin.position.y) / grid.info.resolution))
            if 0 <= x < grid.info.width and 0 <= y < grid.info.height:
                with lock:
                    values.append((phase[0], _monotonic() - phase[1], grid.data[y * grid.info.width + x]))

        sub = rospy.Subscriber('/' + namespace + '/rolling_costmap/costmap/costmap',
                               OccupancyGrid, grid_callback, queue_size=1)

        def evidence_callback(message):
            age = (message.header.stamp - message.time_ref).to_sec() if message.time_ref.to_sec() else None
            with lock:
                evidence.append((phase[0], _monotonic() - phase[1], message.source, age))

        evidence_sub = rospy.Subscriber('/' + namespace + '/rolling_costmap/costmap/fresh_laser/scan_evidence',
                                        TimeReference, evidence_callback, queue_size=10)

        def snapshot_callback(message):
            grid = message.grid
            x = int(math.floor((.525 - grid.info.origin.position.x) / grid.info.resolution))
            y = int(math.floor((.025 - grid.info.origin.position.y) / grid.info.resolution))
            if 0 <= x < grid.info.width and 0 <= y < grid.info.height:
                age = (message.header.stamp - message.scan_stamp).to_sec() if message.scan_fresh else None
                with lock:
                    snapshots.append((phase[0], _monotonic() - phase[1], message.scan_fresh, age,
                                      grid.data[y * grid.info.width + x]))

        snapshot_sub = rospy.Subscriber('/' + namespace + '/rolling_costmap/costmap/snapshot/fresh_grid',
                                        FreshGrid, snapshot_callback, queue_size=10)

        def publish_tf(stamp):
            entries = []
            for parent, child in (('odom', 'base_link'), ('base_link', 'laser_link')):
                item = TransformStamped()
                item.header.stamp = stamp
                item.header.frame_id = parent
                item.child_frame_id = child
                item.transform.rotation.w = 1.0
                entries.append(item)
            transforms.publish(TFMessage(entries))

        def publish_scan(stamp, scan_type):
            scan = LaserScan()
            scan.header.frame_id = 'laser_link'
            scan.header.stamp = stamp
            scan.angle_min, scan.angle_increment = -math.pi/2, math.pi/180
            scan.angle_max = scan.angle_min + 180 * scan.angle_increment
            scan.range_min, scan.range_max, scan.scan_time = .1, 5.0, .1
            scan.ranges = [float('nan') if scan_type == 'unknown' else 2.0] * 181
            scans.publish(scan)

        deadline = _monotonic() + 12.0
        if invalid_label:
            while _monotonic() < deadline and process.poll() is None:
                publish_tf(rospy.Time.now())
                time.sleep(.05)
            if process.poll() is None:
                raise RuntimeError('invalid_configuration_not_rejected:tf=%d,scan=%d' %
                                   (transforms.get_num_connections(), scans.get_num_connections()))
            log.flush()
            log.seek(0)
            if b'fresh laser layer configuration is unsafe or unsupported' not in log.read():
                raise RuntimeError('unexpected_layer_exit')
            with lock:
                if any(value != -1 for _, _, value in values):
                    raise RuntimeError('invalid_configuration_published_free_space')
            print(json.dumps({'mode': 'synthetic_fresh_laser_probe', 'motion_requested': False,
                              'invalid_configuration': invalid_label, 'rejected': True,
                              'physical_clearance_verified': False}, sort_keys=True))
            return 0
        while _monotonic() < deadline and any(publisher.get_num_connections() == 0
                                               for publisher in (transforms, scans, depth)):
            if process.poll() is not None:
                raise RuntimeError('layer_process_exited:%s' % process.returncode)
            publish_tf(rospy.Time.now())
            time.sleep(.05)
        if any(publisher.get_num_connections() == 0 for publisher in (transforms, scans, depth)):
            raise RuntimeError('layer_connections_timeout')

        dynamic = Client('/' + namespace + '/rolling_costmap/costmap/fresh_laser', timeout=3)
        phases = (('fresh', 3.0, 'valid'), ('dropout', 1.4, None), ('old', 1.2, 'old'),
                  ('recovered', 1.6, 'valid'), ('unknown', 1.2, 'unknown'),
                  ('disabled', 1.2, 'valid'), ('reenabled', 1.2, 'valid'),
                  ('depth', 1.5, 'valid'), ('retained_depth', 1.2, 'valid'))
        for label, duration, scan_type in phases:
            if label == 'disabled':
                dynamic.update_configuration({'enabled': False})
            elif label == 'reenabled':
                dynamic.update_configuration({'enabled': True})
            with lock:
                phase[:] = [label, _monotonic()]
            until = _monotonic() + duration
            while _monotonic() < until and not rospy.is_shutdown():
                if process.poll() is not None:
                    raise RuntimeError('layer_process_exited:%s' % process.returncode)
                stamp = rospy.Time.now()
                publish_tf(stamp)
                if scan_type is not None:
                    scan_stamp = stamp - rospy.Duration(2.0) if scan_type == 'old' else stamp
                    publish_scan(scan_stamp, scan_type)
                if label == 'depth':
                    header = Header(stamp=stamp, frame_id='base_link')
                    depth.publish(point_cloud2.create_cloud_xyz32(header, [[.525, .025, .1]]))
                time.sleep(.1)
        sub.unregister()
        evidence_sub.unregister()
        snapshot_sub.unregister()
        with lock:
            observations = list(values)
            checked = list(evidence)
            captured = list(snapshots)
        late = {name: [value for label, age, value in observations if label == name and age >= .7]
                for name, _, _ in phases}
        normal_result = (not all(late.values()) or not any(value == 0 for value in late['fresh'])
                         or not any(value == 0 for value in late['recovered'])
                         or not any(value == 0 for value in late['reenabled'])
                         or any(value != -1 for name in ('dropout', 'old', 'unknown', 'disabled')
                                for value in late[name])
                         or any(value != 100 for name in ('depth', 'retained_depth') for value in late[name]))
        if normal_result:
            raise RuntimeError('freshness_expectations_failed:' + json.dumps(late, sort_keys=True))
        checked_late = {name: [(source, age) for label, elapsed, source, age in checked
                               if label == name and elapsed >= .7]
                        for name, _, _ in phases}
        if (not all(checked_late.values())
                or not any(source == 'fresh_scan' and age is not None and 0 <= age <= .3
                           for source, age in checked_late['fresh'])
                or any(source != 'unavailable' or age is not None
                       for name in ('dropout', 'old', 'disabled') for source, age in checked_late[name])):
            raise RuntimeError('scan_evidence_expectations_failed:' + json.dumps(checked_late, sort_keys=True))
        snapshots_late = {name: [(fresh, age, value) for label, elapsed, fresh, age, value in captured
                                 if label == name and elapsed >= .7] for name, _, _ in phases}
        if (not all(snapshots_late.values())
                or not any(fresh and age is not None and 0 <= age <= .3 and value == 0
                           for fresh, age, value in snapshots_late['fresh'])
                or any(fresh or age is not None or value != -1
                       for name in ('dropout', 'old', 'disabled') for fresh, age, value in snapshots_late[name])
                or any(value != 100 for name in ('depth', 'retained_depth')
                       for _, _, value in snapshots_late[name])):
            raise RuntimeError('bound_grid_expectations_failed:' + json.dumps(snapshots_late, sort_keys=True))
        print(json.dumps({'mode': 'synthetic_fresh_laser_probe', 'motion_requested': False,
                          'global_tf_published': False, 'phase_cells': late,
                          'scan_evidence_phase_counts': {name: len(items) for name, items in checked_late.items()},
                          'bound_grid_phase_counts': {name: len(items) for name, items in snapshots_late.items()},
                          'physical_clearance_verified': False}, sort_keys=True))
        return 0
    finally:
        _stop_group(process)
        if rospy is not None:
            try:
                rospy.delete_param('/' + namespace)
            except (KeyError, rospy.ROSException):
                pass
        log.close()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'mode': 'synthetic_fresh_laser_probe', 'error_type': type(error).__name__,
                          'reason': str(error) if isinstance(error, RuntimeError) else None,
                          'motion_requested': False}))
        sys.exit(1)
