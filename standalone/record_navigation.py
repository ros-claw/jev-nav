#!/usr/bin/env python3
"""Run standalone navigation and keep a local, bounded record of a session."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid


MIB = 1024 ** 2
MIN_FREE = 256 * MIB
DIRECTORY = Path(__file__).resolve().parent


def recording_topics(namespace, extra):
    prefix = '/' + namespace
    cameras = (('/camera/depth/image_raw', '0.5', prefix + '/recording/depth'),)
    topics = [
        '/cmd_vel', '/odom', '/scan', '/limo_status', '/rosclaw_limo/command_watchdog',
        '/tf', '/tf_static', '/camera/depth/camera_info',
        '/rtabmap/info', '/rtabmap/mapGraph',
        prefix + '/tf',
        prefix + '/rolling_costmap/costmap/snapshot/fresh_grid',
        prefix + '/navigation_command_preview',
    ] + [camera[2] for camera in cameras] + extra
    return list(dict.fromkeys(topics)), cameras


def bag_command(directory, topics):
    # rosrun execs the writer so its exit is the writer's, including bag finalization.
    return ['rosrun', 'rosbag', 'record', '--lz4', '--split', '--size', '64', '--buffsize', '32',
            '--min-space', '{}M'.format(MIN_FREE // MIB), '--tcpnodelay',
            '-O', str(directory / 'sensors')] + topics


def bag_bytes(directory):
    size = 0
    for path in directory.glob('sensors_*.bag*'):
        try:
            size += path.stat().st_size
        except FileNotFoundError:
            # The recorder can rename a finished segment between glob and stat.
            continue
    return size


def storage_problem(directory, budget_bytes):
    if shutil.disk_usage(str(directory)).free < MIN_FREE:
        return 'disk_space_low'
    if bag_bytes(directory) >= budget_bytes:
        return 'bag_budget_reached'
    return None


def clock_anchor():
    before = time.time()
    monotonic = time.monotonic()
    after = time.time()
    return {'wall_before': before, 'wall_after': after, 'monotonic': monotonic}


def save_manifest(directory, manifest):
    temp = directory / 'manifest.tmp'
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(directory / 'manifest.json')


def launch(directory, name, command, result=None):
    with (directory / (name + '.log')).open('wb') as log:
        if result is None:
            return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with (directory / result).open('wb') as output:
            return subprocess.Popen(command, stdout=output, stderr=log, start_new_session=True)


def stop(process, timeout):
    if process is None or process.poll() is not None:
        return
    for event, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 2.), (signal.SIGKILL, 2.)):
        try:
            os.killpg(process.pid, event)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--budget-mib', default=512, type=int)
    parser.add_argument('--extra-topic', action='append', default=[])
    parser.add_argument('--map-database', type=Path, help='start RTAB-Map on the same private odometry TF stream')
    parser.add_argument('navigation', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (not args.navigation or args.navigation[0] != '--' or len(args.navigation) == 1
            or not 16 <= args.budget_mib <= 2048
            or any(not topic.startswith('/') for topic in args.extra_topic)):
        parser.error('use recorder options, then -- and navigation arguments')
    if not args.output_root.is_dir():
        parser.error('output root must already exist')
    if args.map_database is not None and not args.map_database.parent.is_dir():
        parser.error('map database directory must already exist')
    budget = args.budget_mib * MIB
    if shutil.disk_usage(str(args.output_root)).free < MIN_FREE + budget:
        parser.error('output filesystem has insufficient space for the recording budget')
    uid = uuid.uuid4().hex[:12]
    namespace = 'limo_native_probe_' + uid
    label = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    directory = args.output_root.resolve() / ('room-' + label + '-' + uid)
    directory.mkdir(mode=0o700)
    print(str(directory), flush=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(DIRECTORY)).decode().strip()
    working_tree_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(DIRECTORY)))
    topics, cameras = recording_topics(namespace, args.extra_topic)
    manifest = {'namespace': namespace, 'revision': revision, 'working_tree_dirty': working_tree_dirty,
                'phase': 'starting',
                'recording_start': clock_anchor(), 'topics': topics, 'budget_mib': args.budget_mib,
                'camera_samples_hz': {destination: float(rate) for _, rate, destination in cameras}}
    save_manifest(directory, manifest)
    bag = navigation = mapper = None
    samplers = []
    interrupted = False

    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    for event in (signal.SIGINT, signal.SIGTERM):
        signal.signal(event, interrupt)
    reason = None
    try:
        bag = launch(directory, 'rosbag', bag_command(directory, topics))
        for index, (source, rate, destination) in enumerate(cameras):
            samplers.append(launch(directory, 'camera-{}'.format(index),
                                   ['rosrun', 'topic_tools', 'throttle', 'messages', source, rate, destination]))
        until = time.monotonic() + 12.
        while (not interrupted and bag.poll() is None and time.monotonic() < until
               and bag_bytes(directory) == 0):
            time.sleep(.1)
        if interrupted:
            reason = 'interrupted'
        elif bag.poll() is not None or bag_bytes(directory) == 0:
            reason = 'bag_not_ready'
        elif any(sampler.poll() is not None for sampler in samplers):
            reason = 'camera_sampler_exited'
        else:
            if args.map_database is not None:
                mapper = launch(directory, 'mapper',
                                ['roslaunch', str(DIRECTORY / 'exploration_mapping.launch'),
                                 'tf_topic:=/' + namespace + '/tf',
                                 'database_path:=' + str(args.map_database.resolve())])
            manifest.update(phase='navigating', navigation_start=clock_anchor())
            if args.map_database is not None:
                manifest['map_database'] = str(args.map_database.resolve())
            save_manifest(directory, manifest)
            navigation = launch(directory, 'navigation',
                                ['/usr/bin/python2', str(DIRECTORY / 'run_navigation.py')]
                                + args.navigation[1:] + ['--timeline', str(directory / 'timeline.jsonl'),
                                                         '--session-namespace', namespace],
                                'navigation.json')
            while navigation.poll() is None and not reason:
                if interrupted:
                    reason = 'interrupted'
                elif bag.poll() is not None:
                    reason = 'bag_exited'
                elif mapper is not None and mapper.poll() is not None:
                    reason = 'mapper_exited'
                elif any(sampler.poll() is not None for sampler in samplers):
                    reason = 'camera_sampler_exited'
                else:
                    reason = storage_problem(directory, budget)
                if reason is None:
                    time.sleep(.25)
    except OSError as error:
        reason = 'io_error_' + str(error.errno)
    finally:
        stop(navigation, 10.)
        stop(mapper, 8.)
        stop(bag, 8.)
        for sampler in samplers:
            stop(sampler, 2.)
        bags = sorted(directory.glob('sensors_*.bag'))
        finalized = bool(bags) and not list(directory.glob('sensors_*.bag.active'))
        manifest.update(phase='finished', recording_end=clock_anchor(), recording_stop=reason,
                        navigation_exit=navigation.returncode if navigation is not None else None,
                        mapper_exit=mapper.returncode if mapper is not None else None,
                        recorder_exit=bag.returncode if bag is not None else None,
                        bag_finalized=finalized,
                        bags=[{'name': path.name, 'bytes': path.stat().st_size} for path in bags])
        save_manifest(directory, manifest)
    return 2 if reason or not finalized or navigation is None else navigation.returncode


if __name__ == '__main__':
    raise SystemExit(main())
