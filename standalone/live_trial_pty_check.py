#!/usr/bin/env python3
"""Isolated integration of LiveTrial, its real relay, and a fake PTY chassis.

Only run inside the private PID, device and network bubblewrap namespace. The
source, model, ROS arbitration and serial guards are test fixtures. Results
prove only simulated serial output and never physical stopping or clearance.

From the jev-nav repository root:

    env -i PATH=/usr/bin:/bin:/sbin bwrap --unshare-all \
      --hostname limo-pty-isolated --die-with-parent \
      --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib --ro-bind /sbin /sbin \
      --ro-bind /opt/ros/melodic /opt/ros/melodic --proc /proc --dev /dev --tmpfs /tmp \
      --ro-bind /tmp/rosclaw-limo-exclusive-candidate /tmp/rosclaw-limo-exclusive-candidate \
      --ro-bind "$PWD/standalone" /sandbox \
      --ro-bind "$PWD/worker" /worker \
      --dir /etc --ro-bind /etc/hosts /etc/hosts \
      --ro-bind /etc/nsswitch.conf /etc/nsswitch.conf --ro-bind /etc/passwd /etc/passwd \
      --ro-bind /etc/group /etc/group --ro-bind /etc/alternatives /etc/alternatives \
      --ro-bind /etc/ld.so.cache /etc/ld.so.cache --chdir /tmp \
      --setenv ROS_HOME /tmp/ros-home --setenv PYTHONDONTWRITEBYTECODE 1 \
      /usr/bin/python3 /sandbox/live_trial_pty_check.py

Append ``--self-test`` to run only the offline frame-classification tests.
The script checks for isolation before creating its PTY or any ROS process.
"""

import hashlib
import json
import os
import pty
import subprocess
import sys
import tempfile
import time
import tty
import unittest
from pathlib import Path

from driver_pty_check import _free_port, _terminate, _wait_master
from relay_deadline_pty_check import _check_safe_namespace
from relay_pty_check import STATUS_MONITOR, FakeChassis, StatusCollector, wait_for

ZERO = (0, 0, 0, 0)
MOTION = (20, 0, 0, 0)
MIN_PRE_MOTION_ZERO_WRITES = 3
MOTION_LIMIT_SEC = 0.3

CHILD = r'''
from __future__ import print_function
import json, time
from live_trial_adapter import DecisionsAdapter, LiveTrial, RelayAdapter, RelayProcessTransport
from relay import _monotonic
from test_live_trial_adapter import FakeArbiter, FakeSerialGuard, source

class RealMonotonicClock(object):
    def now(self):
        return _monotonic()
    def sleep(self, seconds):
        time.sleep(seconds)

clock = RealMonotonicClock()
observation, _ = source(clock)
calls = []
def fake_model(payload, timeout):
    calls.append(True)
    return {'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']}
model = DecisionsAdapter(fake_model)
transport = RelayProcessTransport(clock)
writes = []
status_reads = []
real_write = transport.write_line
real_read = transport.read_line
def record_write(data, timeout):
    accepted = real_write(data, timeout)
    if accepted:
        writes.append({'at': clock.now(), 'frame': json.loads(data)})
    return accepted
transport.write_line = record_write
def record_read(timeout):
    line = real_read(timeout)
    if line is not None:
        status_reads.append({'at': clock.now(), 'line': line})
    return line
transport.read_line = record_read
relay = RelayAdapter(transport, clock)
arbiter, serial = FakeArbiter(), FakeSerialGuard()
trial = LiveTrial(observation, model, relay, arbiter, clock, enable_motion=True,
                  duration_sec=0.6, speed_mps=0.02, period_sec=0.05, serial_guard=serial)
started = clock.now()
result = trial.run()
returned = clock.now()
print(json.dumps({'trial': result, 'writes': writes, 'status_reads': status_reads, 'model_calls': len(calls),
                  'started_at': started, 'returned_at': returned,
                  'fake_arbiter_released': arbiter.released,
                  'fake_serial_released': serial.released}, sort_keys=True))
'''


def summarize_frames(frames, started_at, returned_at, stop_at=None, first_write_at=None, motion_limit_sec=0.3):
    relevant = [(stamp, frame) for stamp, frame in frames if stamp >= started_at]
    if not relevant or any(frame not in (ZERO, MOTION) for _, frame in relevant):
        raise ValueError('unexpected serial speed or missing frames')
    motion = [(stamp, frame) for stamp, frame in relevant if frame == MOTION]
    if not motion:
        raise ValueError('no 0.02 m/s frame')
    first_motion, last_motion = motion[0][0], motion[-1][0]
    if stop_at is None:
        stop_at = returned_at
    if first_write_at is None:
        first_write_at = first_motion
    after_pipe_deadline = [(stamp, frame) for stamp, frame in motion if stamp >= first_write_at + motion_limit_sec]
    after_stop = [(stamp, frame) for stamp, frame in relevant if stamp >= stop_at]
    after_return = [(stamp, frame) for stamp, frame in relevant if stamp >= returned_at]
    after_motion_zero = [stamp for stamp, frame in relevant if stamp > last_motion and frame == ZERO]
    if (last_motion - first_motion > motion_limit_sec + 0.1 or len(after_stop) < 4
            or not after_motion_zero or any(frame != ZERO for _, frame in after_stop + after_return)):
        raise ValueError('bounded motion and repeated post-STOP zeros not demonstrated')
    result = {'forward_frames': len(motion),
            'first_nonzero_monotonic_s': first_motion,
            'last_nonzero_monotonic_s': last_motion,
            'serial_nonzero_span_sec': round(last_motion - first_motion, 6),
            'first_zero_after_last_nonzero_sec': round(after_motion_zero[0] - last_motion, 6),
            'motion_limit_sec': motion_limit_sec,
            'nonzero_after_motion_limit': len(after_pipe_deadline),
            'last_nonzero_minus_motion_limit_sec': round(last_motion - first_write_at - motion_limit_sec, 6),
            'post_stop_zero_frames': len(after_stop), 'post_stop_nonzero_frames': 0,
            'first_serial_zero_after_stop_observed_sec': round(after_stop[0][0] - stop_at, 6),
            'post_return_zero_frames': len(after_return), 'post_return_nonzero_frames': 0,
            'post_return_serial_idle': not after_return}
    if motion_limit_sec == 0.3:
        result['nonzero_after_first_pipe_stamp_plus_0_3'] = len(after_pipe_deadline)
        result['last_nonzero_minus_first_pipe_stamp_plus_0_3_sec'] = round(last_motion - first_write_at - 0.3, 6)
    return result


def main():
    # This must precede even allocating the PTY or starting the ROS master.
    _check_safe_namespace()
    candidate = Path('/tmp/rosclaw-limo-exclusive-candidate/devel/lib/limo_base/limo_base_node')
    if not candidate.is_file():
        raise RuntimeError('isolated exclusive candidate is missing')
    artifacts = [candidate] + [Path('/sandbox') / name for name in
                               ('live_trial_adapter.py', 'forward_trial.py', 'relay.py')]
    digests = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
    master_fd, slave_fd = pty.openpty()
    tty.setraw(slave_fd)
    pty_name = os.ttyname(slave_fd)
    if not pty_name.startswith('/dev/pts/'):
        raise RuntimeError('serial target is not the isolated PTY')
    master = driver = monitor = child = collector = None
    fake = FakeChassis(master_fd)
    with tempfile.TemporaryDirectory(prefix='live-trial-pty-') as temporary:
        run_dir = Path(temporary)
        link = run_dir / 'ttyFake'
        link.symlink_to(pty_name)
        if os.stat(str(link)).st_rdev != os.fstat(slave_fd).st_rdev:
            raise RuntimeError('fake serial target mismatch')
        port = _free_port()
        uri = 'http://127.0.0.1:{}'.format(port)
        prefix = '/tmp/rosclaw-limo-exclusive-candidate/devel'
        env = {'PATH': '/opt/ros/melodic/bin:/usr/bin:/bin', 'ROS_MASTER_URI': uri,
               'ROS_HOSTNAME': '127.0.0.1', 'ROS_IP': '127.0.0.1',
               'ROS_HOME': str(run_dir / 'ros'), 'ROS_LOG_DIR': str(run_dir / 'logs'),
               'ROS_ROOT': '/opt/ros/melodic/share/ros', 'ROS_DISTRO': 'melodic',
               'ROS_PACKAGE_PATH': '/opt/ros/melodic/share',
               'CMAKE_PREFIX_PATH': prefix + ':/opt/ros/melodic',
               'PYTHONPATH': '/sandbox:/worker:' + prefix + '/lib/python2.7/dist-packages:'
                             '/opt/ros/melodic/lib/python2.7/dist-packages',
               'LD_LIBRARY_PATH': prefix + '/lib:/opt/ros/melodic/lib',
               'PYTHONDONTWRITEBYTECODE': '1'}
        result = None
        try:
            with open(str(run_dir / 'master.log'), 'wb') as master_log, \
                 open(str(run_dir / 'driver.log'), 'wb') as driver_log, \
                 open(str(run_dir / 'monitor.log'), 'wb') as monitor_log, \
                 open(str(run_dir / 'child.log'), 'wb') as child_log:
                master = subprocess.Popen(['/opt/ros/melodic/bin/roscore', '-p', str(port)],
                                          env=env, stdout=master_log, stderr=master_log,
                                          start_new_session=True)
                _wait_master(uri, master)
                fake.thread.start()
                fake_port = os.path.relpath(str(link), '/dev')
                driver = subprocess.Popen([
                    str(candidate), '__name:=limo_pty_base', '_port_name:=' + fake_port,
                    '_command_watchdog_enabled:=true', '_command_watchdog_timeout_sec:=0.3',
                    '_command_watchdog_mode_timeout_sec:=1.5', '_pub_odom_tf:=false',
                ], env=env, stdout=driver_log, stderr=driver_log, start_new_session=True)
                if not fake.configured.wait(6.0):
                    raise RuntimeError('candidate did not initialize the fake PTY')
                monitor = subprocess.Popen(['/usr/bin/python2.7', '-B', '-u', '-c', STATUS_MONITOR],
                                           env=env, stdout=subprocess.PIPE, stderr=monitor_log,
                                           start_new_session=True)
                collector = StatusCollector(monitor.stdout)
                collector.thread.start()
                if not collector.ready.wait(5.0):
                    raise RuntimeError('fake driver watchdog monitor not ready')
                initial = wait_for(lambda: collector.snapshot(), 4.0, 'initial watchdog report')[-1][1]
                if initial[5] != '0':
                    raise RuntimeError('driver watchdog already latched')
                started = time.monotonic()
                child = subprocess.Popen(['/usr/bin/python2.7', '-B', '-u', '-c', CHILD],
                                         env=env, stdout=subprocess.PIPE, stderr=child_log,
                                         start_new_session=True)
                stdout, _ = child.communicate(timeout=15.0)
                returned = time.monotonic()
                if child.returncode != 0:
                    raise RuntimeError('isolated LiveTrial process failed: {}'.format(child.returncode))
                result = json.loads(stdout.decode('utf-8'))
                trial, writes = result['trial'], result['writes']
                if (result['model_calls'] != 1 or trial.get('motion_requested') is not True
                        or trial.get('stop_requested') is not True
                        or trial.get('physical_stop_verified') is not False
                        or trial.get('reason') not in ('motion_limit', 'time_limit', 'relay_failure',
                                                       'sensor_unavailable', 'expiry_unavailable')
                        or trial.get('relay_evidence', {}).get('process_exited') is not True
                        or not result['fake_arbiter_released'] or not result['fake_serial_released']):
                    raise RuntimeError('unexpected LiveTrial result: {}'.format(json.dumps(result, sort_keys=True)))
                motion_writes = [index for index, write in enumerate(writes)
                                 if write['frame']['linear'] > 0]
                if (not motion_writes or motion_writes[0] < MIN_PRE_MOTION_ZERO_WRITES
                        or any(write['frame']['linear'] != 0 for write in writes[:motion_writes[0]])
                        or any(write['frame']['linear'] != 0.02 or write['frame']['angular'] != 0
                               for write in writes if write['frame']['linear'] > 0)
                        or writes[-1]['frame']['linear'] != 0):
                    raise RuntimeError('trial did not write the expected zero/motion/zero sequence')
                stopped = [item for item in result['status_reads']
                           if item['line'] in ('STOP motion_deadline\n', 'STOP requested_stop\n',
                                               'FAULT intent_deadline\n')]
                if not stopped:
                    raise RuntimeError('relay did not report a terminal STOP line')
                stop_at = stopped[0]['at']
                first_write_at = writes[motion_writes[0]]['frame']['stamp']
                time.sleep(1.0)
                frames = fake.snapshot()
                summary = summarize_frames(frames, started, returned, stop_at, first_write_at, MOTION_LIMIT_SEC)
                before_motion = [fields for _, fields in collector.snapshot()
                                 if fields[3:6] == ['0', '1', '0']
                                 and fields[6].startswith('/standalone_limo_relay_')
                                 and float(fields[8]) <= summary['first_nonzero_monotonic_s']]
                counts = [int(fields[2]) for fields in before_motion]
                if len(counts) < 2 or max(counts) <= min(counts) or fake.errors or collector.errors:
                    raise RuntimeError('fresh fake driver zero prime not independently observed')
                if any(hashlib.sha256(path.read_bytes()).hexdigest() != digests[path.name] for path in artifacts):
                    raise RuntimeError('test artifact changed while running')
                summary.update({'result': 'passed', 'simulated_serial_only': True,
                                'physical_stop_verified': False, 'physical_device_visible': False,
                                'network_interfaces': ['lo'], 'fake_source': True, 'fake_model': True,
                                'fake_ros_guard': True, 'fake_serial_guard': True,
                                'fake_watchdog_zero_count_min': min(counts),
                                'fake_watchdog_zero_count_max': max(counts),
                                'driver_final_watchdog_latched': collector.snapshot()[-1][1][5] == '1',
                                'watchdog_latch_not_required_after_clean_zero': True,
                                'trial_reason': trial['reason'], 'relay_evidence': trial['relay_evidence'],
                                'stop_line_observed_monotonic_s': stop_at,
                                'pipe_zero_heartbeats_before_motion': motion_writes[0],
                                'pipe_nonzero_writes': len(motion_writes),
                                'pipe_writes_are_not_driver_acknowledgments': True,
                                'relay_internal_deadline_time_not_directly_observed': True,
                                'digests_sha256': digests})
                print(json.dumps(summary, sort_keys=True))
        except Exception as error:
            details = {'result': 'failed', 'error': str(error), 'trial_result': result,
                       'frames': fake.snapshot()[-20:]}
            for name in ('master.log', 'driver.log', 'monitor.log', 'child.log'):
                path = run_dir / name
                if path.is_file():
                    details[name] = path.read_bytes()[-2000:].decode('utf-8', 'replace')
            print(json.dumps(details, sort_keys=True), file=sys.stderr)
            raise
        finally:
            fake.stopping.set()
            _terminate(child)
            _terminate(monitor)
            _terminate(driver)
            _terminate(master)
            if fake.thread.is_alive():
                fake.thread.join(timeout=1.0)
            if collector is not None:
                collector.thread.join(timeout=1.0)
            os.close(master_fd)
            os.close(slave_fd)


class FrameSummaryTests(unittest.TestCase):
    def good_frames(self):
        return [(9.9, ZERO), (10.0, MOTION), (10.05, MOTION), (10.25, MOTION)] + [
            (10.4 + 0.05 * index, ZERO) for index in range(12)]

    def test_good_limited_motion_then_only_zero(self):
        result = summarize_frames(self.good_frames(), 9.8, 10.38)
        self.assertEqual(result['forward_frames'], 3)
        self.assertEqual(result['post_stop_nonzero_frames'], 0)
        self.assertEqual(result['post_return_nonzero_frames'], 0)

    def test_rejects_missing_motion_or_zero_tail(self):
        for frames in ([(10.0, ZERO), (10.5, ZERO)], [(10.0, MOTION)]):
            with self.assertRaises(ValueError):
                summarize_frames(frames, 9.8, 10.38)

    def test_rejects_wrong_speed_and_motion_after_stop(self):
        for bad in ((10.1, (21, 0, 0, 0)), (10.5, MOTION)):
            frames = self.good_frames() + [bad]
            with self.assertRaises(ValueError):
                summarize_frames(frames, 9.8, 10.38)

    def test_reports_serial_tail_after_input_deadline_without_hiding_it(self):
        frames = self.good_frames() + [(10.31, MOTION)]
        frames.sort()
        try:
            result = summarize_frames(frames, 9.8, 10.38, stop_at=10.37, first_write_at=10.0)
        except ValueError:
            self.fail('serial propagation tail should be reported, not discarded')
        self.assertEqual(result['nonzero_after_first_pipe_stamp_plus_0_3'], 1)
        self.assertEqual(result['post_stop_nonzero_frames'], 0)

    def test_idle_after_clean_exit_is_not_confused_with_nonzero_output(self):
        try:
            result = summarize_frames(self.good_frames(), 9.8, 11.5, stop_at=10.38, first_write_at=10.0)
        except ValueError:
            self.fail('clean serial idle is allowed after prior post-STOP zero frames')
        self.assertEqual(result['post_return_zero_frames'], 0)
        self.assertTrue(result['post_return_serial_idle'])

    def test_explicit_two_second_bound_accepts_longer_simulated_motion(self):
        frames = [(10.0, MOTION), (11.5, MOTION)] + [(12.1 + .05 * i, ZERO) for i in range(8)]
        result = summarize_frames(frames, 9.9, 12.09, stop_at=12.09, first_write_at=10.0,
                                  motion_limit_sec=2.0)
        self.assertEqual(result['nonzero_after_motion_limit'], 0)
        self.assertEqual(result['motion_limit_sec'], 2.0)


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        unittest.main(argv=[sys.argv[0]])
    else:
        main()
