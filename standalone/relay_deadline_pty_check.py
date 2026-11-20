#!/usr/bin/env python3
"""Offline relay deadline integration check; never a physical stop test.

Run only under the documented network/PID/device-isolated bubblewrap invocation.
The driver talks to a fake PTY chassis and a private ROS master. Nothing here
proves physical braking, firmware behavior, or motor power removal.

From the repository root, with the existing candidate build present:

    env -i PATH=/usr/bin:/bin:/sbin bwrap --unshare-all \
      --hostname limo-pty-isolated --die-with-parent \
      --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib \
      --ro-bind /sbin /sbin --ro-bind /opt/ros/melodic /opt/ros/melodic \
      --proc /proc --dev /dev --tmpfs /tmp \
      --ro-bind /tmp/rosclaw-limo-base-check /tmp/rosclaw-limo-base-check \
      --ro-bind "$PWD/standalone" /sandbox \
      --dir /etc --ro-bind /etc/hosts /etc/hosts \
      --ro-bind /etc/nsswitch.conf /etc/nsswitch.conf \
      --ro-bind /etc/passwd /etc/passwd --ro-bind /etc/group /etc/group \
      --chdir /tmp --setenv ROS_HOME /tmp/ros-home --setenv PYTHONDONTWRITEBYTECODE 1 \
      /usr/bin/python3 /sandbox/relay_deadline_pty_check.py

The synthetic frame-summary checks can also run without ROS or hardware:
``PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 standalone/relay_deadline_pty_check.py --self-test``.
"""

import fcntl
import hashlib
import json
import os
import pty
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tty
import unittest
from pathlib import Path

from driver_pty_check import _free_port, _terminate, _wait_master
from relay_pty_check import (
    STATUS_MONITOR,
    FakeChassis,
    StatusCollector,
    check_isolation,
    read_line,
    wait_for,
)

ZERO = (0, 0, 0, 0)
MOTION = (20, 0, 0, 0)  # fake protocol: millimeters per second


def summarize_frames(frames, started_at, stop_at, writes_after_stop):
    relevant = [(stamp, frame) for stamp, frame in frames if stamp >= started_at]
    motion = [(stamp, frame) for stamp, frame in relevant if frame != ZERO]
    after_stop = [(stamp, frame) for stamp, frame in relevant if stamp >= stop_at]
    if (not motion or any(frame != MOTION for _, frame in motion) or len(after_stop) < 4
            or any(frame != ZERO for _, frame in after_stop) or writes_after_stop < 5):
        raise ValueError('motion, post-stop zeros, or continued input was not verified')
    last_nonzero_at = motion[-1][0]
    zeros_after_motion = [stamp for stamp, frame in relevant if stamp > last_nonzero_at and frame == ZERO]
    if not zeros_after_motion:
        raise ValueError('no serial zero after the last nonzero')
    first_zero_at = zeros_after_motion[0]
    return {
        'forward_frames': len(motion), 'post_stop_zero_frames': len(after_stop),
        'post_stop_nonzero_frames': 0, 'heartbeat_writes_after_stop': writes_after_stop,
        'last_nonzero_monotonic_s': last_nonzero_at,
        'first_zero_after_last_nonzero_monotonic_s': first_zero_at,
        'zero_after_last_nonzero_sec': round(first_zero_at - last_nonzero_at, 6),
        'last_nonzero_minus_stop_report_sec': round(last_nonzero_at - stop_at, 6),
        'first_zero_minus_stop_report_sec': round(first_zero_at - stop_at, 6),
    }


def _sender(relay, stopping, writes, errors):
    fd = relay.stdin.fileno()
    fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
    sequence = 0
    while not stopping.is_set():
        sequence += 1
        stamp = time.monotonic()
        data = (json.dumps({'sequence': sequence, 'stamp': stamp,
                            'linear': 0.02, 'angular': 0.0}) + '\n').encode('ascii')
        try:
            if os.write(fd, data) != len(data):
                raise RuntimeError('short heartbeat write')
            writes.append((time.monotonic(), sequence))
        except (OSError, RuntimeError) as exc:
            if not stopping.is_set():
                errors.append(str(exc))
            return
        if stopping.wait(0.025):
            return


def _check_safe_namespace():
    check_isolation()
    allowed = {'core', 'fd', 'full', 'null', 'ptmx', 'pts', 'random', 'shm',
               'stderr', 'stdin', 'stdout', 'tty', 'urandom', 'zero'}
    if set(os.listdir('/dev')) - allowed or set(os.listdir('/dev/pts')) != {'ptmx'}:
        raise RuntimeError('unexpected devices or existing PTYs visible; refusing to run')
    if set(name for _, name in socket.if_nameindex()) != {'lo'}:
        raise RuntimeError('network namespace is not loopback-only')


def main():
    _check_safe_namespace()
    candidate = Path('/tmp/rosclaw-limo-base-check/devel/lib/limo_base/limo_base_node')
    relay_script = Path(__file__).with_name('relay.py')
    if not candidate.is_file() or not relay_script.is_file():
        raise RuntimeError('isolated candidate driver or relay script is missing')
    binary_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    relay_digest = hashlib.sha256(relay_script.read_bytes()).hexdigest()
    master_fd, slave_fd = pty.openpty()
    tty.setraw(slave_fd)
    pty_name = os.ttyname(slave_fd)
    if not pty_name.startswith('/dev/pts/'):
        raise RuntimeError('serial path is not an isolated pseudoterminal')
    master = driver = monitor = relay = None
    sender = collector = None
    sender_stop = threading.Event()
    writes, sender_errors = [], []
    fake = FakeChassis(master_fd)
    stop_at = started_at = None
    with tempfile.TemporaryDirectory(prefix='relay-deadline-pty-') as temporary:
        run_dir = Path(temporary)
        link = run_dir / 'ttyFake'
        link.symlink_to(pty_name)
        if os.stat(str(link)).st_rdev != os.fstat(slave_fd).st_rdev:
            raise RuntimeError('PTY serial target mismatch')
        port = _free_port()
        uri = 'http://127.0.0.1:{}'.format(port)
        env = {'PATH': '/opt/ros/melodic/bin:/usr/bin:/bin', 'ROS_MASTER_URI': uri,
               'ROS_HOSTNAME': '127.0.0.1', 'ROS_IP': '127.0.0.1',
               'ROS_HOME': str(run_dir / 'ros'), 'ROS_LOG_DIR': str(run_dir / 'logs'),
               'ROS_ROOT': '/opt/ros/melodic/share/ros', 'ROS_DISTRO': 'melodic',
               'ROS_PACKAGE_PATH': '/opt/ros/melodic/share',
               'CMAKE_PREFIX_PATH': '/tmp/rosclaw-limo-base-check/devel:/opt/ros/melodic',
               'PYTHONPATH': '/tmp/rosclaw-limo-base-check/devel/lib/python2.7/dist-packages:'
                             '/opt/ros/melodic/lib/python2.7/dist-packages',
               'LD_LIBRARY_PATH': '/tmp/rosclaw-limo-base-check/devel/lib:/opt/ros/melodic/lib',
               'PYTHONDONTWRITEBYTECODE': '1'}
        try:
            with open(str(run_dir / 'master.log'), 'wb') as master_log, \
                 open(str(run_dir / 'driver.log'), 'wb') as driver_log, \
                 open(str(run_dir / 'monitor.log'), 'wb') as monitor_log, \
                 open(str(run_dir / 'relay.log'), 'wb') as relay_log:
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
                    raise RuntimeError('candidate failed to initialize fake PTY')
                monitor = subprocess.Popen(['/usr/bin/python2.7', '-B', '-u', '-c', STATUS_MONITOR],
                                           env=env, stdout=subprocess.PIPE, stderr=monitor_log,
                                           start_new_session=True)
                collector = StatusCollector(monitor.stdout)
                collector.thread.start()
                if not collector.ready.wait(5.0):
                    raise RuntimeError('fake driver monitor was not ready')
                initial = wait_for(lambda: collector.snapshot(), 4.0, 'initial watchdog report')[-1][1]
                if initial[5] != '0':
                    raise RuntimeError('fake driver watchdog was already latched')
                relay_started_at = time.monotonic()
                relay = subprocess.Popen(['/usr/bin/python2.7', '-B', '-u', str(relay_script),
                                          '--max-motion-sec', '0.3'], env=env,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=relay_log, start_new_session=True)
                ready = read_line(relay, 8.0)
                ready_at = time.monotonic()
                if ready != 'READY motion_limit_sec=0.300000':
                    raise RuntimeError('unexpected relay ready line: ' + ready)
                started_at = time.monotonic()
                sender = threading.Thread(target=_sender, args=(relay, sender_stop, writes, sender_errors),
                                          daemon=True)
                sender.start()
                stop = read_line(relay, 4.0)
                stop_at = time.monotonic()
                if stop != 'STOP motion_deadline':
                    raise RuntimeError('unexpected relay stop line: ' + stop)
                time.sleep(0.8)  # Keep writing fresh nonzero frames after the latch.
                if relay.poll() is not None or driver.poll() is not None or sender_errors:
                    raise RuntimeError('relay, driver, or continued sender ended prematurely')
                sender_stop.set()
                sender.join(timeout=1.0)
                if sender.is_alive():
                    raise RuntimeError('heartbeat sender did not stop')
                post_writes = sum(when >= stop_at for when, _ in writes)
                recorded = fake.snapshot()
                summary = summarize_frames(recorded, started_at, stop_at, post_writes)
                if summary['forward_frames'] < 2 or fake.errors or collector.errors:
                    raise RuntimeError('fake serial or watchdog status evidence was incomplete')
                statuses = [fields for _, fields in collector.snapshot()
                            if fields[3:6] == ['0', '1', '0'] and
                            fields[6].startswith('/standalone_limo_relay_') and
                            int(fields[2]) > int(initial[2]) and
                            float(fields[7]) >= relay_started_at and float(fields[8]) <= ready_at]
                if not statuses:
                    raise RuntimeError('relay-ready status did not prove a fresh fake driver zero')
                if hashlib.sha256(candidate.read_bytes()).hexdigest() != binary_digest:
                    raise RuntimeError('candidate binary changed during the test')
                summary.update({
                    'result': 'passed', 'simulated_serial_only': True,
                    'physical_stop_verified': False, 'physical_device_visible': False,
                    'network_interfaces': ['lo'], 'relay_ready': ready, 'relay_stop': stop,
                    'relay_sha256': relay_digest, 'candidate_sha256': binary_digest,
                    'stop_report_monotonic_s': stop_at,
                    'first_heartbeat_monotonic_s': writes[0][0],
                    'stop_report_after_first_heartbeat_sec': round(stop_at - writes[0][0], 6),
                    'post_stop_observation_sec': round(time.monotonic() - stop_at, 6),
                    'fake_watchdog_zero_primed': True,
                    'heartbeats_are_pipe_writes_not_individual_acknowledgments': True,
                })
                print(json.dumps(summary, sort_keys=True))
        except Exception as exc:
            recorded = fake.snapshot()
            nonzero = [(stamp, frame) for stamp, frame in recorded if frame != ZERO]
            details = {'result': 'failed', 'error': str(exc), 'stop_report_monotonic_s': stop_at,
                       'last_nonzero': nonzero[-1] if nonzero else None,
                       'post_stop_nonzero': [item for item in nonzero
                                             if stop_at is not None and item[0] >= stop_at],
                       'post_stop_writes': sum(when >= stop_at for when, _ in writes)
                       if stop_at is not None else 0}
            for name in ('master.log', 'driver.log', 'monitor.log', 'relay.log'):
                path = run_dir / name
                if path.is_file():
                    details[name] = path.read_bytes()[-1000:].decode('utf-8', 'replace')
            print(json.dumps(details, sort_keys=True), file=sys.stderr)
            raise
        finally:
            sender_stop.set()
            if sender is not None:
                sender.join(timeout=1.0)
            fake.stopping.set()
            _terminate(relay)
            _terminate(monitor)
            _terminate(driver)
            _terminate(master)
            if fake.thread.is_alive():
                fake.thread.join(timeout=1.0)
            if collector is not None:
                collector.thread.join(timeout=1.0)
            os.close(master_fd)
            os.close(slave_fd)


class SummaryTests(unittest.TestCase):
    def sample(self):
        return [(10.05, (0, 0, 0, 0)), (10.10, (20, 0, 0, 0))] + [
            (10.31 + 0.05 * index, (0, 0, 0, 0)) for index in range(8)]

    def test_records_actual_serial_transition(self):
        result = summarize_frames(self.sample(), 10.0, 10.3, 10)
        self.assertEqual(result.get('last_nonzero_monotonic_s'), 10.10)
        self.assertEqual(result.get('first_zero_after_last_nonzero_monotonic_s'), 10.31)
        self.assertEqual(result.get('post_stop_nonzero_frames'), 0)

    def test_rejects_nonzero_serial_after_stop(self):
        frames = self.sample() + [(10.8, (20, 0, 0, 0))]
        with self.assertRaises(ValueError):
            summarize_frames(frames, 10.0, 10.3, 10)

    def test_requires_motion_zero_frames_and_continued_heartbeats(self):
        for frames, writes in (([(10.4, (0, 0, 0, 0))] * 8, 10),
                               ([(10.1, (20, 0, 0, 0))], 10),
                               (self.sample(), 0)):
            with self.assertRaises(ValueError):
                summarize_frames(frames, 10.0, 10.3, writes)


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        unittest.main(argv=[sys.argv[0]])
    else:
        main()
