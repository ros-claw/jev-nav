#!/usr/bin/env python3
"""Check the real candidate driver and standalone relay using an isolated PTY.

Run only in a network, PID, and device isolated bubblewrap namespace.  The
fake chassis emits modeled status frames; it is not the real controller or
firmware, and the results do not establish a physical stop.
"""

import collections
import fcntl
import json
import os
import pty
import select
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time
import tty
from pathlib import Path

from driver_pty_check import _free_port, _terminate, _wait_master

STATUS_MONITOR = r'''
import sys
import rospy
from std_msgs.msg import String
rospy.init_node("standalone_pty_monitor", anonymous=False, disable_signals=True)
def on_status(message):
    sys.stdout.write("STATUS " + message.data + "\n")
    sys.stdout.flush()
subscriber = rospy.Subscriber("/rosclaw_limo/command_watchdog", String, on_status, queue_size=20)
sys.stdout.write("MONITOR_READY\n")
sys.stdout.flush()
while not rospy.is_shutdown():
    rospy.rostime.wallsleep(0.05)
'''

REPLAY_OLD_COMMAND = r'''
import sys, time
import rospy
from geometry_msgs.msg import Twist
expected = sys.argv[1]
rospy.init_node(expected.lstrip("/"), anonymous=False, disable_signals=True)
publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
deadline = time.time() + 4.0
while publisher.get_num_connections() == 0 and time.time() < deadline:
    time.sleep(0.02)
if publisher.get_num_connections() == 0 or rospy.get_name() != expected:
    raise RuntimeError("old caller could not reconnect to candidate driver")
message = Twist()
message.linear.x = 0.05
for _ in range(30):
    publisher.publish(message)
    time.sleep(0.05)
sys.stdout.write("REPLAYED " + rospy.get_name() + " 30\n")
sys.stdout.flush()
publisher.unregister()
rospy.signal_shutdown("fake replay complete")
'''


def check_isolation():
    if (socket.gethostname() != 'limo-pty-isolated' or os.getpid() != 2 or
            os.getppid() != 1 or os.path.lexists('/dev/ttyTHS1') or
            set(name for _, name in socket.if_nameindex()) != set(['lo'])):
        raise RuntimeError('refusing to run outside the isolated PTY namespace')


def wait_for(predicate, timeout, description):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    raise RuntimeError('timed out waiting for ' + description)


def read_line(process, timeout):
    fd = process.stdout.fileno()
    pending = b''
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('process exited before readiness: {}'.format(process.returncode))
        ready, _, _ = select.select([fd], [], [], min(0.05, deadline - time.monotonic()))
        if ready:
            chunk = os.read(fd, 512)
            if not chunk:
                raise RuntimeError('process closed readiness pipe')
            pending += chunk
            if len(pending) > 2048:
                raise RuntimeError('unexpectedly long readiness output')
            if b'\n' in pending:
                line, remainder = pending.split(b'\n', 1)
                if remainder:
                    raise RuntimeError('unexpected extra readiness output')
                return line.decode('ascii')
    raise RuntimeError('timed out waiting for readiness')


class FakeChassis(object):
    def __init__(self, master_fd):
        self.fd = master_fd
        self.frames = collections.deque(maxlen=4096)
        self.errors = []
        self.lock = threading.Lock()
        self.configured = threading.Event()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def snapshot(self):
        with self.lock:
            return list(self.frames)

    def run(self):
        pending = b''
        last_status = 0.0
        data = bytes([0, 1, 0, 122, 0, 0, 0, 0])
        status = bytes([0x55, 0x0e, 0x02, 0x11]) + data + bytes([0, sum(data) & 0xff])
        while not self.stopping.is_set():
            ready, _, _ = select.select([self.fd], [], [], 0.02)
            if ready:
                try:
                    chunk = os.read(self.fd, 4096)
                except OSError as exc:
                    if not self.stopping.is_set():
                        self.errors.append(str(exc))
                    break
                pending += chunk
                while len(pending) >= 14:
                    if pending[0] != 0x55 or pending[1] != 0x0e:
                        pending = pending[1:]
                        continue
                    frame, pending = pending[:14], pending[14:]
                    if frame[13] != (sum(frame[4:12]) & 0xff):
                        self.errors.append('invalid outgoing serial checksum')
                        continue
                    identifier = frame[2] * 256 + frame[3]
                    if identifier == 0x421:
                        self.configured.set()
                    elif identifier == 0x111:
                        with self.lock:
                            self.frames.append((time.monotonic(), struct.unpack('>hhhh', frame[4:12])))
            now = time.monotonic()
            if self.configured.is_set() and now - last_status >= 0.501:
                try:
                    os.write(self.fd, status)
                except OSError as exc:
                    if not self.stopping.is_set():
                        self.errors.append(str(exc))
                    break
                last_status = now


class StatusCollector(object):
    def __init__(self, stream):
        self.stream = stream
        self.statuses = collections.deque(maxlen=2048)
        self.errors = []
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def snapshot(self):
        with self.lock:
            return list(self.statuses)

    def run(self):
        for raw in self.stream:
            line = raw.decode('ascii', 'replace').strip()
            if line == 'MONITOR_READY':
                self.ready.set()
            elif line.startswith('STATUS '):
                fields = line.split()[1:]
                if len(fields) == 9 and fields[0] == 'v2':
                    with self.lock:
                        self.statuses.append((time.monotonic(), fields))
                else:
                    self.errors.append('malformed candidate status')


def sender_loop(process, stopping, errors):
    fd = process.stdin.fileno()
    fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
    sequence = 0
    while not stopping.is_set():
        sequence += 1
        data = (json.dumps({'sequence': sequence, 'stamp': time.monotonic(),
                            'linear': 0.05, 'angular': 0.0}) + '\n').encode('ascii')
        try:
            if os.write(fd, data) != len(data):
                raise RuntimeError('short relay command write')
        except (OSError, RuntimeError) as exc:
            if not stopping.is_set():
                errors.append(str(exc))
            return
        if stopping.wait(0.05):
            return


def main():
    check_isolation()
    candidate = Path('/tmp/rosclaw-limo-base-check/devel/lib/limo_base/limo_base_node')
    relay_script = Path(__file__).with_name('relay.py')
    if not candidate.is_file() or not relay_script.is_file():
        raise RuntimeError('candidate driver binary or relay is missing')
    probe = socket.socket()
    probe.settimeout(0.2)
    try:
        if probe.connect_ex(('127.0.0.1', 11311)) == 0:
            raise RuntimeError('unexpected existing default ROS master')
    finally:
        probe.close()

    master_fd, slave_fd = pty.openpty()
    tty.setraw(slave_fd)
    pty_name = os.ttyname(slave_fd)
    if not pty_name.startswith('/dev/pts/'):
        raise RuntimeError('serial device is not a pseudoterminal')
    master = driver = monitor = relay = replay = None
    sender_stop = threading.Event()
    sender = None
    collector = None
    fake = FakeChassis(master_fd)
    with tempfile.TemporaryDirectory(prefix='limo-relay-pty-') as temporary:
        run_dir = Path(temporary)
        link = run_dir / 'ttyFake'
        link.symlink_to(pty_name)
        if os.stat(str(link)).st_rdev != os.fstat(slave_fd).st_rdev:
            raise RuntimeError('fake serial target mismatch')
        port = _free_port()
        uri = 'http://127.0.0.1:{}'.format(port)
        # Whitelist only ROS runtime variables; do not propagate ambient keys.
        env = {'PATH': '/opt/ros/melodic/bin:/usr/bin:/bin',
               'ROS_MASTER_URI': uri, 'ROS_HOSTNAME': '127.0.0.1', 'ROS_IP': '127.0.0.1',
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
                    raise RuntimeError('candidate did not initialize fake serial port')
                monitor = subprocess.Popen(['/usr/bin/python2.7', '-B', '-u', '-c', STATUS_MONITOR],
                                           env=env, stdout=subprocess.PIPE, stderr=monitor_log,
                                           start_new_session=True)
                collector = StatusCollector(monitor.stdout)
                collector.thread.start()
                if not collector.ready.wait(5.0):
                    raise RuntimeError('watchdog status monitor did not initialize')
                initial_statuses = wait_for(lambda: collector.snapshot(), 4.0,
                                            'initial candidate driver status')
                baseline_status = initial_statuses[-1][1]
                if baseline_status[5] != '0':
                    raise RuntimeError('candidate watchdog was already latched before relay start')
                baseline_zero_count = int(baseline_status[2])
                relay_started = time.monotonic()
                relay = subprocess.Popen(['/usr/bin/python2.7', '-B', '-u', str(relay_script)],
                                         env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=relay_log, start_new_session=True)
                ready_line = read_line(relay, 8.0)
                ready_at = time.monotonic()
                if ready_line != 'READY':
                    raise RuntimeError('relay did not become ready: ' + ready_line)
                sender_errors = []
                sender = threading.Thread(target=sender_loop, args=(relay, sender_stop, sender_errors),
                                          daemon=True)
                sender.start()

                def primed_and_moving():
                    statuses = [item for item in collector.snapshot()
                                if item[1][3:6] == ['0', '1', '0'] and
                                item[1][6].startswith('/standalone_limo_relay_') and
                                int(item[1][2]) > baseline_zero_count and
                                float(item[1][7]) >= relay_started and
                                float(item[1][8]) <= ready_at]
                    motions = [item for item in fake.snapshot()
                               if item[0] >= relay_started and item[1] == (50, 0, 0, 0)]
                    zeros = [item for item in fake.snapshot()
                             if relay_started <= item[0] <= ready_at and item[1] == (0, 0, 0, 0)]
                    if statuses and len(motions) >= 4 and zeros:
                        return statuses[-1][1]
                    return None

                primed_status = wait_for(primed_and_moving, 4.0, 'relay prime and forward frames')
                caller = primed_status[6]
                boot_id = primed_status[1]
                if sender_errors or relay.poll() is not None or driver.poll() is not None:
                    raise RuntimeError('relay/driver failed while forwarding')
                wait_for(lambda: any(time.monotonic() - stamp < 0.12 and frame == (50, 0, 0, 0)
                                     for stamp, frame in fake.snapshot()), 1.0,
                         'recent motion immediately before relay death')
                kill_at = time.monotonic()
                if any(fields[5] == '1' and float(fields[8]) < kill_at
                       for _, fields in collector.snapshot() if fields[1] == boot_id):
                    raise RuntimeError('driver had latched before relay death')
                sender_stop.set()
                relay.kill()  # SIGKILL: no relay shutdown or zero burst runs.
                relay.wait(timeout=3.0)
                if relay.returncode != -signal.SIGKILL:
                    raise RuntimeError('relay was not killed by SIGKILL')
                sender.join(timeout=1.0)

                def latched_after_kill():
                    matches = [item for item in collector.snapshot()
                               if item[0] >= kill_at and item[1][1] == boot_id and
                               item[1][5] == '1' and float(item[1][8]) >= kill_at]
                    return matches[-1] if matches else None

                latched_at, latched_status = wait_for(latched_after_kill, 3.0, 'driver watchdog latch')
                if latched_status[6] != caller or latched_status[3:5] != ['0', '1']:
                    raise RuntimeError('latched status changed command source or lost fresh mode')
                wait_for(lambda: len([item for item in fake.snapshot() if item[0] >= latched_at and
                                      item[1] == (0, 0, 0, 0)]) >= 4, 2.0,
                         'repeated watchdog zero serial frames')
                replay_start = time.monotonic()
                driver_log_offset = os.stat(str(run_dir / 'driver.log')).st_size
                replay = subprocess.Popen(['/usr/bin/python2.7', '-B', '-u', '-c',
                                           REPLAY_OLD_COMMAND, caller], env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          start_new_session=True)
                out, err = replay.communicate(timeout=8.0)
                replay_end = time.monotonic()
                expected = 'REPLAYED {} 30'.format(caller)
                if replay.returncode != 0 or expected not in out.decode('utf-8', 'replace'):
                    raise RuntimeError('same-caller replay failed: ' + err.decode('utf-8', 'replace')[-800:])
                time.sleep(0.2)
                with open(str(run_dir / 'driver.log'), 'rb') as check_log:
                    check_log.seek(driver_log_offset)
                    replay_driver_log = check_log.read().decode('utf-8', 'replace')
                rejected = 'LIMO command watchdog rejected another publisher; zero requested'
                if rejected not in replay_driver_log:
                    raise RuntimeError('no driver callback rejection observed during same-caller replay')
                recorded = fake.snapshot()
                post_latch = [frame for stamp, frame in recorded if stamp >= latched_at]
                replay_frames = [frame for stamp, frame in recorded if replay_start <= stamp <= replay_end]
                all_status = collector.snapshot()
                replay_status = [fields for stamp, fields in all_status
                                 if replay_start <= stamp <= replay_end and fields[1] == boot_id]
                before_kill_status = [fields for _, fields in all_status if fields[1] == boot_id and
                                      relay_started <= float(fields[8]) < kill_at]
                if (not before_kill_status or any(fields[5] != '0' for fields in before_kill_status)):
                    raise RuntimeError('driver was already latched before relay death')
                if (fake.errors or collector.errors or sender_errors or len(post_latch) < 4 or
                        any(frame != (0, 0, 0, 0) for frame in post_latch) or
                        len(replay_frames) < 5 or any(frame != (0, 0, 0, 0) for frame in replay_frames) or
                        not replay_status or any(fields[5] != '1' for fields in replay_status)):
                    raise RuntimeError('serial frames or watchdog state did not stay latched at zero')
                evidence = {
                    'result': 'passed', 'simulated_serial_only': True,
                    'physical_serial_visible': False, 'network_interfaces': ['lo'],
                    'fake_status_period_sec': 0.501, 'driver_watchdog_timeout_sec': 0.3,
                    'relay_ready': True, 'relay_caller': caller, 'driver_boot_id': boot_id,
                    'prime_count_increased': int(primed_status[2]) > baseline_zero_count,
                    'mode_fresh_at_latch': True,
                    'prime_zero_frames': len([item for item in recorded if relay_started <= item[0] <= ready_at
                                              and item[1] == (0, 0, 0, 0)]),
                    'forward_frames_before_kill': len([item for item in recorded
                                                       if relay_started <= item[0] < kill_at
                                                       and item[1] == (50, 0, 0, 0)]),
                    'relay_termination': 'SIGKILL', 'watchdog_latched': True,
                    'latch_observed_after_kill_sec': round(latched_at - kill_at, 3),
                    'zero_frames_after_latch': len(post_latch), 'same_caller_replay_sent': 30,
                    'driver_rejected_replayed_commands': True,
                    'replay_zero_frames': len(replay_frames), 'replay_nonzero_frames': 0,
                }
                print(json.dumps(evidence, sort_keys=True))
        except Exception as exc:
            details = []
            for name in ('master.log', 'driver.log', 'monitor.log', 'relay.log'):
                path = run_dir / name
                if path.is_file():
                    content = path.read_bytes()[-1400:].decode('utf-8', 'replace').strip()
                    if content:
                        details.append(name + ': ' + content)
            raise RuntimeError('{}\n{}'.format(exc, '\n'.join(details)))
        finally:
            sender_stop.set()
            if sender is not None:
                sender.join(timeout=1.0)
            fake.stopping.set()
            _terminate(replay)
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


if __name__ == '__main__':
    main()
