#!/usr/bin/env python
from __future__ import print_function

import os
import select
import signal
import subprocess
import sys
import time
import unittest

relay_path = os.path.join(os.path.dirname(__file__), 'relay.py')
try:
    import imp
    relay = imp.load_source('standalone_relay', relay_path)
except ImportError:  # Python 3.12 removed imp; ROS Melodic still uses Python 2.
    import importlib.util
    spec = importlib.util.spec_from_file_location('standalone_relay', relay_path)
    relay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(relay)


class GateTests(unittest.TestCase):
    def test_navigation_can_pause_and_resume_until_a_controller_disconnect(self):
        gate = relay.RelayGate(10.0, forward_navigation=True)
        for sequence, linear in enumerate((0.0, 0.014, 0.0, 0.014, 0.02)):
            now = 10.0 + sequence * .05
            self.assertTrue(gate.accept({'sequence': sequence, 'stamp': now, 'linear': linear,
                                         'angular': 0.0}, now))
            self.assertEqual(gate.command(now), (linear, 0.0))
            self.assertFalse(gate.stopped)
        self.assertEqual(gate.command(10.41), (0.0, 0.0))
        self.assertEqual(gate.reason, 'heartbeat')
        self.assertFalse(gate.accept({'sequence': 6, 'stamp': 10.42, 'linear': 0.014, 'angular': 0.0}, 10.42))

    def test_navigation_has_a_separate_exact_forward_limit(self):
        cases = [('0.02000000000000000001', '0'), ('0.014', '1e-1000'), ('-0.001', '0')]
        for linear, angular in cases:
            text = ('{"sequence":0,"stamp":10,"linear":%s,"angular":%s}\n' % (linear, angular)).encode('ascii')
            gate = relay.RelayGate(10.0, forward_navigation=True)
            self.assertFalse(gate.accept(relay.FrameReader().feed(text)[0], 10.0))
            self.assertEqual(gate.reason, 'invalid_frame')

    def test_required_intent_deadline_expires_before_heartbeat_and_latches(self):
        gate = relay.RelayGate(10.0, require_intent_deadline=True)
        frame = {'sequence': 0, 'stamp': 10.0, 'linear': 0.02, 'angular': 0.0,
                 'expires_at': 10.06}
        self.assertTrue(gate.accept(frame, 10.0))
        self.assertEqual(gate.command(10.05), (0.02, 0.0))
        self.assertEqual(gate.command(10.06), (0.0, 0.0))
        self.assertEqual(gate.reason, 'intent_deadline')
        self.assertFalse(gate.accept(dict(frame, sequence=1, stamp=10.07, expires_at=10.15), 10.07))

    def test_required_intent_deadline_rejects_missing_stale_or_unbounded_values(self):
        frame = {'sequence': 0, 'stamp': 10.0, 'linear': 0.02, 'angular': 0.0}
        for expiry in (None, 9.99, 10.0, 10.201, float('nan'), True):
            gate = relay.RelayGate(10.0, require_intent_deadline=True)
            value = frame if expiry is None else dict(frame, expires_at=expiry)
            self.assertFalse(gate.accept(value, 10.0))
            self.assertEqual(gate.command(10.01), (0.0, 0.0))

    def test_independent_motion_deadline_cannot_be_extended_by_heartbeats(self):
        gate = relay.RelayGate(10.0, max_motion_sec=0.4)
        for sequence, stamp in enumerate((10.05, 10.15, 10.25, 10.35, 10.44)):
            frame = {'sequence': sequence, 'stamp': stamp, 'linear': 0.02, 'angular': 0.0}
            self.assertTrue(gate.accept(frame, stamp))
        self.assertEqual(gate.command(10.451), (0.0, 0.0))
        self.assertEqual(gate.reason, 'motion_deadline')
        self.assertFalse(gate.accept({'sequence': 6, 'stamp': 10.46, 'linear': 0.02, 'angular': 0.0}, 10.46))

    def test_invalid_independent_motion_deadline_is_rejected(self):
        for value in (0.0, -1.0, 31.0, float('nan'), True):
            with self.assertRaises(ValueError):
                relay.RelayGate(10.0, max_motion_sec=value)
    def test_fresh_motion_expires_and_cannot_resume(self):
        self.assertIsNotNone(relay, 'relay implementation is required')
        gate = relay.RelayGate(10.0)
        frame = {'sequence': 0, 'stamp': 10.0, 'linear': 0.04, 'angular': 0.0}
        self.assertTrue(gate.accept(frame, 10.0))
        self.assertEqual(gate.command(10.1), (0.04, 0.0))
        self.assertEqual(gate.command(10.21), (0.0, 0.0))
        self.assertFalse(gate.accept(dict(frame, sequence=1, stamp=10.22), 10.22))
        self.assertEqual(gate.command(10.22), (0.0, 0.0))


    def test_rejects_invalid_frames_and_latches(self):
        valid = {'sequence': 1, 'stamp': 10.0, 'linear': 0.05, 'angular': 0.0}
        bad_frames = [dict(valid, sequence=True), dict(valid, sequence=-1),
                      dict(valid, sequence=9223372036854775808),
                      dict(valid, linear=0.050001), dict(valid, linear=-0.000001),
                      dict(valid, angular=0.000001), dict(valid, angular=-0.000001),
                      dict(valid, linear=float('nan')), dict(valid, angular=float('inf')),
                      dict(valid, stamp=10.01), dict(valid, stamp=9.7),
                      dict(valid, linear=True), dict(valid, extra=0)]
        bad_frames.append({'sequence': 1, 'stamp': 10.0, 'linear': 0.0})
        for bad in bad_frames:
            gate = relay.RelayGate(10.0)
            self.assertFalse(gate.accept(bad, 10.0), repr(bad))
            self.assertFalse(gate.accept(valid, 10.01), repr(bad))
            self.assertEqual(gate.command(10.01), (0.0, 0.0))

    def test_only_forward_bounded_straight_motion_is_accepted(self):
        frame = {'sequence': 1, 'stamp': 10.0, 'linear': 0.05, 'angular': 0.0}
        for allowed in (dict(frame, linear=0.0), frame, dict(frame, angular=-0.0)):
            gate = relay.RelayGate(10.0)
            self.assertTrue(gate.accept(allowed, 10.0), repr(allowed))
            self.assertEqual(gate.command(10.01), (allowed['linear'], 0.0))
        for rejected in (dict(frame, linear=-0.000001), dict(frame, linear=0.050001),
                         dict(frame, angular=-0.000001), dict(frame, angular=0.000001)):
            gate = relay.RelayGate(10.0)
            self.assertFalse(gate.accept(rejected, 10.0), repr(rejected))
            self.assertEqual(gate.command(10.01), (0.0, 0.0))
            self.assertFalse(gate.accept(dict(frame, sequence=2, stamp=10.02), 10.02))

    def test_json_exact_forward_speed_limit_is_accepted(self):
        for value in ('0.05', '5e-2', '0.05000000000000000000000000'):
            text = ('{"sequence":1,"stamp":10.0,"linear":%s,"angular":0.0}\n' %
                    value).encode('ascii')
            gate = relay.RelayGate(10.0)
            self.assertTrue(gate.accept(relay.FrameReader().feed(text)[0], 10.0), text)
            self.assertEqual(gate.command(10.01), (0.05, 0.0))

    def test_json_precision_cannot_hide_reverse_excess_or_turning(self):
        numbers = [('-1e-1000', '0'), ('0.05000000000000000000000001', '0'),
                   ('0.04', '1e-1000'), ('0.04', '-1e-1000')]
        for linear, angular in numbers:
            text = ('{"sequence":1,"stamp":10.0,"linear":%s,"angular":%s}\n' %
                    (linear, angular)).encode('ascii')
            gate = relay.RelayGate(10.0)
            frame = relay.FrameReader().feed(text)[0]
            self.assertFalse(gate.accept(frame, 10.0), text)
            self.assertEqual(gate.command(10.01), (0.0, 0.0))

    def test_sequence_and_stamp_must_increase(self):
        first = {'sequence': 1, 'stamp': 10.0, 'linear': 0.04, 'angular': 0.0}
        for bad in (dict(first), dict(first, sequence=2),
                    dict(first, sequence=2, stamp=9.99)):
            gate = relay.RelayGate(10.0)
            self.assertTrue(gate.accept(first, 10.0))
            self.assertFalse(gate.accept(bad, 10.01))
            self.assertEqual(gate.command(10.01), (0.0, 0.0))
        gate = relay.RelayGate(10.0)
        self.assertTrue(gate.accept(first, 10.0))
        self.assertTrue(gate.accept(dict(first, sequence=2, stamp=10.1), 10.1))
        self.assertEqual(gate.command(10.19), (0.04, 0.0))

    def test_zero_after_motion_latches_session(self):
        gate = relay.RelayGate(10.0)
        first = {'sequence': 1, 'stamp': 10.0, 'linear': 0.04, 'angular': 0.0}
        self.assertTrue(gate.accept(first, 10.0))
        self.assertTrue(gate.accept(dict(first, sequence=2, stamp=10.1, linear=0), 10.1))
        self.assertEqual(gate.command(10.1), (0.0, 0.0))
        self.assertFalse(gate.accept(dict(first, sequence=3, stamp=10.11), 10.11))


class WorkerProgressTests(unittest.TestCase):
    def test_ros_worker_checks_intent_expiry_at_publication_time(self):
        shared = relay._Shared()
        shared.set_command((0.02, 0.0), 10.0, 10.06)
        self.assertEqual(shared.command_for_publish(lambda: 10.05)[1], (0.02, 0.0))
        fault, command, _ = shared.command_for_publish(lambda: 10.06)
        self.assertEqual(command, (0.0, 0.0))
        self.assertEqual(fault, 'intent_deadline')

    def test_progress_clock_is_sampled_under_same_lock_as_progress(self):
        shared = relay._Shared()
        shared.progressed(10.1)
        method = getattr(shared, 'worker_stalled', None)
        self.assertIsNotNone(method, 'worker progress must be checked atomically')
        def clock():
            acquired = shared.lock.acquire(False)
            if acquired:
                shared.lock.release()
            self.assertFalse(acquired, 'sample time after holding the progress lock')
            return 10.11
        self.assertFalse(method(clock))


class FramingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(getattr(relay, 'FrameReader', None), 'FrameReader is required')
        self.reader = relay.FrameReader()

    def test_reads_partial_and_multiple_frames(self):
        self.assertEqual(self.reader.feed(b'{"sequence":1,'), [])
        self.assertEqual(self.reader.feed(b'"stamp":1,"linear":0,"angular":0}\n{}\n'),
                         [{'sequence': 1, 'stamp': 1, 'linear': 0, 'angular': 0}, {}])

    def test_rejects_oversize_duplicate_invalid_and_nonfinite_json(self):
        bad = [b'x' * 513, b'{"sequence":1,"sequence":2}\n',
               b'{"stamp": NaN}\n', b'\xff\n', b'garbage\n']
        for payload in bad:
            reader = relay.FrameReader()
            with self.assertRaises(ValueError):
                reader.feed(payload)


class DriverStatusTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(getattr(relay, 'DriverStatus', None), 'DriverStatus is required')
        self.status = relay.DriverStatus('/relay')

    def test_ready_requires_a_fresh_new_zero_from_this_relay(self):
        self.status.observe('v2 boot 4 0 1 0 /relay 9.98 10.0', 10.0)
        self.assertFalse(self.status.ready(10.01))
        self.status.observe('v2 boot 5 0 1 0 /other 10.03 10.03', 10.03)
        self.assertFalse(self.status.ready(10.04))
        self.status.observe('v2 boot 6 0 1 0 /relay 10.06 10.06', 10.06)
        self.assertTrue(self.status.ready(10.07))
        self.assertFalse(self.status.ready(10.33))
        self.status.observe('v2 boot 7 0 1 0 /relay 10.34 10.34', 10.34)
        self.assertFalse(self.status.ready(10.34))

    def test_status_clock_is_sampled_after_lock_is_acquired(self):
        self.status.observe('v2 boot 4 0 1 0 /relay 9.98 10.0', 10.0)
        self.status.observe('v2 boot 5 0 1 0 /relay 10.05 10.05', 10.05)
        self.assertTrue(self.status.ready(lambda: 10.06))

    def test_latches_on_mode_boot_latch_timestamp_or_zero_source_fault(self):
        bad_status = ['v2 newboot 6 0 1 0 /relay 10.08 10.08',
                      'v2 boot 4 0 1 0 /relay 10.08 10.08',
                      'v2 boot 6 1 1 0 /relay 10.08 10.08',
                      'v2 boot 6 0 0 0 /relay 10.08 10.08',
                      'v2 boot 6 0 1 1 /relay 10.08 10.08',
                      'v2 boot 6 0 1 0 /other 10.08 10.08',
                      'v2 boot 6 0 1 0 /relay 11.00 11.00',
                      'v2 boot 6 0 1 0 /relay 9.00 9.00', 'malformed']
        for bad in bad_status:
            status = relay.DriverStatus('/relay')
            status.observe('v2 boot 4 0 1 0 /relay 9.98 10.0', 10.0)
            status.observe('v2 boot 5 0 1 0 /relay 10.05 10.05', 10.05)
            self.assertTrue(status.ready(10.06))
            status.observe(bad, 10.08)
            status.observe('v2 boot 7 0 1 0 /relay 10.09 10.09', 10.09)
            self.assertFalse(status.ready(10.09), bad)


FAKE_ROS = r"""
import os, sys, time, types, threading
try:
    import imp
    relay = imp.load_source('candidate', sys.argv[1])
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location('candidate', sys.argv[1])
    relay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(relay)
mode = sys.argv[2]
relay.STARTUP_TIMEOUT_SEC = 0.8
started = relay._monotonic()
real_clock = relay._monotonic
if mode in ('main_clock_fails', 'worker_clock_fails'):
    def broken_clock():
        main_thread = threading.current_thread().name == 'MainThread'
        broken_thread = main_thread if mode == 'main_clock_fails' else not main_thread
        if broken_thread and real_clock() - started > 0.7:
            raise OSError('clock unavailable')
        return real_clock()
    relay._monotonic = broken_clock
real_ppid = os.getppid
if mode == 'parent_loss':
    os.getppid = lambda: real_ppid() if relay._monotonic() - started < 1.3 else 1

rospy = types.ModuleType('rospy')
geometry = types.ModuleType('geometry_msgs')
geometry_msg = types.ModuleType('geometry_msgs.msg')
std = types.ModuleType('std_msgs')
std_msg = types.ModuleType('std_msgs.msg')
class Vector(object):
    def __init__(self):
        self.x = self.y = self.z = 0.0
class Twist(object):
    def __init__(self):
        self.linear = Vector()
        self.angular = Vector()
class String(object):
    def __init__(self, data=''):
        self.data = data
geometry_msg.Twist = Twist
std_msg.String = String
state = {'callback': None, 'count': 0, 'zero_at': 0.0, 'hung': False, 'motion_seen': False}
class Publisher(object):
    def __init__(self, topic, cls, queue_size):
        assert topic == '/cmd_vel' and cls is Twist and queue_size == 1
    def get_num_connections(self):
        if mode == 'pause_connection' and state['motion_seen'] and not state['hung']:
            state['hung'] = True
            os.write(2, b'WAITING_FOR_CONNECTION\n')
            time.sleep(0.08)
        lost = mode == 'connection_loss' and relay._monotonic() - started > 0.8
        return 0 if mode == 'no_connection' or lost else 1
    def publish(self, msg):
        assert (msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y) == (0.0, 0.0, 0.0, 0.0)
        linear, angular = msg.linear.x, msg.angular.z
        if linear or angular:
            state['motion_seen'] = True
        if mode == 'connection_loss' and relay._monotonic() - started > 0.8 and (linear or angular):
            os.write(2, b'UNSAFE_PUBLISH_WITHOUT_CONNECTION\n')
        if mode == 'hang_publish' and (linear or angular) and not state['hung']:
            state['hung'] = True
            time.sleep(4.0)
        os.write(2, ('CMD %.4f %.4f\n' % (linear, angular)).encode('ascii'))
        now = relay._monotonic()
        if not linear and not angular:
            state['count'] += 1
            state['zero_at'] = now
        if state['callback'] and mode != 'no_status' and not (
                mode == 'status_loss' and now - started > 0.8):
            data = 'v2 boot %d 0 1 0 /relay_test %.9f %.9f' % (
                state['count'], state['zero_at'], now)
            state['callback'](String(data))
    def unregister(self): pass
def init_node(*args, **kwargs):
    if mode == 'hang_init':
        time.sleep(4.0)
rospy.init_node = init_node
rospy.get_name = lambda: '/relay_test'
rospy.Publisher = Publisher
def subscribe(topic, cls, callback, **kwargs):
    assert topic == '/rosclaw_limo/command_watchdog' and cls is String
    state['callback'] = callback
    return object()
rospy.Subscriber = subscribe
rospy.is_shutdown = lambda: False
rospy.signal_shutdown = lambda message: None
sys.modules.update({'rospy': rospy, 'geometry_msgs': geometry,
                    'geometry_msgs.msg': geometry_msg, 'std_msgs': std,
                    'std_msgs.msg': std_msg})
sys.exit(relay.main(0.3 if mode == 'absolute_deadline' else None,
                   forward_navigation=mode == 'forward_navigation'))
"""


class SubprocessTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(getattr(relay, 'main', None), 'relay main is required')
        self.assertIsNotNone(getattr(relay, '_monotonic', None), 'monotonic clock is required')
        self.proc = None
        self.lines = {1: [], 2: []}
        self.buffers = {1: b'', 2: b''}

    def start(self, mode='normal'):
        self.proc = subprocess.Popen(
            [sys.executable, '-B', '-c', FAKE_ROS,
             os.path.join(os.path.dirname(__file__), 'relay.py'), mode],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def tearDown(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.send_signal(signal.SIGTERM)
                deadline = time.time() + 1.5
                while self.proc.poll() is None and time.time() < deadline:
                    time.sleep(0.01)
                if self.proc.poll() is None:
                    self.proc.kill()
            self.proc.wait()
            for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                if stream is not None:
                    stream.close()

    def pump(self, duration=0.0):
        end = time.time() + duration
        while True:
            remaining = max(0.0, end - time.time())
            ready, _, _ = select.select([self.proc.stdout, self.proc.stderr], [], [], remaining)
            if not ready:
                break
            for stream in ready:
                index = 1 if stream is self.proc.stdout else 2
                data = os.read(stream.fileno(), 4096)
                if data:
                    self.buffers[index] += data
                    while b'\n' in self.buffers[index]:
                        line, self.buffers[index] = self.buffers[index].split(b'\n', 1)
                        self.lines[index].append(line.decode('ascii'))
            if time.time() >= end:
                break

    def await_line(self, index, prefix, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.pump(0.05)
            for line in self.lines[index]:
                if line.startswith(prefix):
                    return line
            if self.proc.poll() is not None:
                break
        self.fail('missing %r: stdout=%r stderr=%r exit=%r' % (
            prefix, self.lines[1], self.lines[2], self.proc.poll()))

    def send(self, seq, linear, angular=0.0):
        frame = '{"sequence":%d,"stamp":%.9f,"linear":%.6f,"angular":%.6f}\n' % (
            seq, relay._monotonic(), linear, angular)
        os.write(self.proc.stdin.fileno(), frame.encode('ascii'))

    def test_startup_continuous_motion_and_explicit_stop(self):
        self.start()
        self.await_line(1, 'READY')
        for seq in range(1, 5):
            self.send(seq, 0.04, 0.0)
            time.sleep(0.05)
        self.await_line(2, 'CMD 0.0400 0.0000')
        self.send(5, 0.0)
        self.await_line(1, 'STOP requested_stop')
        self.send(6, 0.04)
        self.pump(0.2)
        commands = [x for x in self.lines[2] if x.startswith('CMD ')]
        last_nonzero = max(i for i, x in enumerate(commands) if x != 'CMD 0.0000 0.0000')
        self.assertGreaterEqual(len(commands) - last_nonzero - 1, 2)
        self.assertTrue(all(x == 'CMD 0.0000 0.0000' for x in commands[last_nonzero + 1:]))

    def test_navigation_publishes_zero_then_resumes_and_stops_on_heartbeat_loss(self):
        self.start('forward_navigation')
        self.await_line(1, 'READY forward_navigation')
        sequence = 0
        for linear in (.014, 0., .014):
            start = len(self.lines[2])
            for _ in range(3):
                self.send(sequence, linear)
                sequence += 1
                self.pump(.05)
            self.assertIn('CMD %.4f 0.0000' % linear, self.lines[2][start:])
        self.await_line(1, 'FAULT heartbeat')
        self.pump(.15)
        self.assertEqual(self.lines[2][-1], 'CMD 0.0000 0.0000')

    def test_relay_enforces_independent_motion_deadline_with_fresh_heartbeats(self):
        self.start('absolute_deadline')
        self.await_line(1, 'READY motion_limit_sec=0.300000')
        for sequence in range(1, 12):
            self.send(sequence, 0.02)
            self.pump(0.05)
        self.await_line(1, 'STOP motion_deadline')
        self.pump(0.15)
        commands = [line for line in self.lines[2] if line.startswith('CMD ')]
        self.assertIn('CMD 0.0200 0.0000', commands)
        last_motion = max(i for i, line in enumerate(commands) if line == 'CMD 0.0200 0.0000')
        self.assertGreaterEqual(len(commands) - last_motion - 1, 2)
        self.assertTrue(all(line == 'CMD 0.0000 0.0000' for line in commands[last_motion + 1:]))

    def test_excess_or_turning_or_reverse_command_latches_zero(self):
        bad_commands = [(-0.000001, 0.0), (0.050001, 0.0),
                        (0.04, -0.000001), (0.04, 0.000001)]
        for linear, angular in bad_commands:
            self.lines = {1: [], 2: []}
            self.buffers = {1: b'', 2: b''}
            self.start()
            self.await_line(1, 'READY')
            self.send(1, 0.04)
            self.await_line(2, 'CMD 0.0400 0.0000')
            self.send(2, linear, angular)
            self.await_line(1, 'FAULT invalid_frame')
            self.send(3, 0.04)
            self.pump(0.15)
            self.assertEqual(self.lines[2][-1], 'CMD 0.0000 0.0000')
            self.tearDown()
            self.proc = None

    def test_subnormal_turning_on_wire_latches_zero(self):
        self.start()
        self.await_line(1, 'READY')
        self.send(1, 0.04)
        self.await_line(2, 'CMD 0.0400 0.0000')
        payload = ('{"sequence":2,"stamp":%.9f,"linear":0.04,"angular":1e-1000}\n' %
                   relay._monotonic()).encode('ascii')
        os.write(self.proc.stdin.fileno(), payload)
        self.await_line(1, 'FAULT invalid_frame')
        self.send(3, 0.04)
        self.pump(0.15)
        self.assertEqual(self.lines[2][-1], 'CMD 0.0000 0.0000')

    def test_steady_fresh_frames_keep_session_active(self):
        self.start()
        self.await_line(1, 'READY')
        for seq in range(1, 31):
            self.send(seq, 0.04)
            self.pump(0.04)
        self.assertFalse(any(x.startswith(('FAULT ', 'STOP ')) for x in self.lines[1]))
        self.assertGreaterEqual(self.lines[2].count('CMD 0.0400 0.0000'), 10)

    def test_heartbeat_expiry_latches_even_if_parent_resumes(self):
        self.start()
        self.await_line(1, 'READY')
        self.send(1, 0.04)
        self.await_line(2, 'CMD 0.0400 0.0000')
        self.await_line(1, 'FAULT heartbeat')
        self.send(2, 0.04)
        self.pump(0.15)
        self.assertEqual(self.lines[2][-1], 'CMD 0.0000 0.0000')

    def test_shutdown_signal_requests_repeated_zero(self):
        self.start()
        self.await_line(1, 'READY')
        self.send(1, 0.04)
        self.await_line(2, 'CMD 0.0400 0.0000')
        self.proc.send_signal(signal.SIGTERM)
        self.pump(0.55)
        commands = [x for x in self.lines[2] if x.startswith('CMD ')]
        last_motion = max(i for i, x in enumerate(commands) if x != 'CMD 0.0000 0.0000')
        self.assertGreaterEqual(len(commands) - last_motion - 1, 3)
        self.assertTrue(all(x == 'CMD 0.0000 0.0000' for x in commands[last_motion + 1:]))

    def test_pipe_eof_latches_and_repeats_zero(self):
        self.start()
        self.await_line(1, 'READY')
        self.send(1, 0.04)
        self.await_line(2, 'CMD 0.0400 0.0000')
        self.proc.stdin.close()
        self.proc.stdin = None
        self.await_line(1, 'FAULT parent_eof')
        self.pump(0.2)
        self.assertGreaterEqual(self.lines[2].count('CMD 0.0000 0.0000'), 4)
        self.assertEqual(self.lines[2][-1], 'CMD 0.0000 0.0000')

    def test_no_ros_connection_never_becomes_ready(self):
        self.start('no_connection')
        self.await_line(1, 'FAULT startup_timeout')
        self.assertNotIn('READY', self.lines[1])
        self.assertTrue(self.lines[2])
        self.assertTrue(all(x == 'CMD 0.0000 0.0000' for x in self.lines[2]))

    def test_no_driver_status_never_becomes_ready(self):
        self.start('no_status')
        self.await_line(1, 'FAULT startup_timeout')
        self.assertNotIn('READY', self.lines[1])
        self.assertTrue(self.lines[2])
        self.assertTrue(all(x == 'CMD 0.0000 0.0000' for x in self.lines[2]))

    def test_status_loss_latches_while_frames_remain_fresh(self):
        self.start('status_loss')
        self.await_line(1, 'READY')
        deadline = time.time() + 1.5
        seq = 1
        while time.time() < deadline and not any(x.startswith('FAULT ') for x in self.lines[1]):
            self.send(seq, 0.04)
            seq += 1
            self.pump(0.05)
        self.await_line(1, 'FAULT driver_status')
        self.pump(0.15)
        self.assertEqual(self.lines[2][-1], 'CMD 0.0000 0.0000')

    def test_parent_loss_with_pipe_still_open_latches(self):
        self.start('parent_loss')
        self.await_line(1, 'READY')
        deadline = time.time() + 2.0
        seq = 1
        while time.time() < deadline and not any(x.startswith('FAULT ') for x in self.lines[1]):
            self.send(seq, 0.04)
            seq += 1
            self.pump(0.05)
        self.await_line(1, 'FAULT parent_lost')
        self.pump(0.15)
        self.assertEqual(self.lines[2][-1], 'CMD 0.0000 0.0000')

    def test_blocked_ros_initialization_reports_timeout_and_exits_on_signal(self):
        self.start('hang_init')
        self.await_line(1, 'FAULT startup_timeout', timeout=1.5)
        self.proc.send_signal(signal.SIGTERM)
        deadline = time.time() + 1.3
        while self.proc.poll() is None and time.time() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(self.proc.poll(), 'ROS init must not block shutdown')

    def test_control_clock_failure_requests_repeated_zero_and_nonzero_exit(self):
        self.start('main_clock_fails')
        self.await_line(1, 'READY')
        seq = 1
        deadline = time.time() + 1.2
        while self.proc.poll() is None and time.time() < deadline:
            try:
                self.send(seq, 0.04)
            except OSError:
                break
            seq += 1
            self.pump(0.05)
        self.await_line(1, 'FAULT control_error')
        self.pump(0.5)
        self.assertNotEqual(self.proc.poll(), 0)
        commands = [x for x in self.lines[2] if x.startswith('CMD ')]
        self.assertGreaterEqual(commands[-5:].count('CMD 0.0000 0.0000'), 3)

    def test_worker_clock_failure_still_attempts_repeated_zero(self):
        self.start('worker_clock_fails')
        self.await_line(1, 'READY')
        seq = 1
        deadline = time.time() + 1.2
        while time.time() < deadline and not any(x.startswith('FAULT ') for x in self.lines[1]):
            self.send(seq, 0.04)
            seq += 1
            self.pump(0.05)
        self.await_line(1, 'FAULT ros_error')
        self.pump(0.5)
        commands = [x for x in self.lines[2] if x.startswith('CMD ')]
        self.assertGreaterEqual(commands[-5:].count('CMD 0.0000 0.0000'), 3)

    def test_a_blocked_ros_publish_does_not_block_the_control_loop(self):
        self.start('hang_publish')
        self.await_line(1, 'READY')
        self.send(1, 0.04)
        self.await_line(1, 'FAULT ', timeout=1.0)
        self.assertIsNone(self.proc.poll())
        self.proc.send_signal(signal.SIGTERM)
        deadline = time.time() + 1.3
        while self.proc.poll() is None and time.time() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(self.proc.poll(), 'shutdown must not wait indefinitely for ROS')

    def test_connection_loss_is_checked_before_each_motion_publication(self):
        self.start('connection_loss')
        self.await_line(1, 'READY')
        deadline = time.time() + 1.5
        seq = 1
        while time.time() < deadline and not any(x.startswith('FAULT ') for x in self.lines[1]):
            self.send(seq, 0.04)
            seq += 1
            self.pump(0.05)
        self.await_line(1, 'FAULT connection_lost')
        self.assertNotIn('UNSAFE_PUBLISH_WITHOUT_CONNECTION', self.lines[2])

    def test_fault_while_connection_check_waits_cannot_send_stale_motion(self):
        self.start('pause_connection')
        self.await_line(1, 'READY')
        self.send(1, 0.04)
        self.await_line(2, 'WAITING_FOR_CONNECTION')
        self.proc.stdin.close()
        self.proc.stdin = None
        self.await_line(1, 'FAULT parent_eof')
        self.pump(0.3)
        index = self.lines[2].index('WAITING_FOR_CONNECTION')
        later = [x for x in self.lines[2][index + 1:] if x.startswith('CMD ')]
        self.assertTrue(later)
        self.assertTrue(all(x == 'CMD 0.0000 0.0000' for x in later), repr(later))

    def test_partial_oversize_frame_faults_and_cannot_resume(self):
        self.start()
        self.await_line(1, 'READY')
        self.send(1, 0.04)
        self.await_line(2, 'CMD 0.0400 0.0000')
        os.write(self.proc.stdin.fileno(), b'x' * 513)
        self.await_line(1, 'FAULT invalid_frame')
        self.send(2, 0.04)
        self.pump(0.15)
        self.assertEqual(self.lines[2][-1], 'CMD 0.0000 0.0000')

    def test_input_before_ready_never_produces_motion(self):
        self.start()
        self.send(1, 0.04)
        self.await_line(1, 'FAULT startup_input')
        self.pump(0.15)
        self.assertNotIn('READY', self.lines[1])
        self.assertTrue(all(x == 'CMD 0.0000 0.0000' for x in self.lines[2]))


if __name__ == '__main__':
    unittest.main()
