#!/usr/bin/env python3
"""Run the threaded facade against a private ROS master and fake PTY chassis.

Use the bubblewrap command documented in live_trial_pty_check.py, replacing its
last line with ``/usr/bin/python3 /sandbox/threaded_relay_pty_check.py live``
or ``... expired``, ``... long``, ``... long_expired`` or ``... long_model_pause``.
These scenarios use the real trial, RelayProcessTransport,
relay and candidate driver; source, model and ROS/serial guards are fixtures.
No firmware, actual exclusive ownership, path clearance or stopping is proved.
"""
import collections
import hashlib
import io
import json
import sys
from pathlib import Path

import live_trial_pty_check as harness

MONITOR = r'''
import sys
import rospy
import rosgraph
import json
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from relay import _monotonic
rospy.init_node('threaded_pty_monitor', anonymous=False, disable_signals=True)
def status(message):
    sys.stdout.write('STATUS ' + message.data + '\n')
    sys.stdout.flush()
def cmd(message):
    caller = getattr(message, '_connection_header', {}).get('callerid', 'unknown')
    sys.stdout.write('CMD %.9f %.9f %.9f %s\n' %
                     (_monotonic(), message.linear.x, message.angular.z, caller))
    sys.stdout.flush()
watch = rospy.Subscriber('/rosclaw_limo/command_watchdog', String, status, queue_size=100)
velocity = rospy.Subscriber('/cmd_vel', Twist, cmd, queue_size=100)
sys.stdout.write('MONITOR_READY\n')
sys.stdout.flush()
master = rosgraph.Master(rospy.get_name())
while not rospy.is_shutdown():
    try:
        publishers, _, _ = master.getSystemState()
        velocity_callers = next((nodes for topic, nodes in publishers if topic == '/cmd_vel'), [])
        sys.stdout.write('GRAPH ' + json.dumps({'at': _monotonic(), 'cmd_vel': velocity_callers}) + '\n')
        sys.stdout.flush()
    except Exception:
        sys.stdout.write('GRAPH_ERROR\n')
        sys.stdout.flush()
    rospy.rostime.wallsleep(0.05)
'''


CHILD_TEMPLATE = r'''
from __future__ import print_function
import json, threading, time
from live_trial_adapter import DecisionsAdapter, LiveTrial, LongerForwardTrial, RelayProcessTransport
from relay import _monotonic
from test_live_trial_adapter import FakeArbiter, FakeSerialGuard, source
from threaded_relay_adapter import ThreadedRelayAdapter

case = CASE_NAME
class Clock(object):
    def now(self):
        return _monotonic()
    def sleep(self, seconds):
        time.sleep(seconds)

clock = Clock()
observation, _ = source(clock)
transport = RelayProcessTransport(clock)
writes, reads, submissions = [], [], []
real_write, real_read = transport.write_line, transport.read_line
def record_write(data, timeout):
    accepted = real_write(data, timeout)
    if accepted:
        writes.append({'at': clock.now(), 'writer': threading.current_thread().ident,
                       'frame': json.loads(data)})
    return accepted
def record_read(timeout):
    line = real_read(timeout)
    if line is not None:
        reads.append({'at': clock.now(), 'line': line})
    return line
transport.write_line, transport.read_line = record_write, record_read
limit = 2.0 if case in ('long', 'long_expired', 'long_model_pause') else 0.3
relay = ThreadedRelayAdapter(transport, clock, motion_limit_sec=limit)
real_submit = relay.send_frame
def submit(sequence, stamp, linear, angular, valid_until=None):
    accepted = real_submit(sequence, stamp, linear, angular, valid_until=valid_until)
    if accepted:
        submissions.append({'sequence': sequence, 'stamp': stamp, 'linear': linear,
                            'valid_until': valid_until})
    return accepted
relay.send_frame = submit
pause = {}
class PausedSource(object):
    def snapshot(self):
        trigger = (case == 'live' and relay.armed or
                   case in ('expired', 'long_expired') and relay.evidence['wire_nonzero'] > 0)
        if trigger and not pause:
            pause['start'] = clock.now()
            time.sleep(0.30)
            pause['end'] = clock.now()
        return observation.snapshot()
    def __getattr__(self, name):
        return getattr(observation, name)

calls = []
def fake_model(payload, timeout):
    calls.append(True)
    if case == 'long_model_pause':
        pause['start'] = clock.now()
        time.sleep(0.30)
        pause['end'] = clock.now()
    return {'ok': True, 'choice': 'forward', 'captured_at': payload['captured_at']}
arbiter, serial = FakeArbiter(), FakeSerialGuard()
trial_class = LongerForwardTrial if case in ('long', 'long_expired', 'long_model_pause') else LiveTrial
trial = trial_class(PausedSource(), DecisionsAdapter(fake_model), relay, arbiter, clock,
                    enable_motion=True, duration_sec=2.0 if case in ('long', 'long_expired', 'long_model_pause') else 0.6,
                    speed_mps=0.02, period_sec=0.05, serial_guard=serial)
started = clock.now()
result = trial.run()
returned = clock.now()
output = {'trial': result, 'writes': writes, 'status_reads': reads, 'model_calls': len(calls),
          'started_at': started, 'returned_at': returned, 'pause': pause,
          'submissions': submissions, 'control_thread': threading.current_thread().ident,
          'fake_arbiter_released': arbiter.released, 'fake_serial_released': serial.released}
with open('/tmp/threaded-live-child.json', 'w') as stream:
    json.dump(output, stream, sort_keys=True)
print(json.dumps(output, sort_keys=True))
'''


class Collector(harness.StatusCollector):
    instances = []

    def __init__(self, stream):
        super(Collector, self).__init__(stream)
        self.commands = collections.deque(maxlen=4096)
        self.graphs = collections.deque(maxlen=4096)
        self.instances.append(self)

    def run(self):
        for raw in self.stream:
            line = raw.decode('ascii', 'replace').strip()
            if line == 'MONITOR_READY':
                self.ready.set()
            elif line.startswith('STATUS '):
                fields = line.split()[1:]
                if len(fields) == 9 and fields[0] == 'v2':
                    with self.lock:
                        self.statuses.append((harness.time.monotonic(), fields))
                else:
                    self.errors.append('malformed driver status')
            elif line.startswith('CMD '):
                fields = line.split()
                if len(fields) != 5:
                    self.errors.append('malformed cmd_vel observation')
                else:
                    with self.lock:
                        self.commands.append((float(fields[1]), float(fields[2]),
                                              float(fields[3]), fields[4]))
            elif line.startswith('GRAPH '):
                try:
                    state = json.loads(line[6:])
                    with self.lock:
                        self.graphs.append(state)
                except (TypeError, ValueError):
                    self.errors.append('malformed ROS graph observation')
            elif line == 'GRAPH_ERROR':
                self.errors.append('ROS graph query failed')


def main(case):
    if case not in ('live', 'expired', 'long', 'long_expired', 'long_model_pause'):
        raise ValueError('unknown scenario')
    # The reused harness refuses to allocate a PTY unless namespace checks pass.
    harness.CHILD = CHILD_TEMPLATE.replace('CASE_NAME', repr(case))
    harness.STATUS_MONITOR = MONITOR
    harness.StatusCollector = Collector
    # The thread may reuse a just-written zero; relay READY separately proves
    # a fresh zero was completed at the fake driver before any motion.
    harness.MIN_PRE_MOTION_ZERO_WRITES = 1
    harness.MOTION_LIMIT_SEC = 2.0 if case in ('long', 'long_expired', 'long_model_pause') else 0.3
    captured = io.StringIO()
    threaded_path = Path('/sandbox/threaded_relay_adapter.py')
    threaded_digest = hashlib.sha256(threaded_path.read_bytes()).hexdigest()
    real_stdout = sys.stdout
    try:
        sys.stdout = captured
        harness.main()
    finally:
        sys.stdout = real_stdout
    summary = json.loads(captured.getvalue())
    child = json.loads(Path('/tmp/threaded-live-child.json').read_text())
    if hashlib.sha256(threaded_path.read_bytes()).hexdigest() != threaded_digest:
        raise RuntimeError('threaded adapter changed during integration')
    writes, submissions, pause = child['writes'], child['submissions'], child['pause']
    if case != 'long' and (not pause or not 0.25 <= pause['end'] - pause['start'] <= 0.5):
        raise RuntimeError('controlled processing pause was not observed')
    if not submissions or not writes:
        raise RuntimeError('no control intent or wire writes')
    writer_ids = set(item['writer'] for item in writes)
    if len(writer_ids) != 1 or child['control_thread'] in writer_ids:
        raise RuntimeError('pipe writes did not use exactly one independent worker')
    for item in writes:
        frame = item['frame']
        if frame['linear'] > 0:
            earlier = [entry['stamp'] for entry in submissions if entry['stamp'] <= frame['stamp']]
            if not earlier or not 0 <= frame['stamp'] - max(earlier) < 0.16:
                raise RuntimeError('a nonzero pipe frame lacked a fresh control intent')
            bounds = [entry['valid_until'] for entry in submissions
                      if entry['stamp'] <= frame['stamp'] and entry['valid_until'] is not None]
            if not bounds or frame.get('expires_at') > bounds[-1]:
                raise RuntimeError('a nonzero pipe frame exceeded its observation deadline')
    commands = list(Collector.instances[-1].commands)
    moving = [item for item in commands if item[1] > 0]
    if (not moving or any(abs(item[1] - 0.02) > 1e-7 or item[2] != 0
                          or not item[3].startswith('/standalone_limo_relay_') for item in moving)):
        raise RuntimeError('nonzero /cmd_vel was not independently observed from the relay')
    stop_at = summary['stop_line_observed_monotonic_s']
    post_stop = [item for item in commands if item[0] >= stop_at + 0.15]
    if any(item[1] != 0 or item[2] != 0 for item in post_stop):
        raise RuntimeError('nonzero cmd_vel continued after stop propagation')
    if len([item for item in commands if item[0] >= stop_at and item[1] == 0 and item[2] == 0]) < 3:
        raise RuntimeError('insufficient observed zero cmd_vel after stop')
    graphs = list(Collector.instances[-1].graphs)
    active = [state for state in graphs if any(name.startswith('/standalone_limo_relay_')
                                               for name in state['cmd_vel'])]
    after = [state for state in graphs if state['at'] >= child['returned_at'] + 0.4]
    if (not active or not after or any(any(name.startswith('/standalone_limo_relay_')
                                           for name in state['cmd_vel']) for state in after)):
        raise RuntimeError('relay cmd_vel registration was not removed after process exit')
    extra = {'scenario': case, 'controlled_pause_sec': round(pause['end'] - pause['start'], 6) if pause else None,
             'unique_pipe_writer_threads': len(writer_ids), 'control_thread_writes_pipe': False,
             'setpoints_submitted': len(submissions), 'all_nonzero_pipe_frames_within_intent_ttl': True,
             'cmd_vel_nonzero_observations': len(moving),
             'cmd_vel_last_nonzero_monotonic_s': moving[-1][0],
             'cmd_vel_nonzero_after_stop_plus_0_15': 0,
             'ros_graph_samples_after_exit': len(after), 'relay_cmd_vel_registration_removed': True,
             'threaded_relay_sha256': threaded_digest}
    if case in ('live', 'long_model_pause'):
        during = [item for item in writes if pause['start'] <= item['at'] <= pause['end']]
        if len(during) < 4 or any(item['frame']['linear'] != 0 for item in during):
            raise RuntimeError('zero heartbeats did not continue during the idle processing pause')
        expected = ('motion_limit',) if case == 'live' else ('motion_limit', 'time_limit')
        if summary['trial_reason'] not in expected:
            raise RuntimeError('full live trial did not reach its motion limit')
        if case == 'long_model_pause' and summary['serial_nonzero_span_sec'] <= 1.0:
            raise RuntimeError('long model-pause trial did not sustain forward commands')
        extra['zero_pipe_heartbeats_during_idle_pause'] = len(during)
    elif case in ('expired', 'long_expired'):
        fault = child['trial']['relay_evidence'].get('worker_fault')
        stop_line = child['trial']['relay_evidence'].get('stop_line')
        expected_faults = ('setpoint_expired', 'relay_unhealthy', 'requested_stop') if case == 'long_expired' else (
            'setpoint_expired', 'relay_unhealthy')
        expected_reasons = ('relay_failure', 'sensor_unavailable', 'expiry_unavailable') if case == 'long_expired' else (
            'relay_failure',)
        if (fault not in expected_faults
                or stop_line not in ('STOP requested_stop\n', 'FAULT intent_deadline\n')
                or summary['trial_reason'] not in expected_reasons):
            raise RuntimeError('unrefreshed intent did not permanently stop the worker')
        extra['worker_fault'] = fault
        extra['pause_started_after_first_nonzero'] = True
    elif summary['serial_nonzero_span_sec'] <= 1.0:
        raise RuntimeError('longer trial did not sustain simulated forward commands: ' + json.dumps({
            'span_sec': summary['serial_nonzero_span_sec'], 'trial': child['trial'],
            'submissions': submissions[-5:], 'writes': writes[-8:]}))
    summary.update(extra)
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) == 2 else '')
