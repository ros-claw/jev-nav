#!/usr/bin/env python2
"""ROS 1 / Python 2.7 velocity relay for a *candidate*, not a deployed system.

A supervisor starts one process with a private inherited stdin pipe and reads
stdout without blocking. Wait for ``READY\n`` before writing newline-delimited
UTF-8 JSON objects with exactly ``sequence``, ``stamp``, ``linear``, ``angular``.
Sequence is a strictly increasing nonnegative integer; stamp is a strictly
increasing CLOCK_MONOTONIC timestamp from this same host. Start sending within
0.2 seconds of READY and keep sending fresh frames faster than the 0.2-second
heartbeat deadline. Linear x is restricted to [0.0, 0.05] m/s and angular z
must be zero; every other Twist field remains zero. JSON velocity limits use
the exact input decimal, before conversion to floats. Frames are at most 512
bytes before the newline. Stdin and stdout are nonblocking in this process.

When --require-intent-deadline is set, each frame also contains an absolute
``expires_at`` in the same monotonic clock domain. The relay latches zero at
that deadline even if its ordinary heartbeat has not yet expired.

An optional --max-motion-sec bounds relay time from the first nonzero frame.
The READY line then includes motion_limit_sec; the deadline latches zero even
if fresh heartbeats continue. Scheduler stalls, transport, host failure and
actual chassis braking can delay or prevent a physical stop.

With --forward-navigation the READY line is ``READY forward_navigation``.
The forward limit is 0.02 m/s. Zero is an ordinary setpoint and the local
controller heartbeat still latches on disconnect. The trial deadline flags
are reserved for the separate bounded trial protocol.

The relay publishes /cmd_vel and subscribes to the candidate driver's
/rosclaw_limo/command_watchdog (std_msgs/String v2); that topic name is a driver
contract and does not require the ROSClaw service. Before READY it publishes
zero while waiting for a stable ROS connection, then requires a recent driver
report with mode=0, primed=1, latched=0, and a new zero-write counter from this
relay's ROS caller ID. These 20 Hz driver reports have a 0.25-second freshness
limit. They are distinct from the separately handled approximately 2 Hz chassis
status and from the 0.2-second control-frame heartbeat.

On fault, stdout attempts one ``FAULT <reason>\n``; the command is latched at
zero and repeatedly published while possible. A zero command after any motion
frame ends the session with ``STOP requested_stop\n`` and the same zero latch.
Neither event resumes in the same process: automatic resume within a task is
not supported; the supervisor must start a fresh, explicitly authorized
session. EOF, parent loss, stale or malformed input, driver status/connection
loss, and a stalled ROS worker all latch. SIGINT/SIGTERM requests a short zero
burst and bounded worker join. An unavailable driver watchdog prevents READY.
The parent must itself use timeouts, drain or bound its pipes, and reap the
relay. Status writes are best effort and are not acknowledgments of motion.

The watchdog report describes a driver-recorded zero write, not a physical stop.
ROS publication and connection count do not establish chassis behavior. A
blocked ROS call can delay a zero or leave an in-flight nonzero message, and no
process can publish during host power loss. Safe operation additionally needs a
verified independent driver/base watchdog, exclusive command arbitration, a
physical emergency stop, and validated sensors, modes, and surroundings. The
candidate driver watchdog is disabled by default in its source; its deployment
and behavior have not been established by this module or its mocked tests.
"""
from __future__ import print_function

import ctypes
import errno
import fcntl
import json
import math
import os
import select
import signal
import stat
import sys
import threading
import time
from decimal import Decimal, InvalidOperation

try:
    INTEGER_TYPES = (int, long)
except NameError:
    INTEGER_TYPES = (int,)

HEARTBEAT_SEC = 0.2
MAX_LINEAR = 0.05
_MAX_LINEAR_DECIMAL = Decimal('0.05')
MAX_NAV_LINEAR = .02
_MAX_NAV_LINEAR_DECIMAL = Decimal('0.02')


class _JSONFloat(float):
    """Keep the exact wire value for the safety envelope before float rounding."""
    def __new__(cls, text):
        try:
            result = float.__new__(cls, text)
            result.exact = Decimal(text)
            return result
        except (InvalidOperation, ValueError, OverflowError):
            raise ValueError('invalid JSON number')


def _safe_velocity(linear, angular, forward_navigation=False):
    exact_linear = linear.exact if isinstance(linear, _JSONFloat) else linear
    exact_angular = angular.exact if isinstance(angular, _JSONFloat) else angular
    if forward_navigation:
        upper = _MAX_NAV_LINEAR_DECIMAL if isinstance(linear, _JSONFloat) else MAX_NAV_LINEAR
    else:
        upper = _MAX_LINEAR_DECIMAL if isinstance(linear, _JSONFloat) else MAX_LINEAR
    return 0 <= exact_linear <= upper and exact_angular == 0


def _number(value):
    if isinstance(value, bool) or not isinstance(value, INTEGER_TYPES + (float,)):
        return False
    try:
        return not math.isnan(value) and not math.isinf(value)
    except (ValueError, OverflowError):
        return False


class RelayGate(object):
    def __init__(self, started_at, max_motion_sec=None, require_intent_deadline=False, forward_navigation=False):
        if (not _number(started_at) or (max_motion_sec is not None
                and (not _number(max_motion_sec) or not 0 < max_motion_sec <= 30))
                or type(require_intent_deadline) is not bool or type(forward_navigation) is not bool
                or forward_navigation and (max_motion_sec is not None or require_intent_deadline)):
            raise ValueError('invalid start time')
        self.last_at = float(started_at)
        self.last_stamp = None
        self.sequence = -1
        self.linear = 0.0
        self.angular = 0.0
        self.stopped = False
        self.reason = None
        self.had_motion = False
        self.motion_started_at = None
        self.max_motion_sec = max_motion_sec
        self.require_intent_deadline = require_intent_deadline
        self.forward_navigation = forward_navigation
        self.intent_deadline = None

    def stop(self, reason):
        self.stopped = True
        if self.reason is None:
            self.reason = reason
        self.linear, self.angular = 0.0, 0.0

    def accept(self, frame, now):
        if self.stopped:
            return False
        if (self.intent_deadline is not None and _number(now) and now >= self.intent_deadline):
            self.stop('intent_deadline')
            return False
        if (self.motion_started_at is not None and self.max_motion_sec is not None
                and _number(now) and now >= self.motion_started_at + self.max_motion_sec):
            self.stop('motion_deadline')
            return False
        keys = set(['sequence', 'stamp', 'linear', 'angular'])
        if self.require_intent_deadline:
            keys.add('expires_at')
        valid = (isinstance(frame, dict) and
                 set(frame) == keys and
                 isinstance(frame.get('sequence'), INTEGER_TYPES) and
                 not isinstance(frame.get('sequence'), bool) and
                 self.sequence < frame['sequence'] <= 9223372036854775807 and
                 _number(now) and _number(frame.get('stamp')) and
                 _number(frame.get('linear')) and _number(frame.get('angular')))
        if valid:
            valid = (0 <= now - self.last_at < HEARTBEAT_SEC and
                     0 <= now - frame['stamp'] < HEARTBEAT_SEC and
                     (self.last_stamp is None or frame['stamp'] > self.last_stamp) and
                     _safe_velocity(frame['linear'], frame['angular'], self.forward_navigation))
        if valid and self.require_intent_deadline:
            expiry = frame['expires_at']
            valid = (_number(expiry) and frame['stamp'] < expiry <= frame['stamp'] + HEARTBEAT_SEC
                     and now < expiry)
        if not valid:
            self.stop('invalid_frame')
            return False
        self.sequence = frame['sequence']
        self.last_at = float(now)
        self.last_stamp = float(frame['stamp'])
        self.linear = float(frame['linear'])
        self.angular = float(frame['angular'])
        self.intent_deadline = float(frame['expires_at']) if self.require_intent_deadline else None
        if self.linear or self.angular:
            self.had_motion = True
            if self.motion_started_at is None:
                self.motion_started_at = float(now)
        elif self.had_motion and not self.forward_navigation:
            self.stop('requested_stop')
        return True

    def command(self, now):
        if (self.intent_deadline is not None and _number(now) and now >= self.intent_deadline):
            self.stop('intent_deadline')
        elif not _number(now) or not 0 <= now - self.last_at < HEARTBEAT_SEC:
            self.stop('heartbeat')
        elif (self.motion_started_at is not None and self.max_motion_sec is not None
              and now >= self.motion_started_at + self.max_motion_sec):
            self.stop('motion_deadline')
        if self.stopped:
            return (0.0, 0.0)
        return (self.linear, self.angular)


MAX_FRAME_BYTES = 512
STATUS_MAX_AGE_SEC = 0.25


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError('non-finite JSON number')


class FrameReader(object):
    def __init__(self):
        self.pending = b''

    def feed(self, data):
        self.pending += data
        result = []
        while b'\n' in self.pending:
            line, self.pending = self.pending.split(b'\n', 1)
            if len(line) > MAX_FRAME_BYTES:
                raise ValueError('oversized frame')
            try:
                result.append(json.loads(line.decode('utf-8'), object_pairs_hook=_unique_pairs,
                                         parse_float=_JSONFloat, parse_constant=_invalid_constant))
            except (ValueError, UnicodeError):
                raise ValueError('invalid frame')
        if len(self.pending) > MAX_FRAME_BYTES:
            raise ValueError('oversized frame')
        return result


class DriverStatus(object):
    """Track a candidate driver's written-zero report, independently of commands."""
    def __init__(self, expected_publisher):
        self.lock = threading.Lock()
        self.expected_publisher = expected_publisher
        self.boot_id = None
        self.initial_count = None
        self.initial_generated_at = None
        self.count = None
        self.mode = None
        self.primed = None
        self.zero_source = None
        self.zero_written_at = None
        self.received_at = None
        self.generated_at = None
        self.active = False
        self.fault = False

    def observe(self, data, now):
        with self.lock:
            if self.fault:
                return
            try:
                if not _number(now) or len(data) > 256:
                    raise ValueError('invalid status')
                fields = data.split()
                if len(fields) != 9 or fields[0] != 'v2' or not fields[1]:
                    raise ValueError('invalid status')
                if not all(value.isdigit() for value in fields[2:6]):
                    raise ValueError('invalid status')
                count, mode, primed, latched = [int(value) for value in fields[2:6]]
                zero_at, generated = float(fields[7]), float(fields[8])
                if (count > 18446744073709551615 or mode > 255 or primed > 1 or
                        latched != 0 or not _number(zero_at) or not _number(generated) or
                        not 0 <= now - generated < STATUS_MAX_AGE_SEC or
                        not 0 <= zero_at <= generated):
                    raise ValueError('invalid or stale status')
                if self.boot_id is None:
                    self.boot_id = fields[1]
                    self.initial_count = count
                    self.initial_generated_at = generated
                elif (fields[1] != self.boot_id or count < self.count or
                      generated < self.generated_at or now < self.received_at or
                      zero_at < self.zero_written_at or
                      (count > self.count and zero_at <= self.zero_written_at)):
                    raise ValueError('reordered status')
                if self.active and (mode != 0 or primed != 1 or
                                    (count > self.count and fields[6] != self.expected_publisher)):
                    raise ValueError('driver changed state or zero source')
                self.count, self.mode, self.primed = count, mode, primed
                self.zero_source, self.zero_written_at = fields[6], zero_at
                self.received_at, self.generated_at = now, generated
            except (TypeError, ValueError, OverflowError):
                self.fault = True

    def ready(self, now):
        with self.lock:
            if callable(now):
                try:
                    now = now()
                except Exception:
                    self.fault = True
                    return False
            if self.fault or not _number(now):
                self.fault = True
                return False
            if self.received_at is None:
                return False
            if not (0 <= now - self.received_at < STATUS_MAX_AGE_SEC and
                    0 <= now - self.generated_at < STATUS_MAX_AGE_SEC):
                self.fault = True
                return False
            if self.active:
                return True
            ready = (self.count > self.initial_count and self.mode == 0 and self.primed == 1 and
                     self.zero_source == self.expected_publisher and
                     self.zero_written_at >= self.initial_generated_at and
                     0 <= now - self.zero_written_at < STATUS_MAX_AGE_SEC)
            if ready:
                self.active = True
            return ready


PERIOD_SEC = 0.05
STARTUP_TIMEOUT_SEC = 3.0
CONNECTION_STABLE_SEC = 0.2
WORKER_STALL_SEC = 0.2
SHUTDOWN_ZEROS_SEC = 0.5
SHUTDOWN_JOIN_SEC = 0.8


class _Timespec(ctypes.Structure):
    _fields_ = [('seconds', ctypes.c_long), ('nanoseconds', ctypes.c_long)]


# clock_gettime is a short, nonblocking clock read. A CDLL call releases the
# Python 2 GIL on every read and can make a time check wait behind sensor work.
_CLOCK_GETTIME = ctypes.PyDLL(None, use_errno=True).clock_gettime
_CLOCK_GETTIME.argtypes = (ctypes.c_int, ctypes.POINTER(_Timespec))
_CLOCK_GETTIME.restype = ctypes.c_int


def _monotonic():
    value = _Timespec()
    if _CLOCK_GETTIME(1, ctypes.byref(value)) != 0:
        raise OSError(ctypes.get_errno(), 'clock_gettime failed')
    return value.seconds + value.nanoseconds / 1000000000.0


class _Shared(object):
    """Small locked state; no ROS operation is called while holding the lock."""
    def __init__(self):
        self.lock = threading.Lock()
        self.fault = None
        self.ready = False
        self.armed = False
        self.command = (0.0, 0.0)
        self.command_at = None
        self.intent_deadline = None
        self.progress = None
        self.shutdown_at = None

    def snapshot(self):
        with self.lock:
            return (self.fault, self.ready, self.armed, self.command, self.command_at,
                    self.progress, self.shutdown_at)

    def stop(self, reason):
        with self.lock:
            if self.fault is None:
                self.fault = reason
            self.command = (0.0, 0.0)

    def mark_ready(self):
        with self.lock:
            if self.fault is None:
                self.ready = True

    def arm(self, now):
        with self.lock:
            if self.fault is None:
                self.armed = True
                self.command_at = now

    def set_command(self, command, now, intent_deadline=None):
        with self.lock:
            if self.fault is None:
                self.command = command
                self.command_at = now
                self.intent_deadline = intent_deadline

    def command_for_publish(self, clock):
        """Check the command deadline under the same lock as the command."""
        with self.lock:
            now = clock()
            if not _number(now):
                if self.fault is None:
                    self.fault = 'clock_invalid'
                self.command = (0.0, 0.0)
            elif self.intent_deadline is not None and now >= self.intent_deadline:
                if self.fault is None:
                    self.fault = 'intent_deadline'
                self.command = (0.0, 0.0)
            return self.fault, self.command, self.shutdown_at

    def progressed(self, now):
        with self.lock:
            self.progress = now

    def worker_stalled(self, clock):
        with self.lock:
            now = clock()
            return self.progress is None or not 0 <= now - self.progress < WORKER_STALL_SEC

    def shutdown(self, now):
        with self.lock:
            if self.fault is None:
                self.fault = 'shutdown'
            self.command = (0.0, 0.0)
            self.shutdown_at = now


def _ros_worker(shared, parent_pid):
    """Own every potentially blocking ROS call away from the control loop."""
    publisher = None
    rospy = None
    failed = False
    try:
        import rospy
        from geometry_msgs.msg import Twist
        from std_msgs.msg import String

        rospy.init_node('standalone_limo_relay', anonymous=True, disable_signals=True,
                        disable_rosout=True, disable_rostime=True)
        publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        status = DriverStatus(rospy.get_name())
        _subscriber = rospy.Subscriber('/rosclaw_limo/command_watchdog', String,
                                     lambda msg: status.observe(msg.data, _monotonic()),
                                     queue_size=1)
        # Keep the subscription alive for the entire session.
        stable_since = None
        while True:
            now = _monotonic()
            fault, ready, armed, command, command_at, _, shutdown_at = shared.snapshot()
            if shutdown_at is not None and now - shutdown_at >= SHUTDOWN_ZEROS_SEC:
                break
            if os.getppid() != parent_pid:
                shared.stop('parent_lost')
                fault = 'parent_lost'
            try:
                if rospy.is_shutdown():
                    shared.stop('ros_shutdown')
                    fault = 'ros_shutdown'
                connected = publisher.get_num_connections() > 0
                now = _monotonic()
                if ready and not connected:
                    shared.stop('connection_lost')
                    fault = 'connection_lost'
                if ready and not status.ready(_monotonic):
                    shared.stop('driver_status')
                    fault = 'driver_status'
                if armed and (command_at is None or not 0 <= now - command_at < HEARTBEAT_SEC):
                    shared.stop('heartbeat')
                    fault = 'heartbeat'
                latest_fault, latest_command, latest_shutdown = shared.command_for_publish(_monotonic)
                if (fault is not None or latest_fault is not None or not ready or not armed or
                        shutdown_at is not None or latest_shutdown is not None):
                    command = (0.0, 0.0)
                else:
                    command = latest_command
                message = Twist()
                message.linear.x, message.angular.z = command
                publisher.publish(message)
                now = _monotonic()
                if not ready:
                    if connected:
                        if stable_since is None:
                            stable_since = now
                        if now - stable_since >= CONNECTION_STABLE_SEC and status.ready(_monotonic):
                            shared.mark_ready()
                    else:
                        stable_since = None
                shared.progressed(_monotonic())
            except Exception:
                shared.stop('ros_error')
            time.sleep(PERIOD_SEC)
    except Exception:
        failed = True
        shared.stop('ros_error')
    finally:
        if publisher is not None and failed:
            # The monotonic clock or another worker operation failed. Request a
            # short zero burst without depending on that clock being available.
            for _ in range(10):
                try:
                    publisher.publish(Twist())
                except Exception:
                    pass
                time.sleep(PERIOD_SEC)
        # Cleanup calls stay in this daemon thread; the control loop never waits
        # without a deadline for an unresponsive ROS operation.
        if publisher is not None:
            try:
                publisher.unregister()
            except Exception:
                pass
        if rospy is not None:
            try:
                rospy.signal_shutdown('relay session ended')
            except Exception:
                pass


_stopping = False


def _request_shutdown(_signum, _frame):
    global _stopping
    _stopping = True


def _nonblocking(fd):
    fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)


def _emit(fd, line):
    try:
        data = (line + '\n').encode('ascii')
        return os.write(fd, data) == len(data)
    except (OSError, IOError):
        return False


def main(max_motion_sec=None, require_intent_deadline=False, forward_navigation=False):
    """Run one session, using inherited pipe stdin and status messages on stdout."""
    global _stopping
    _stopping = False
    RelayGate(0., max_motion_sec, require_intent_deadline, forward_navigation)
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)
    input_fd, output_fd = sys.stdin.fileno(), sys.stdout.fileno()
    if not stat.S_ISFIFO(os.fstat(input_fd).st_mode):
        return 1
    _nonblocking(input_fd)
    _nonblocking(output_fd)
    parent_pid = os.getppid()
    if parent_pid <= 1:
        return 1
    started_at = _monotonic()
    shared = _Shared()
    worker = threading.Thread(target=_ros_worker, args=(shared, parent_pid))
    worker.daemon = True
    worker.start()
    reader = FrameReader()
    gate = None
    ready_sent = False
    fault_sent = False
    now = started_at
    try:
        while not _stopping:
            fault, _, _, _, _, _, _ = shared.snapshot()
            try:
                wait = PERIOD_SEC
                if gate is not None and gate.intent_deadline is not None:
                    wait = min(wait, max(0.0, gate.intent_deadline - _monotonic()))
                inputs, _, _ = select.select([] if fault else [input_fd], [], [], wait)
            except (OSError, select.error) as exc:
                if getattr(exc, 'errno', None) == errno.EINTR or (exc.args and exc.args[0] == errno.EINTR):
                    continue
                shared.stop('pipe_error')
                inputs = []
            if inputs:
                try:
                    chunk = os.read(input_fd, MAX_FRAME_BYTES + 1)
                    if not chunk:
                        shared.stop('parent_eof')
                    elif not ready_sent:
                        shared.stop('startup_input')
                    else:
                        for frame in reader.feed(chunk):
                            now = _monotonic()
                            if not gate.accept(frame, now):
                                shared.stop(gate.reason or 'invalid_frame')
                                break
                            if gate.stopped:
                                shared.stop(gate.reason)
                                break
                            shared.set_command(gate.command(now), now, gate.intent_deadline)
                except (OSError, IOError) as exc:
                    if getattr(exc, 'errno', None) not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR):
                        shared.stop('pipe_error')
                except (ValueError, TypeError):
                    shared.stop('invalid_frame')
            now = _monotonic()
            if os.getppid() != parent_pid:
                shared.stop('parent_lost')
            fault, ready, armed, _, _, progress, _ = shared.snapshot()
            if not ready_sent and not fault and now - started_at >= STARTUP_TIMEOUT_SEC:
                shared.stop('startup_timeout')
            elif ready_sent and not fault:
                gate.command(now)
                if gate.stopped:
                    shared.stop(gate.reason)
                elif shared.worker_stalled(_monotonic):
                    shared.stop('ros_stalled')
            fault, ready, armed, _, _, _, _ = shared.snapshot()
            if not fault and ready and not ready_sent:
                gate = RelayGate(now, max_motion_sec=max_motion_sec,
                                 require_intent_deadline=require_intent_deadline,
                                 forward_navigation=forward_navigation)
                shared.arm(now)
                ready_line = ('READY' if max_motion_sec is None else
                              'READY motion_limit_sec=%.6f' % max_motion_sec)
                if forward_navigation:
                    ready_line = 'READY forward_navigation'
                if _emit(output_fd, ready_line):
                    ready_sent = True
                else:
                    shared.stop('status_pipe')
            fault, _, _, _, _, _, _ = shared.snapshot()
            if fault and not fault_sent:
                prefix = 'STOP ' if fault in ('requested_stop', 'motion_deadline') else 'FAULT '
                _emit(output_fd, prefix + fault)
                fault_sent = True
    except Exception:
        shared.stop('control_error')
        if not fault_sent:
            _emit(output_fd, 'FAULT control_error')
    finally:
        fault = shared.snapshot()[0]
        try:
            shutdown_at = _monotonic()
        except Exception:
            shutdown_at = now
        shared.shutdown(shutdown_at)
        worker.join(SHUTDOWN_JOIN_SEC)
    return 1 if fault not in (None, 'requested_stop') else 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Bounded forward-only LIMO relay')
    parser.add_argument('--max-motion-sec', type=float, default=None)
    parser.add_argument('--require-intent-deadline', action='store_true')
    parser.add_argument('--forward-navigation', action='store_true')
    arguments = parser.parse_args()
    if arguments.forward_navigation and (arguments.require_intent_deadline or arguments.max_motion_sec is not None):
        parser.error('forward navigation and trial deadline flags are mutually exclusive')
    sys.exit(main(arguments.max_motion_sec, arguments.require_intent_deadline, arguments.forward_navigation))
