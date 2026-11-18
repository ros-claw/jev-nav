#!/usr/bin/env python2
"""Fixed daemon-owned velocity relay for a future LIMO mapless worker.

This program is not registered with the running robot pack. A command stream
must arrive through its inherited stdin. Only the hardware daemon account can
start the ROS part. Chassis and host power-loss behavior are not established.
"""

import ctypes
import json
import math
import os
import pwd
import select
import signal
import sys
import threading
import time

try:
    INTEGER_TYPES = (int, long)  # noqa: F821
except NameError:
    INTEGER_TYPES = (int,)

HEARTBEAT_SEC = 0.2
MAX_LINEAR = 0.12
MAX_ANGULAR = 0.4
MAX_PENDING_BYTES = 2048
STATUS_MAX_AGE_SEC = 0.25
_shutdown = False


def _number(value):
    if isinstance(value, bool) or not isinstance(value, INTEGER_TYPES + (float,)):
        return False
    try:
        return not math.isnan(value) and not math.isinf(value)
    except (OverflowError, ValueError):
        return False


class RelayGate(object):  # noqa: UP004 - Python 2 ROS worker
    """Require fresh, ordered, bounded commands and latch on any loss."""

    def __init__(self, started_at):
        if not _number(started_at):
            raise ValueError("invalid start time")
        self.last_at = float(started_at)
        self.last_sequence = -1
        self.linear = 0.0
        self.angular = 0.0
        self.stopped = False

    def accept(self, frame, now):
        if self.stopped:
            return False
        valid = (
            isinstance(frame, dict)
            and set(frame) == set(["sequence", "stamp", "linear", "angular"])
            and isinstance(frame.get("sequence"), INTEGER_TYPES)
            and not isinstance(frame.get("sequence"), bool)
            and frame["sequence"] > self.last_sequence
            and _number(now)
            and _number(frame.get("stamp"))
            and _number(frame.get("linear"))
            and _number(frame.get("angular"))
        )
        if valid:
            valid = (
                self.last_at <= now
                and 0 <= now - frame["stamp"] <= HEARTBEAT_SEC
                and now - self.last_at <= HEARTBEAT_SEC
                and abs(frame["linear"]) <= MAX_LINEAR
                and abs(frame["angular"]) <= MAX_ANGULAR
            )
        if not valid:
            self.stopped = True
            return False
        self.last_sequence = frame["sequence"]
        self.last_at = float(now)
        self.linear = float(frame["linear"])
        self.angular = float(frame["angular"])
        return True

    def command(self, now):
        if not _number(now) or now < self.last_at or now - self.last_at > HEARTBEAT_SEC:
            self.stopped = True
        if self.stopped:
            return (0.0, 0.0)
        return (self.linear, self.angular)


class _Timespec(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_long), ("nanoseconds", ctypes.c_long)]


def _monotonic():
    value = _Timespec()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.clock_gettime(1, ctypes.byref(value)) != 0:
        raise OSError(ctypes.get_errno(), "clock_gettime failed")
    return value.seconds + value.nanoseconds / 1e9


def _request_shutdown(_signum, _frame):
    global _shutdown
    _shutdown = True


class DriverStatus(object):  # noqa: UP004 - Python 2 ROS worker
    """Require a fresh status and a completed zero after our first observation."""

    def __init__(self, expected_publisher):
        if not isinstance(expected_publisher, str) or not expected_publisher.startswith("/"):
            raise ValueError("invalid relay publisher name")
        self.lock = threading.Lock()
        self.expected_publisher = expected_publisher
        self.boot_id = None
        self.initial_zero_writes = None
        self.zero_writes = None
        self.mode = None
        self.primed = False
        self.latched = True
        self.received_at = None
        self.generated_at = None
        self.zero_publisher = None
        self.fault = False
        self.accepted_mode = None

    def observe(self, data, now):
        with self.lock:
            try:
                if not isinstance(data, str) or len(data) > 256 or not _number(now):
                    raise ValueError("invalid status")
                fields = data.split()
                if len(fields) != 8 or fields[0] != "v1" or not fields[1]:
                    raise ValueError("invalid status")
                if not all(field.isdigit() for field in fields[2:6]):
                    raise ValueError("invalid status")
                count, mode, primed, latched = [int(value) for value in fields[2:6]]
                generated_at = float(fields[7])
                if (not _number(generated_at) or not 0 <= now - generated_at <= STATUS_MAX_AGE_SEC
                        or (self.generated_at is not None and generated_at < self.generated_at)):
                    raise ValueError("stale or reordered driver status")
                if count > 18446744073709551615 or mode > 255 or primed > 1 or latched > 1:
                    raise ValueError("invalid status")
                if self.received_at is not None and now < self.received_at:
                    raise ValueError("clock moved backwards")
                if latched or (self.accepted_mode is not None and mode != self.accepted_mode):
                    raise ValueError("driver latched or changed motion mode")
                if (self.accepted_mode is not None and self.zero_writes is not None
                        and count > self.zero_writes and fields[6] != self.expected_publisher):
                    raise ValueError("zero command came from another publisher")
                if self.boot_id is None:
                    self.boot_id = fields[1]
                    self.initial_zero_writes = count
                elif fields[1] != self.boot_id or count < self.zero_writes:
                    raise ValueError("driver restarted or counter regressed")
                self.zero_writes = count
                self.mode = mode
                self.primed = primed == 1
                self.latched = latched == 1
                self.received_at = float(now)
                self.generated_at = generated_at
                self.zero_publisher = fields[6]
            except (ValueError, OverflowError):
                self.fault = True

    def ready(self, now):
        with self.lock:
            ready = bool(
                not self.fault and self.received_at is not None and _number(now)
                and 0 <= now - self.received_at <= STATUS_MAX_AGE_SEC
                and 0 <= now - self.generated_at <= STATUS_MAX_AGE_SEC
                and self.zero_writes > self.initial_zero_writes
                and self.zero_publisher == self.expected_publisher
                and self.mode in (0, 1, 2) and self.primed and not self.latched
            )
            if ready and self.accepted_mode is None:
                self.accepted_mode = self.mode
            return ready


def prime_connection(connected, publish_zero, clock, sleep, cancelled, driver_ready, timeout=3.0):
    """Wait for a stable connection and a fresh completed-zero driver report."""
    started = clock()
    previous = started
    stable_since = None
    if not _number(started):
        return False
    while not cancelled():
        now = clock()
        if not _number(now) or now < previous or now - started >= timeout:
            return False
        previous = now
        publish_zero()
        if connected():
            if stable_since is None:
                stable_since = now
            if now - stable_since >= 0.2 and driver_ready():
                return True
        else:
            stable_since = None
        sleep(0.05)
    return False


def main():
    # This executable belongs to the daemon-side adapter, not the agent.
    try:
        daemon_uid = pwd.getpwnam("rosclaw-hw").pw_uid
    except KeyError:
        return 1
    if os.geteuid() != daemon_uid:
        return 1

    import rospy
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)
    rospy.init_node("rosclaw_limo_mapless_relay", anonymous=True, disable_signals=True)
    publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    status = DriverStatus(rospy.get_name())
    _status_subscriber = rospy.Subscriber(
        "/rosclaw_limo/command_watchdog", String,
        lambda msg: status.observe(msg.data, _monotonic()), queue_size=1)
    pending = b""
    motion_acknowledged = False
    try:
        # The parent waits for READY before sending commands. A readable pipe
        # here means it closed or violated that protocol, so cancel startup.
        def cancelled():
            return (_shutdown or rospy.is_shutdown() or
                    bool(select.select([sys.stdin.fileno()], [], [], 0.0)[0]))

        if not prime_connection(lambda: publisher.get_num_connections() > 0,
                                lambda: publisher.publish(Twist()),
                                _monotonic, time.sleep, cancelled,
                                lambda: status.ready(_monotonic())):
            return 1
        gate = RelayGate(_monotonic())
        sys.stdout.write("READY\n")
        sys.stdout.flush()
        while not _shutdown and not rospy.is_shutdown():
            ready, _, _ = select.select([sys.stdin.fileno()], [], [], 0.05)
            if ready:
                chunk = os.read(sys.stdin.fileno(), MAX_PENDING_BYTES)
                if not chunk:
                    break
                pending += chunk
                if len(pending) > MAX_PENDING_BYTES:
                    break
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    try:
                        frame = json.loads(line.decode("utf-8"))
                    except (ValueError, UnicodeError):
                        gate.stopped = True
                        break
                    if not gate.accept(frame, _monotonic()):
                        break
            linear, angular = gate.command(_monotonic())
            if publisher.get_num_connections() <= 0 or not status.ready(_monotonic()):
                gate.stopped = True
                linear, angular = 0.0, 0.0
            message = Twist()
            message.linear.x = linear
            message.angular.z = angular
            publisher.publish(message)
            if not motion_acknowledged and (linear or angular) and publisher.get_num_connections() > 0:
                sys.stdout.write("MOTION_ACK {}\n".format(gate.last_sequence))  # noqa: UP032 - Python 2
                sys.stdout.flush()
                motion_acknowledged = True
            if gate.stopped:
                break
    finally:
        # Continue publishing zero briefly even if the producer disappears.
        for _ in range(10):
            try:
                publisher.publish(Twist())
                time.sleep(0.05)
            except Exception:  # noqa: BLE001 - shutdown is best effort
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
