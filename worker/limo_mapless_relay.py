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
import time

try:
    INTEGER_TYPES = (int, long)  # noqa: F821
except NameError:
    INTEGER_TYPES = (int,)

HEARTBEAT_SEC = 0.2
MAX_LINEAR = 0.12
MAX_ANGULAR = 0.4
MAX_PENDING_BYTES = 2048
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

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)
    rospy.init_node("rosclaw_limo_mapless_relay", anonymous=True, disable_signals=True)
    publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    gate = RelayGate(_monotonic())
    pending = b""
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    try:
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
            message = Twist()
            message.linear.x = linear
            message.angular.z = angular
            publisher.publish(message)
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
