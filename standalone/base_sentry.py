#!/usr/bin/env python2
"""PTY-only model of a parent-owned, zero-only LIMO emergency sentry.

This fixture creates its own PTY and an inert child; it cannot select or open a
physical serial device, and it has no live-driver or ROS integration. It is not
a stopping guarantee. Real operation would need a driver that inherits the
parent's FD, an authenticated heartbeat, provable ownership of all writers,
verified UART/controller behavior, and independently tested physical stopping.
Neither this process nor an on-host watchdog survives host failure or power loss.
"""

from __future__ import division

import errno
import fcntl
import math
import os
import pty
import select
import signal
import stat
import sys
import termios
import threading
import time
import tty

try:
    NUMBER_TYPES = (int, long, float)  # noqa: F821 - ROS Melodic Python 2
except NameError:
    NUMBER_TYPES = (int, float)

RESET_PADDING = bytes(bytearray([0] * 14))
ZERO_FRAME = bytes(bytearray([0x55, 0x0e, 0x01, 0x11] + [0] * 10))


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _waitpid(pid, flags):
    return os.waitpid(pid, flags)


def _signal_child(pid, sig):
    return os.kill(pid, sig)


class OfflinePtySentry(object):  # noqa: UP004 - ROS Melodic Python 2
    """Test a direct child and fixed zero output on a newly created PTY.

    The fake child inherits the parent's slave FD at fork and never forks or
    writes serial data. heartbeat() simulates a trusted timestamped source; no
    real heartbeat transport is implemented. tick() never waits for the child:
    timeout sends TERM then KILL, and writes start only after waitpid confirms
    this direct child exited. Concurrent tick calls are serialized.
    """

    def __init__(self, clock, heartbeat_timeout, kill_grace, zero_period):
        if (not callable(clock) or not all(_finite(x) for x in
                (heartbeat_timeout, kill_grace, zero_period))
                or heartbeat_timeout <= 0 or kill_grace <= 0 or zero_period <= 0):
            raise ValueError("invalid sentry timing")
        self._clock = clock
        self.heartbeat_timeout = heartbeat_timeout
        self.kill_grace = kill_grace
        self.zero_period = zero_period
        self._lock = threading.RLock()
        self.state = "new"
        self.fault_reason = None
        self.reaped = False
        self.child_pid = None
        self.master_fd = None
        self._slave_fd = None
        self._master_identity = None
        self._slave_identity = None
        self._last_now = None
        self._last_heartbeat = None
        self._term_at = None
        self._kill_sent = False
        self._last_zero_at = None

    def _fault(self, reason):
        self.state = "fault"
        if self.fault_reason is None:
            self.fault_reason = reason
        return "fault"

    def _now(self):
        try:
            now = self._clock()
            if not _finite(now) or now < 0 or (self._last_now is not None and now < self._last_now):
                self._fault("invalid_clock")
                return None
            self._last_now = now
            return now
        except Exception:
            self._fault("invalid_clock")
            return None

    def start_fixture(self, ignore_term=False, exit_immediately=False):
        """Create the private PTY and one inert direct child, never a real driver."""
        with self._lock:
            if self.state != "new" or type(ignore_term) is not bool or type(exit_immediately) is not bool:
                raise ValueError("invalid fixture start")
            if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
                raise RuntimeError("fixture requires default SIGCHLD handling")
            now = self._now()
            if now is None:
                raise ValueError("invalid fixture clock")
            ready_r = ready_w = None
            try:
                self.master_fd, self._slave_fd = pty.openpty()
                master_stat, slave_stat = os.fstat(self.master_fd), os.fstat(self._slave_fd)
                if (not stat.S_ISCHR(master_stat.st_mode) or not stat.S_ISCHR(slave_stat.st_mode)
                        or (os.major(master_stat.st_rdev), os.minor(master_stat.st_rdev)) != (5, 2)
                        or not 136 <= os.major(slave_stat.st_rdev) <= 143
                        or not os.ttyname(self._slave_fd).startswith("/dev/pts/")):
                    raise RuntimeError("fixture accepts only a freshly created PTY")
                self._master_identity = (master_stat.st_dev, master_stat.st_ino, master_stat.st_rdev)
                self._slave_identity = (slave_stat.st_dev, slave_stat.st_ino, slave_stat.st_rdev)
                tty.setraw(self._slave_fd, termios.TCSANOW)
                attrs = termios.tcgetattr(self._slave_fd)
                attrs[2] &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB)
                attrs[2] |= termios.CS8 | termios.CLOCAL | termios.CREAD
                attrs[4] = attrs[5] = termios.B460800
                termios.tcsetattr(self._slave_fd, termios.TCSANOW, attrs)
                fcntl.ioctl(self._slave_fd, termios.TIOCEXCL)
                for fd in (self.master_fd, self._slave_fd):
                    fcntl.fcntl(fd, fcntl.F_SETFD, fcntl.fcntl(fd, fcntl.F_GETFD) | fcntl.FD_CLOEXEC)
                    fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
                ready_r, ready_w = os.pipe()
                pid = os.fork()
                if pid == 0:
                    try:
                        os.close(self.master_fd)
                        os.close(ready_r)
                        if ignore_term:
                            signal.signal(signal.SIGTERM, signal.SIG_IGN)
                        os.write(ready_w, b"R")
                        os.close(ready_w)
                        if exit_immediately:
                            os._exit(0)
                        while True:
                            time.sleep(60)
                    except BaseException:
                        os._exit(1)
                self.child_pid = pid
                os.close(ready_w)
                ready_w = None
                if not select.select([ready_r], [], [], 2.0)[0] or os.read(ready_r, 1) != b"R":
                    raise RuntimeError("fixture child failed to start")
                os.close(ready_r)
                ready_r = None
                self._last_heartbeat = now
                self.state = "running"
            except Exception:
                if ready_r is not None:
                    os.close(ready_r)
                if ready_w is not None:
                    os.close(ready_w)
                self.close()
                raise

    def heartbeat(self, stamp):
        """Accept a simulated fresh heartbeat; never an authenticated live signal."""
        with self._lock:
            if self.state != "running":
                return False
            now = self._now()
            if (now is None or now - self._last_heartbeat >= self.heartbeat_timeout
                    or not _finite(stamp) or stamp <= self._last_heartbeat
                    or not 0 <= now - stamp <= self.heartbeat_timeout):
                return False
            self._last_heartbeat = stamp
            return True

    def _write_fixed(self, padding):
        # No frame or path is accepted from the caller. The only target is the
        # slave FD returned by this object's own pty.openpty().
        payload = bytes(bytearray([0] * 14)) if padding else bytes(bytearray(
            [0x55, 0x0e, 0x01, 0x11] + [0] * 10))
        # Duplicate atomically with CLOEXEC so a close/reuse of the original
        # descriptor cannot redirect this particular write after verification.
        duplicate = fcntl.fcntl(self._slave_fd, getattr(fcntl, "F_DUPFD_CLOEXEC", 1030), 3)
        try:
            info = os.fstat(duplicate)
            if ((info.st_dev, info.st_ino, info.st_rdev) != self._slave_identity
                    or not stat.S_ISCHR(info.st_mode) or not 136 <= os.major(info.st_rdev) <= 143
                    or not os.isatty(duplicate)):
                raise OSError(errno.EBADF, "fixture PTY descriptor changed")
            return os.write(duplicate, payload)
        finally:
            os.close(duplicate)

    def _is_own_slave(self, fd):
        info = os.fstat(fd)
        return ((info.st_dev, info.st_ino, info.st_rdev) == self._slave_identity
                and stat.S_ISCHR(info.st_mode) and 136 <= os.major(info.st_rdev) <= 143
                and os.isatty(fd))

    def _write_complete(self, padding):
        try:
            return self._write_fixed(padding) == 14
        except (OSError, IOError):
            return False

    def tick(self):
        with self._lock:
            if self.state in ("new", "fault", "closed"):
                return self.state
            now = self._now()
            if now is None:
                return "fault"
            if self.state == "stopping":
                if not self.reaped:
                    return self._fault("child_not_reaped")
                if self._last_zero_at is None or now - self._last_zero_at >= self.zero_period:
                    if not self._write_complete(False):
                        return self._fault("short_or_failed_zero_write")
                    self._last_zero_at = now
                return self.state
            # A dedicated parent with default SIGCHLD and a single waitpid
            # consumer is required; lost parenthood never authorizes a signal.
            if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
                return self._fault("sigchld_not_default")
            try:
                pid, status = _waitpid(self.child_pid, os.WNOHANG)
            except (OSError, IOError):
                return self._fault("child_identity_unavailable")
            if pid == self.child_pid:
                if not (os.WIFEXITED(status) or os.WIFSIGNALED(status)):
                    return self._fault("child_exit_unverified")
                self.reaped = True
                self.state = "stopping"
                if not self._write_complete(True):
                    return self._fault("short_or_failed_padding_write")
                if not self._write_complete(False):
                    return self._fault("short_or_failed_zero_write")
                self._last_zero_at = now
                return self.state
            if pid != 0:
                return self._fault("unexpected_child_identity")
            if self.state == "running" and now - self._last_heartbeat >= self.heartbeat_timeout:
                try:
                    _signal_child(self.child_pid, signal.SIGTERM)
                except OSError as exc:
                    if exc.errno != errno.ESRCH:
                        return self._fault("terminate_failed")
                self._term_at = now
                self.state = "terminating"
            elif (self.state == "terminating" and not self._kill_sent
                  and now - self._term_at >= self.kill_grace):
                try:
                    _signal_child(self.child_pid, signal.SIGKILL)
                except OSError as exc:
                    if exc.errno != errno.ESRCH:
                        return self._fault("kill_failed")
                self._kill_sent = True
            return self.state

    def close(self):
        with self._lock:
            if self.child_pid is not None and not self.reaped:
                try:
                    pid, _status = os.waitpid(self.child_pid, os.WNOHANG)
                    if pid == 0:
                        os.kill(self.child_pid, signal.SIGKILL)
                        deadline = time.time() + 1.0
                        while time.time() < deadline:
                            pid, _status = os.waitpid(self.child_pid, os.WNOHANG)
                            if pid == self.child_pid:
                                break
                            time.sleep(0.005)
                    if pid == self.child_pid:
                        self.reaped = True
                except (OSError, IOError):
                    pass
            if self._slave_fd is not None:
                duplicate = None
                try:
                    duplicate = fcntl.fcntl(self._slave_fd, getattr(fcntl, "F_DUPFD_CLOEXEC", 1030), 3)
                    if self._is_own_slave(duplicate):
                        try:
                            fcntl.ioctl(duplicate, termios.TIOCNXCL)
                        except (OSError, IOError):
                            pass
                        if self._is_own_slave(self._slave_fd):
                            os.close(self._slave_fd)
                except (OSError, IOError):
                    pass
                finally:
                    if duplicate is not None:
                        os.close(duplicate)
                self._slave_fd = None
            if self.master_fd is not None:
                try:
                    info = os.fstat(self.master_fd)
                    if (stat.S_ISCHR(info.st_mode)
                            and (info.st_dev, info.st_ino, info.st_rdev) == self._master_identity):
                        os.close(self.master_fd)
                except (OSError, IOError):
                    pass
                self.master_fd = None
            self.state = "closed"


if __name__ == "__main__":
    sys.stderr.write("PTY fixture only; no physical serial support or deployment.\n")
    sys.exit(2)
