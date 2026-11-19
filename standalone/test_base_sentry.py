"""Offline PTY tests; these do not verify hardware, firmware, or exclusive ownership."""

import errno
import os
import select
import termios
import threading
import time

import pytest


def module():
    try:
        import base_sentry
    except ImportError:
        pytest.fail("offline base sentry is not implemented", pytrace=False)
    return base_sentry


class Clock(object):
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


def wire(sentry):
    chunks = []
    while select.select([sentry.master_fd], [], [], 0)[0]:
        try:
            chunk = os.read(sentry.master_fd, 65536)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EIO):
                break
            raise
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def wait_stopping(sentry):
    deadline = time.time() + 2.0
    while time.time() < deadline:
        state = sentry.tick()
        if state in ("stopping", "fault"):
            return state
        time.sleep(0.005)
    pytest.fail("fixture child was not reaped in time", pytrace=False)


class SourceParser(object):
    """Reproduce only the supplied limo_driver.cpp receive state machine."""

    def __init__(self):
        self.state = 0
        self.fid = 0
        self.data = []
        self.checksum = 0

    def feed(self, wire_bytes):
        result = []
        for value in bytearray(wire_bytes):
            if self.state == 0:
                if value == 0x55:
                    self.state = 1
            elif self.state == 1:
                self.state = 2 if value == 14 else 0
            elif self.state == 2:
                self.fid = value << 8
                self.state = 3
            elif self.state == 3:
                self.fid |= value
                self.data = []
                self.checksum = 0
                self.state = 4
            elif self.state == 4:
                if len(self.data) < 8:
                    self.data.append(value)
                    self.checksum = (self.checksum + value) & 0xff
                else:
                    self.state = 5
            else:
                if value == self.checksum:
                    result.append((self.fid, tuple(self.data)))
                self.state = 0
        return result


def command(speed):
    return bytes(bytearray([0x55, 14, 1, 0x11, 0, speed, 0, 0, 0, 0, 0, 0, 0, speed]))


def test_zero_frame_matches_source_protocol():
    sentry = module()
    assert bytes(bytearray([0x55, 14, 1, 0x11] + [0] * 10)) == sentry.ZERO_FRAME
    assert bytes(bytearray([0] * 14)) == sentry.RESET_PADDING


def test_partial_motion_plus_unpadded_zero_can_complete_nonzero_source_model_frame():
    sentry = module()
    parser = SourceParser()
    parser.feed(command(14)[:12])
    frames = parser.feed(sentry.ZERO_FRAME)
    assert (0x111, (0, 14, 0, 0, 0, 0, 0, 0)) in frames


def test_padding_then_zero_for_all_old_frame_truncations_and_positive_speeds():
    sentry = module()
    for speed in range(1, 51):
        prior = command(speed)
        for length in range(15):
            parser = SourceParser()
            parser.feed(prior[:length])
            padding_frames = parser.feed(sentry.RESET_PADDING)
            assert not any(fid == 0x111 and any(data) for fid, data in padding_frames)
            frames = parser.feed(sentry.ZERO_FRAME * 3)
            assert frames == [(0x111, (0,) * 8)] * 3


def test_alive_child_and_fresh_heartbeat_do_not_write_any_bytes():
    sentry_module = module()
    clock = Clock()
    sentry = sentry_module.OfflinePtySentry(clock, 0.2, 0.1, 0.05)
    try:
        sentry.start_fixture(ignore_term=True)
        clock.now = 10.1
        assert sentry.heartbeat(10.1)
        assert sentry.tick() == "running"
        assert wire(sentry) == b""
    finally:
        sentry.close()


def test_fixture_owns_only_a_pty_with_matching_serial_parameters():
    sentry = module().OfflinePtySentry(Clock(), 0.2, 0.1, 0.05)
    try:
        sentry.start_fixture(ignore_term=True)
        assert os.ttyname(sentry._slave_fd).startswith("/dev/pts/")
        attrs = termios.tcgetattr(sentry._slave_fd)
        assert attrs[4] == termios.B460800 and attrs[5] == termios.B460800
        assert attrs[2] & termios.CSIZE == termios.CS8
        assert not attrs[2] & termios.PARENB and not attrs[2] & termios.CSTOPB
        with pytest.raises(OSError):
            os.open(os.ttyname(sentry._slave_fd), os.O_RDWR | os.O_NOCTTY)
    finally:
        sentry.close()


def test_a_leaked_duplicate_can_still_write_after_direct_child_exit():
    sentry_module = module()
    sentry = sentry_module.OfflinePtySentry(Clock(), 0.2, 0.1, 0.05)
    duplicate = None
    try:
        sentry.start_fixture(exit_immediately=True)
        duplicate = os.dup(sentry._slave_fd)
        assert wait_stopping(sentry) == "stopping" and sentry.reaped
        assert wire(sentry) == sentry_module.RESET_PADDING + sentry_module.ZERO_FRAME
        assert os.write(duplicate, bytes(bytearray([0]))) == 1
        assert wire(sentry) == bytes(bytearray([0]))
    finally:
        if duplicate is not None:
            os.close(duplicate)
        sentry.close()


def test_heartbeat_timeout_waits_for_real_child_reaping_before_padding_or_zero():
    sentry_module = module()
    clock = Clock()
    sentry = sentry_module.OfflinePtySentry(clock, 0.2, 0.1, 0.05)
    try:
        sentry.start_fixture(ignore_term=True)
        clock.now = 10.21
        assert sentry.tick() == "terminating"
        assert wire(sentry) == b""
        clock.now = 10.32
        assert sentry.tick() == "terminating"
        assert wire(sentry) == b""
        assert wait_stopping(sentry) == "stopping"
        assert sentry.reaped is True
        assert wire(sentry) == sentry_module.RESET_PADDING + sentry_module.ZERO_FRAME
    finally:
        sentry.close()


def test_delayed_heartbeat_cannot_refresh_an_expired_watchdog():
    sentry_module = module()
    clock = Clock()
    sentry = sentry_module.OfflinePtySentry(clock, 0.2, 0.1, 0.05)
    try:
        sentry.start_fixture(ignore_term=True)
        clock.now = 10.3
        assert not sentry.heartbeat(10.01)
        assert sentry.tick() == "terminating"
        assert wire(sentry) == b""
    finally:
        sentry.close()


def test_new_heartbeat_cannot_resurrect_an_already_expired_watchdog():
    sentry_module = module()
    clock = Clock()
    sentry = sentry_module.OfflinePtySentry(clock, 0.2, 0.1, 0.05)
    try:
        sentry.start_fixture(ignore_term=True)
        clock.now = 10.3
        assert not sentry.heartbeat(10.29)
        assert sentry.tick() == "terminating"
        assert wire(sentry) == b""
    finally:
        sentry.close()


def test_clock_regression_faults_without_issuing_a_command():
    sentry_module = module()
    clock = Clock()
    sentry = sentry_module.OfflinePtySentry(clock, 0.2, 0.1, 0.05)
    try:
        sentry.start_fixture(ignore_term=True)
        clock.now = 9.9
        assert sentry.tick() == "fault"
        assert wire(sentry) == b""
    finally:
        sentry.close()


def test_normal_child_exit_triggers_padding_and_repeated_fixed_zero_frames():
    sentry_module = module()
    clock = Clock()
    sentry = sentry_module.OfflinePtySentry(clock, 0.2, 0.1, 0.05)
    try:
        sentry.start_fixture(exit_immediately=True)
        assert wait_stopping(sentry) == "stopping"
        assert wire(sentry) == sentry_module.RESET_PADDING + sentry_module.ZERO_FRAME
        clock.now += 0.06
        assert sentry.tick() == "stopping"
        clock.now += 0.06
        assert sentry.tick() == "stopping"
        assert wire(sentry) == sentry_module.ZERO_FRAME * 2
    finally:
        sentry.close()


def test_pid_reuse_or_foreign_reaping_never_signals_or_writes(monkeypatch):
    sentry_module = module()
    clock = Clock()
    sentry = sentry_module.OfflinePtySentry(clock, 0.2, 0.1, 0.05)
    sentry.start_fixture(ignore_term=True)
    signals = []
    def lost_child(_pid, _flags):
        raise OSError(errno.ECHILD, "not our child")
    monkeypatch.setattr(sentry_module, "_waitpid", lost_child)
    monkeypatch.setattr(sentry_module, "_signal_child", lambda pid, sig: signals.append((pid, sig)))
    try:
        clock.now = 10.21
        assert sentry.tick() == "fault"
        assert signals == []
        assert wire(sentry) == b""
    finally:
        monkeypatch.undo()
        sentry.close()


def test_concurrent_ticks_serialize_and_send_one_frame_per_period():
    sentry_module = module()
    clock = Clock()
    sentry = sentry_module.OfflinePtySentry(clock, 0.2, 0.1, 0.05)
    try:
        sentry.start_fixture(exit_immediately=True)
        assert wait_stopping(sentry) == "stopping"
        wire(sentry)
        clock.now += 0.06
        threads = [threading.Thread(target=sentry.tick) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert wire(sentry) == sentry_module.ZERO_FRAME
    finally:
        sentry.close()


def test_reused_sentry_descriptor_is_detected_before_a_zero_write():
    sentry_module = module()
    clock = Clock()
    sentry = sentry_module.OfflinePtySentry(clock, 0.2, 0.1, 0.05)
    replacement = None
    try:
        sentry.start_fixture(exit_immediately=True)
        assert wait_stopping(sentry) == "stopping"
        wire(sentry)
        replacement = os.open("/dev/null", os.O_WRONLY)
        os.dup2(replacement, sentry._slave_fd)
        clock.now += 0.06
        assert sentry.tick() == "fault"
    finally:
        if replacement is not None:
            os.close(replacement)
        sentry.close()


def test_close_does_not_apply_tty_ioctl_to_a_reused_descriptor(monkeypatch):
    sentry_module = module()
    sentry = sentry_module.OfflinePtySentry(Clock(), 0.2, 0.1, 0.05)
    sentry.start_fixture(ignore_term=True)
    old_number = sentry._slave_fd
    replacement = os.open("/dev/null", os.O_WRONLY)
    os.dup2(replacement, old_number)
    calls = []
    real_ioctl = sentry_module.fcntl.ioctl
    def record_ioctl(fd, operation, *args):
        if operation == termios.TIOCNXCL:
            calls.append(fd)
        return real_ioctl(fd, operation, *args)
    monkeypatch.setattr(sentry_module.fcntl, "ioctl", record_ioctl)
    try:
        sentry.close()
        assert calls == []
    finally:
        monkeypatch.undo()
        os.close(replacement)
        try:
            os.close(old_number)
        except OSError:
            pass


def test_short_zero_write_is_a_latched_fault_and_is_never_retried(monkeypatch):
    sentry_module = module()
    clock = Clock()
    sentry = sentry_module.OfflinePtySentry(clock, 0.2, 0.1, 0.05)
    sentry.start_fixture(exit_immediately=True)
    real_write = sentry._write_fixed
    def partial(padding):
        return real_write(True) if padding else os.write(sentry._slave_fd, sentry_module.ZERO_FRAME[:5])
    monkeypatch.setattr(sentry, "_write_fixed", partial)
    try:
        assert wait_stopping(sentry) == "fault"
        assert wire(sentry) == sentry_module.RESET_PADDING + sentry_module.ZERO_FRAME[:5]
        clock.now += 1.0
        assert sentry.tick() == "fault"
        assert wire(sentry) == b""
    finally:
        monkeypatch.undo()
        sentry.close()


def test_short_padding_is_a_latched_fault_without_a_following_zero_frame(monkeypatch):
    sentry_module = module()
    clock = Clock()
    sentry = sentry_module.OfflinePtySentry(clock, 0.2, 0.1, 0.05)
    sentry.start_fixture(exit_immediately=True)
    monkeypatch.setattr(sentry, "_write_fixed", lambda padding: os.write(sentry._slave_fd,
                                                                        sentry_module.RESET_PADDING[:4]))
    try:
        assert wait_stopping(sentry) == "fault"
        assert wire(sentry) == sentry_module.RESET_PADDING[:4]
        clock.now += 1.0
        assert sentry.tick() == "fault"
        assert wire(sentry) == b""
    finally:
        monkeypatch.undo()
        sentry.close()
