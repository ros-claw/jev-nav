import importlib.util
import json
import os
import select
import subprocess
import time
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "worker" / "limo_mapless_relay.py"
spec = importlib.util.spec_from_file_location("limo_mapless_relay", path)
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


def frame(seq, stamp, linear=0.1, angular=0.0):
    return {"sequence": seq, "stamp": stamp, "linear": linear, "angular": angular}


def test_commands_expire_and_cannot_resume_after_heartbeat_loss():
    gate = relay.RelayGate(started_at=10.0)
    assert gate.accept(frame(1, 10.0), now=10.0)
    assert gate.command(now=10.1) == (0.1, 0.0)
    assert gate.command(now=10.21) == (0.0, 0.0)
    assert not gate.accept(frame(2, 10.22), now=10.22)


def test_replayed_or_out_of_bounds_commands_latch_zero():
    for invalid in (
        frame(1, 10.0),
        frame(2, 10.01, linear=0.13),
        frame(2, 10.01, angular=float("nan")),
        frame(2, 11.0),
    ):
        gate = relay.RelayGate(started_at=10.0)
        assert gate.accept(frame(1, 10.0), now=10.0)
        assert not gate.accept(invalid, now=10.01)
        assert gate.command(now=10.01) == (0.0, 0.0)


def test_startup_without_control_never_commands_motion():
    gate = relay.RelayGate(started_at=10.0)
    assert gate.command(now=10.1) == (0.0, 0.0)
    assert gate.command(now=10.3) == (0.0, 0.0)
    assert not gate.accept(frame(1, 10.3), now=10.3)


def test_startup_sends_zero_during_a_stable_subscriber_connection():
    prime = getattr(relay, "prime_connection", None)
    assert callable(prime), "relay startup priming is unavailable"
    current = [10.0]
    writes = []
    connections = iter((0, 1, 1, 0, 1, 1, 1, 1, 1, 1))

    def connected():
        return next(connections, 1) > 0

    def sleep(seconds):
        current[0] += seconds

    assert prime(connected, lambda: writes.append(current[0]), lambda: current[0], sleep,
                 lambda: False, lambda: True, timeout=1.0)
    assert len(writes) >= 6
    assert current[0] >= 10.35


def test_startup_fails_without_a_connected_subscriber():
    prime = getattr(relay, "prime_connection", None)
    assert callable(prime), "relay startup priming is unavailable"
    current = [10.0]
    assert not prime(lambda: False, lambda: None, lambda: current[0],
                     lambda seconds: current.__setitem__(0, current[0] + seconds),
                     lambda: False, lambda: True, timeout=.2)


def test_startup_stops_on_cancel_or_clock_regression():
    writes = []
    assert not relay.prime_connection(lambda: True, lambda: writes.append(1),
                                      lambda: 10.0, lambda seconds: None, lambda: True, lambda: True)
    assert writes == []
    times = iter((10.0, 10.0, 9.9))
    assert not relay.prime_connection(lambda: True, lambda: writes.append(1),
                                      lambda: next(times), lambda seconds: None, lambda: False,
                                      lambda: True)
    assert writes == [1]


def test_startup_requires_driver_confirmation_even_with_a_subscriber():
    current = [10.0]
    assert not relay.prime_connection(
        lambda: True, lambda: None, lambda: current[0],
        lambda seconds: current.__setitem__(0, current[0] + seconds),
        lambda: False, lambda: False, timeout=.5)


def test_driver_status_requires_a_new_written_zero_and_fresh_valid_status():
    status = relay.DriverStatus("/relay")
    assert not status.ready(10.0)
    status.observe("v2 boot_1 7 0 1 0 /relay 10.0 10.0", 10.0)
    assert not status.ready(10.01)
    status.observe("v2 boot_1 8 0 1 0 /relay 10.05 10.05", 10.05)
    assert status.ready(10.1)
    assert not status.ready(10.31)
    status.observe("v2 boot_1 8 0 1 1 /relay 10.32 10.32", 10.32)
    assert not status.ready(10.32)
    status.observe("v2 boot_1 9 0 1 0 /relay 10.33 10.33", 10.33)
    assert not status.ready(10.33)


def test_driver_status_latches_on_restart_or_counter_rollback():
    for next_frame in ("v2 boot_2 1 0 1 0 /relay 10.1 10.1", "v2 boot_1 6 0 1 0 /relay 10.1 10.1"):
        status = relay.DriverStatus("/relay")
        status.observe("v2 boot_1 7 0 1 0 /relay 10.0 10.0", 10.0)
        status.observe("v2 boot_1 8 0 1 0 /relay 10.05 10.05", 10.05)
        assert status.ready(10.05)
        status.observe(next_frame, 10.1)
        status.observe("v2 boot_1 9 0 1 0 /relay 10.15 10.15", 10.15)
        assert not status.ready(10.15)


def test_driver_status_rejects_malformed_and_unsupported_mode():
    status = relay.DriverStatus("/relay")
    status.observe("v2 boot_1 0 0 1 0 /relay 1.0 1.0", 1.0)
    status.observe("v2 boot_1 1 255 1 0 /relay 1.1 1.1", 1.1)
    assert not status.ready(1.1)
    status.observe("not a driver status", 1.15)
    assert not status.ready(1.15)


def test_driver_status_latches_if_mode_changes_after_readiness():
    status = relay.DriverStatus("/relay")
    status.observe("v2 boot_1 7 0 1 0 /relay 10.0 10.0", 10.0)
    status.observe("v2 boot_1 8 0 1 0 /relay 10.05 10.05", 10.05)
    assert status.ready(10.05)
    status.observe("v2 boot_1 8 1 1 0 /relay 10.1 10.1", 10.1)
    status.observe("v2 boot_1 9 0 1 0 /relay 10.15 10.15", 10.15)
    assert not status.ready(10.15)


def test_driver_status_rejects_old_future_and_reordered_reports():
    for bad_frame, received in (("v2 boot_1 8 0 1 0 /relay 9.0 9.0", 10.1),
                                ("v2 boot_1 8 0 1 0 /relay 11.0 11.0", 10.1),
                                ("v2 boot_1 8 0 1 0 /relay 9.99 9.99", 10.1)):
        status = relay.DriverStatus("/relay")
        status.observe("v2 boot_1 7 0 1 0 /relay 10.0 10.0", 10.0)
        status.observe(bad_frame, received)
        status.observe("v2 boot_1 9 0 1 0 /relay 10.15 10.15", 10.15)
        assert not status.ready(10.15)


def test_driver_status_zero_must_come_from_this_relay():
    status = relay.DriverStatus(expected_publisher="/relay_one")
    status.observe("v2 boot_1 7 0 1 0 /another_publisher 10.0 10.0", 10.0)
    status.observe("v2 boot_1 8 0 1 0 /another_publisher 10.05 10.05", 10.05)
    assert not status.ready(10.05)
    status.observe("v2 boot_1 9 0 1 0 /relay_one 10.1 10.1", 10.1)
    assert status.ready(10.1)
    status.observe("v2 boot_1 10 0 1 0 /another_publisher 10.15 10.15", 10.15)
    assert not status.ready(10.15)


def test_driver_status_accepts_the_maximum_supported_publisher_name():
    publisher = "/" + "a" * 79
    status = relay.DriverStatus(publisher)
    for count, now in ((100, 123456.0), (101, 123456.1)):
        status.observe(f"v2 123456789012345_12345 {count} 0 1 0 {publisher} {now:.6f} {now:.6f}", now)
    assert status.ready(123456.1)


def test_stop_frame_is_a_barrier_until_driver_reports_a_new_zero_write():
    status = relay.DriverStatus("/relay")
    status.observe("v2 boot_1 7 0 1 0 /relay 9.9 10.0", 10.0)
    status.observe("v2 boot_1 8 0 1 0 /relay 10.0 10.05", 10.05)
    barrier = relay.ZeroBarrier()
    assert barrier.begin(status, 10.05)
    assert barrier.command(0.1, 0.0, status, 10.06) == (0.0, 0.0)
    # A delayed report of a zero written before this barrier cannot release it.
    status.observe("v2 boot_1 9 0 1 0 /relay 10.04 10.08", 10.08)
    assert barrier.command(0.1, 0.0, status, 10.08) == (0.0, 0.0)
    status.observe("v2 boot_1 10 0 1 0 /relay 10.1 10.1", 10.1)
    assert barrier.command(0.1, 0.0, status, 10.1) == (0.1, 0.0)


def test_stop_barrier_times_out_if_driver_never_reports_a_new_zero():
    status = relay.DriverStatus("/relay")
    status.observe("v2 boot_1 7 0 1 0 /relay 10.0 10.0", 10.0)
    status.observe("v2 boot_1 8 0 1 0 /relay 10.05 10.05", 10.05)
    barrier = relay.ZeroBarrier()
    assert barrier.begin(status, 10.05)
    status.observe("v2 boot_1 8 0 1 0 /relay 10.05 10.6", 10.6)
    assert barrier.command(0.1, 0.0, status, 10.6) == (0.0, 0.0)
    assert barrier.timed_out


@pytest.mark.parametrize(("confirm_driver", "mixed_frames", "latch_after_motion", "real_queue", "fail_subscribe", "fail_publisher"),
                         [(True, False, False, False, False, False), (False, False, False, False, False, False),
                          (True, True, False, False, False, False), (True, False, True, False, False, False),
                          (True, False, False, True, False, False), (True, False, False, True, True, False),
                          (True, False, False, True, False, True)])
def test_relay_process_emits_zero_when_control_pipe_closes(tmp_path, confirm_driver, mixed_frames,
                                                         latch_after_motion, real_queue, fail_subscribe, fail_publisher):
    output = tmp_path / "commands.jsonl"
    fixture = tmp_path / "fake_ros.py"
    fixture.write_text("""
import imp, json, os, sys, types
sys.path.insert(0, "/opt/ros/melodic/lib/python2.7/dist-packages")
from rospy.impl.tcpros_pubsub import QueuedConnection
m = imp.load_source("relay", sys.argv[1])
m.os.geteuid = lambda: 999
m.pwd.getpwnam = lambda name: type("User", (), {"pw_uid": 999})()
class Twist(object):
    def __init__(self):
        self.linear = type("Vector", (), {"x": 0.0})()
        self.angular = type("Vector", (), {"z": 0.0})()
status = {"callback": None, "zeros": 0, "zero_at": 0.0}
class String(object): pass
class Connection(object):
    def __init__(self): self.done = False; self.callback = None
    def set_cleanup_callback(self, callback): self.callback = callback
    def write_data(self, data): return True
    def close(self):
        self.done = True
        if self.callback: self.callback(self)
connection = Connection()
queue = QueuedConnection(connection, 1) if os.environ.get("REAL_QUEUE") == "1" else None
class Publisher(object):
    def __init__(self, *args, **kwargs):
        if os.environ.get("FAIL_PUBLISHER") == "1": raise RuntimeError("publisher failed")
    def unregister(self): connection.close()
    def get_num_connections(self): return 1
    def publish(self, message):
        with open(sys.argv[2], "a") as output:
            output.write(json.dumps([message.linear.x, message.angular.z]) + "\\n")
        if not message.linear.x and not message.angular.z:
            status["zeros"] += 1
            status["zero_at"] = m._monotonic()
        if status["callback"] and os.environ.get("CONFIRM_DRIVER") == "1":
            msg = String()
            msg.data = "v2 boot_1 %s 0 1 0 /relay %s %s" % (status["zeros"], status["zero_at"], m._monotonic())
            status["callback"](msg)
def subscriber(topic, message, callback, **kwargs):
    if os.environ.get("FAIL_SUBSCRIBE") == "1": raise RuntimeError("subscribe failed")
    status["callback"] = callback
    return object()
ros = types.ModuleType("rospy")
ros.init_node = lambda *args, **kwargs: None
ros.is_shutdown = lambda: False
ros.get_name = lambda: "/relay"
ros.Publisher = Publisher
ros.Subscriber = subscriber
ros.signal_shutdown = lambda reason: connection.close()
geometry = types.ModuleType("geometry_msgs")
messages = types.ModuleType("geometry_msgs.msg")
messages.Twist = Twist
std = types.ModuleType("std_msgs")
std_messages = types.ModuleType("std_msgs.msg")
std_messages.String = String
sys.modules["rospy"] = ros
sys.modules["geometry_msgs"] = geometry
sys.modules["geometry_msgs.msg"] = messages
sys.modules["std_msgs"] = std
sys.modules["std_msgs.msg"] = std_messages
sys.exit(m.main())
""")
    process = subprocess.Popen(
        ["/usr/bin/python2", str(fixture), str(path), str(output)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(os.environ, CONFIRM_DRIVER="1" if confirm_driver else "0",
                 REAL_QUEUE="1" if real_queue else "0", FAIL_SUBSCRIBE="1" if fail_subscribe else "0",
                 FAIL_PUBLISHER="1" if fail_publisher else "0"),
    )
    try:
        if fail_subscribe or fail_publisher:
            assert select.select([process.stdout], [], [], 5.0)[0]
            assert process.stdout.readline() == b""
            assert process.wait(timeout=1) != 0
            return
        if not confirm_driver:
            assert select.select([process.stdout], [], [], 5.0)[0]
            assert process.stdout.readline() == b""
            assert process.wait(timeout=1) != 0
            commands = [json.loads(line) for line in output.read_text().splitlines()]
            assert commands and all(command == [0.0, 0.0] for command in commands)
            return
        assert process.stdout.readline() == b"READY\n"
        baseline = len(output.read_text().splitlines())
        stamp = time.monotonic()
        messages = ([frame(1, stamp, linear=0.0), frame(2, stamp)] if mixed_frames
                    else [frame(1, stamp)])
        process.stdin.write("".join(json.dumps(item) + "\n" for item in messages).encode())
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 2.0)[0]
        assert process.stdout.readline() == (b"MOTION_ACK 2\n" if mixed_frames else b"MOTION_ACK 1\n")
        if latch_after_motion:
            baseline = len(output.read_text().splitlines())
            stamp = time.monotonic()
            messages = [frame(2, stamp, linear=0.0), frame(3, stamp)]
            process.stdin.write("".join(json.dumps(item) + "\n" for item in messages).encode())
            process.stdin.flush()
            assert process.wait(timeout=4) == 0
            commands = [json.loads(line) for line in output.read_text().splitlines()]
            assert commands[baseline:] and all(command == [0.0, 0.0] for command in commands[baseline:])
            assert process.stdout.readline() == b"STOP_LATCHED\n"
            return
        time.sleep(0.08)
        process.stdin.close()
        assert process.wait(timeout=4) == 0
        commands = [json.loads(line) for line in output.read_text().splitlines()]
        assert any(command[0] > 0 for command in commands)
        if mixed_frames:
            first_motion = next(index for index in range(baseline, len(commands)) if commands[index][0] > 0)
            assert any(command == [0.0, 0.0] for command in commands[baseline:first_motion])
        assert commands[-1] == [0.0, 0.0]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
