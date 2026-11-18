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
    status.observe("v1 boot_1 7 0 1 0 /relay 10.0", 10.0)
    assert not status.ready(10.01)
    status.observe("v1 boot_1 8 0 1 0 /relay 10.05", 10.05)
    assert status.ready(10.1)
    assert not status.ready(10.31)
    status.observe("v1 boot_1 8 0 1 1 /relay 10.32", 10.32)
    assert not status.ready(10.32)
    status.observe("v1 boot_1 9 0 1 0 /relay 10.33", 10.33)
    assert not status.ready(10.33)


def test_driver_status_latches_on_restart_or_counter_rollback():
    for next_frame in ("v1 boot_2 1 0 1 0 /relay 10.1", "v1 boot_1 6 0 1 0 /relay 10.1"):
        status = relay.DriverStatus("/relay")
        status.observe("v1 boot_1 7 0 1 0 /relay 10.0", 10.0)
        status.observe("v1 boot_1 8 0 1 0 /relay 10.05", 10.05)
        assert status.ready(10.05)
        status.observe(next_frame, 10.1)
        status.observe("v1 boot_1 9 0 1 0 /relay 10.15", 10.15)
        assert not status.ready(10.15)


def test_driver_status_rejects_malformed_and_unsupported_mode():
    status = relay.DriverStatus("/relay")
    status.observe("v1 boot_1 0 0 1 0 /relay 1.0", 1.0)
    status.observe("v1 boot_1 1 255 1 0 /relay 1.1", 1.1)
    assert not status.ready(1.1)
    status.observe("not a driver status", 1.15)
    assert not status.ready(1.15)


def test_driver_status_latches_if_mode_changes_after_readiness():
    status = relay.DriverStatus("/relay")
    status.observe("v1 boot_1 7 0 1 0 /relay 10.0", 10.0)
    status.observe("v1 boot_1 8 0 1 0 /relay 10.05", 10.05)
    assert status.ready(10.05)
    status.observe("v1 boot_1 8 1 1 0 /relay 10.1", 10.1)
    status.observe("v1 boot_1 9 0 1 0 /relay 10.15", 10.15)
    assert not status.ready(10.15)


def test_driver_status_rejects_old_future_and_reordered_reports():
    for bad_frame, received in (("v1 boot_1 8 0 1 0 /relay 9.0", 10.1),
                                ("v1 boot_1 8 0 1 0 /relay 11.0", 10.1),
                                ("v1 boot_1 8 0 1 0 /relay 9.99", 10.1)):
        status = relay.DriverStatus("/relay")
        status.observe("v1 boot_1 7 0 1 0 /relay 10.0", 10.0)
        status.observe(bad_frame, received)
        status.observe("v1 boot_1 9 0 1 0 /relay 10.15", 10.15)
        assert not status.ready(10.15)


def test_driver_status_zero_must_come_from_this_relay():
    status = relay.DriverStatus(expected_publisher="/relay_one")
    status.observe("v1 boot_1 7 0 1 0 /another_publisher 10.0", 10.0)
    status.observe("v1 boot_1 8 0 1 0 /another_publisher 10.05", 10.05)
    assert not status.ready(10.05)
    status.observe("v1 boot_1 9 0 1 0 /relay_one 10.1", 10.1)
    assert status.ready(10.1)
    status.observe("v1 boot_1 10 0 1 0 /another_publisher 10.15", 10.15)
    assert not status.ready(10.15)


def test_driver_status_accepts_the_maximum_supported_publisher_name():
    publisher = "/" + "a" * 79
    status = relay.DriverStatus(publisher)
    for count, now in ((100, 123456.0), (101, 123456.1)):
        status.observe(f"v1 123456789012345_12345 {count} 0 1 0 {publisher} {now:.6f}", now)
    assert status.ready(123456.1)


@pytest.mark.parametrize("confirm_driver", [True, False])
def test_relay_process_emits_zero_when_control_pipe_closes(tmp_path, confirm_driver):
    output = tmp_path / "commands.jsonl"
    fixture = tmp_path / "fake_ros.py"
    fixture.write_text("""
import imp, json, os, sys, types
m = imp.load_source("relay", sys.argv[1])
m.os.geteuid = lambda: 999
m.pwd.getpwnam = lambda name: type("User", (), {"pw_uid": 999})()
class Twist(object):
    def __init__(self):
        self.linear = type("Vector", (), {"x": 0.0})()
        self.angular = type("Vector", (), {"z": 0.0})()
status = {"callback": None, "zeros": 0}
class String(object): pass
class Publisher(object):
    def __init__(self, *args, **kwargs): pass
    def get_num_connections(self): return 1
    def publish(self, message):
        with open(sys.argv[2], "a") as output:
            output.write(json.dumps([message.linear.x, message.angular.z]) + "\\n")
        if not message.linear.x and not message.angular.z: status["zeros"] += 1
        if status["callback"] and os.environ.get("CONFIRM_DRIVER") == "1":
            msg = String()
            msg.data = "v1 boot_1 %s 0 1 0 /relay %s" % (status["zeros"], m._monotonic())
            status["callback"](msg)
def subscriber(topic, message, callback, **kwargs):
    status["callback"] = callback
    return object()
ros = types.ModuleType("rospy")
ros.init_node = lambda *args, **kwargs: None
ros.is_shutdown = lambda: False
ros.get_name = lambda: "/relay"
ros.Publisher = Publisher
ros.Subscriber = subscriber
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
        env=dict(os.environ, CONFIRM_DRIVER="1" if confirm_driver else "0"),
    )
    try:
        if not confirm_driver:
            assert select.select([process.stdout], [], [], 5.0)[0]
            assert process.stdout.readline() == b""
            assert process.wait(timeout=1) != 0
            commands = [json.loads(line) for line in output.read_text().splitlines()]
            assert commands and all(command == [0.0, 0.0] for command in commands)
            return
        assert process.stdout.readline() == b"READY\n"
        process.stdin.write((json.dumps(frame(1, time.monotonic())) + "\n").encode())
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 2.0)[0]
        assert process.stdout.readline() == b"MOTION_ACK 1\n"
        time.sleep(0.08)
        process.stdin.close()
        assert process.wait(timeout=4) == 0
        commands = [json.loads(line) for line in output.read_text().splitlines()]
        assert any(command[0] > 0 for command in commands)
        assert commands[-1] == [0.0, 0.0]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
