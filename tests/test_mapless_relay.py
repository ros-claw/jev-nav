import importlib.util
import json
import select
import subprocess
import time
from pathlib import Path

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


def test_relay_process_emits_zero_when_control_pipe_closes(tmp_path):
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
class Publisher(object):
    def __init__(self, *args, **kwargs): pass
    def get_num_connections(self): return 1
    def publish(self, message):
        with open(sys.argv[2], "a") as output:
            output.write(json.dumps([message.linear.x, message.angular.z]) + "\\n")
ros = types.ModuleType("rospy")
ros.init_node = lambda *args, **kwargs: None
ros.is_shutdown = lambda: False
ros.Publisher = Publisher
geometry = types.ModuleType("geometry_msgs")
messages = types.ModuleType("geometry_msgs.msg")
messages.Twist = Twist
sys.modules["rospy"] = ros
sys.modules["geometry_msgs"] = geometry
sys.modules["geometry_msgs.msg"] = messages
sys.exit(m.main())
""")
    process = subprocess.Popen(
        ["/usr/bin/python2", str(fixture), str(path), str(output)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
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
