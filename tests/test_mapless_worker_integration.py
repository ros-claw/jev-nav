"""Exercise worker wiring with synthetic ROS and process transports."""

import importlib.util
import io
import json
import math
import os
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

WORKER = Path(__file__).resolve().parents[1] / "worker"


def load(name):
    spec = importlib.util.spec_from_file_location(name, WORKER / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "matching_depth,decisions_continue", [(True, True), (False, True), (True, False)]
)
def test_continuous_worker_wiring_with_synthetic_ros(
    monkeypatch, tmp_path, matching_depth, decisions_continue
):
    worker = load("limo_mapless_worker")
    controller = load("limo_mapless_controller")
    relay_module = load("limo_mapless_relay")
    clock = [100.0]
    subscriptions = []
    frames = []
    decision_calls = []

    def stamp():
        captured = clock[0]
        return SimpleNamespace(to_sec=lambda: captured)

    def header(frame):
        return SimpleNamespace(frame_id=frame, stamp=stamp())

    def message(name):
        if name == "scan":
            return SimpleNamespace(
                header=header("laser"),
                ranges=[3.0] * 360,
                angle_min=-math.pi,
                angle_increment=math.pi / 180,
                range_min=0.1,
                range_max=12.0,
            )
        if name == "status":
            return SimpleNamespace(
                header=header("base"), error_code=0, motion_mode=0, battery_voltage=12.0
            )
        if name == "odom":
            velocity = 0.1 if clock[0] < 101.0 else 0.0
            twist = SimpleNamespace(
                linear=SimpleNamespace(x=velocity, y=0.0), angular=SimpleNamespace(z=0.0)
            )
            pose = SimpleNamespace(
                position=SimpleNamespace(x=(clock[0] - 100.0) * 0.1, y=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            return SimpleNamespace(
                header=header("odom"),
                child_frame_id="base_link",
                twist=SimpleNamespace(twist=twist),
                pose=SimpleNamespace(pose=pose),
            )
        frame = "depth" if name != "info" or matching_depth else ""
        return SimpleNamespace(header=header(frame), width=6, height=4, encoding="32FC1", kind=name)

    class Subscription:
        def __init__(self, callback, name):
            self.callback, self.name = callback, name
            subscriptions.append(self)
            callback(message(name), name)

        def unregister(self):
            pass

    def sleep(seconds):
        clock[0] += seconds
        for sub in subscriptions:
            sub.callback(message(sub.name), sub.name)

    class RelayInput:
        def __init__(self, process):
            self.process = process

        def write(self, data):
            frame = json.loads(data)
            frames.append(frame)
            if (frame["linear"] or frame["angular"]) and not self.process.acknowledged:
                self.process.acknowledged = True
                position = self.process.stdout.tell()
                self.process.stdout.seek(0, io.SEEK_END)
                self.process.stdout.write(("MOTION_ACK {}\n".format(frame["sequence"])).encode())
                self.process.stdout.seek(position)

        def flush(self):
            pass

        def close(self):
            self.process.finished = True

    class RelayProcess:
        def __init__(self):
            self.finished = False
            self.acknowledged = False
            self.stdin = RelayInput(self)
            self.stdout = io.BytesIO(b"READY\n")

        def poll(self):
            return 0 if self.finished else None

        def terminate(self):
            self.finished = True

        def kill(self):
            self.finished = True

        def wait(self):
            return 0

    class DecisionProcess:
        returncode = 0

        def communicate(self, payload):
            observation = json.loads(payload)
            return json.dumps(
                {"ok": True, "choice": "forward", "captured_at": observation["captured_at"]}
            ).encode(), b""

    def popen(command, **kwargs):
        return (
            RelayProcess() if command[-1].endswith("limo_mapless_relay.py") else DecisionProcess()
        )

    class InlineThread:
        def __init__(self, target, args):
            self.target, self.args, self.daemon = target, args, False

        def start(self):
            self.target(*self.args)

        def join(self, timeout=None):
            pass

    def fake_module(name, **attrs):
        result = ModuleType(name)
        for key, value in attrs.items():
            setattr(result, key, value)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    ros = fake_module(
        "rospy",
        init_node=lambda *a, **k: None,
        is_shutdown=lambda: False,
        Subscriber=lambda topic, typ, callback, callback_args, **kw: Subscription(
            callback, callback_args
        ),
        Time=SimpleNamespace(now=stamp),
    )
    del ros
    transform = SimpleNamespace(euler_from_quaternion=lambda q: (0.0, 0.0, 0.0))
    fake_module(
        "tf",
        TransformListener=lambda: SimpleNamespace(
            lookupTransform=lambda *a: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        ),
        transformations=transform,
    )
    fake_module(
        "cv2",
        IMWRITE_JPEG_QUALITY=1,
        imencode=lambda *a: (True, np.array([255, 216, 0], dtype=np.uint8)),
    )
    fake_module(
        "cv_bridge",
        CvBridge=lambda: SimpleNamespace(
            imgmsg_to_cv2=lambda msg, **kw: np.ones((4, 6), dtype=np.float32)
        ),
    )
    for package, names in (
        ("limo_base", ("LimoStatus",)),
        ("nav_msgs", ("Odometry",)),
        ("sensor_msgs", ("CameraInfo", "Image", "LaserScan")),
    ):
        fake_module(package)
        fake_module(package + ".msg", **{name: type(name, (), {}) for name in names})
    relay_module._monotonic = lambda: clock[0]
    monkeypatch.setitem(sys.modules, "limo_mapless_relay", relay_module)
    monkeypatch.setitem(sys.modules, "limo_mapless_controller", controller)
    monkeypatch.setattr(
        worker,
        "os",
        SimpleNamespace(
            geteuid=lambda: 999,
            getppid=lambda: 123,
            environ={"ROSCLAW_LIMO_DECISIONS_KEY_FILE": str(tmp_path / "key")},
            path=os.path,
        ),
    )
    (tmp_path / "key").write_text("fixture")
    monkeypatch.setattr(
        worker, "pwd", SimpleNamespace(getpwnam=lambda name: SimpleNamespace(pw_uid=999))
    )
    monkeypatch.setattr(
        worker,
        "select",
        SimpleNamespace(
            select=lambda readers, *_: (
                [reader for reader in readers if reader.tell() < len(reader.getvalue())],
                [],
                [],
            )
        ),
    )
    monkeypatch.setattr(worker, "subprocess", SimpleNamespace(Popen=popen, PIPE=-1))

    def decision_response(command, evidence, **kwargs):
        decision_calls.append(evidence)
        if not decisions_continue and len(decision_calls) > 1:
            return {"ok": False}
        return json.loads(DecisionProcess().communicate(json.dumps(evidence).encode())[0])

    monkeypatch.setattr(worker, "run_decision_process", decision_response)
    monkeypatch.setattr(
        worker,
        "threading",
        SimpleNamespace(Lock=threading.Lock, Event=threading.Event, Thread=InlineThread),
    )
    monkeypatch.setattr(worker, "time", SimpleNamespace(sleep=sleep))

    result = worker.execute(
        {
            "duration_sec": 1.0 if decisions_continue else 10.0,
            "action_id": "test",
            "daemon_pid": 123,
        }
    )
    assert any(frame["linear"] > 0 for frame in frames) is matching_depth
    assert result["ok"] is (matching_depth and decisions_continue)
    assert result["relay_acknowledged"] is matching_depth
    assert result["physical_stop_verified"] is False
    assert result["stopped_odometry"]["samples"] >= 3
    if matching_depth:
        assert decision_calls[0]["laser"]["forward_view"]["center"]["valid_rays"] > 0
    if not matching_depth:
        assert result["reason"] == "no_compatible_observation"
    elif not decisions_continue:
        assert result["reason"] == "model_decision_expired"
