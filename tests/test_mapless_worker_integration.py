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


def test_delayed_decisions_keep_simulated_forward_motion_until_obstacle():
    worker = load("limo_mapless_worker")
    controller_module = load("limo_mapless_controller")
    controller = controller_module.MaplessController(started_at=0.0, deadline=16.0)
    position = 0.0
    active_anchor = None
    request_sample = {"pose": (0.0, 0.0, 0.0), "pose_uncertainty": (0.0, 0.0)}
    request_stamp = 0.0
    response_at = 2.23
    accepted = 0
    frames = []

    for step in range(300):
        now = step * 0.05
        blocked = 10.0 <= now < 10.3
        sample = {
            "pose": (position, 0.0, 0.0), "pose_uncertainty": (0.0, 0.0),
            "scan_at": now, "status_at": now, "odom_at": now, "depth_at": now,
            "front": 0.3 if blocked else 3.0, "left": 2.0, "right": 2.0, "rear": 2.0,
            "depth_min": 1.0, "depth_valid_ratio": 0.2,
            "error_code": 0, "motion_mode": 0, "battery_voltage": 12.0,
            "linear_x_mps": 0.0, "angular_z_radps": 0.0,
        }
        if now >= response_at:
            accepted_response, anchor = worker.apply_revalidated_decision(
                controller, "forward", request_stamp, request_sample, sample, now
            )
            assert accepted_response and anchor is not None
            active_anchor = anchor
            accepted += 1
            request_sample, request_stamp = sample, now
            response_at = now + 2.23
        current = sample if (active_anchor is not None and worker.decision_pose_current(
            active_anchor[0], sample["pose"], active_anchor[1], active_anchor[2])) else None
        linear, angular = controller.tick(current, now)
        position += linear * 0.05
        frames.append((now, linear, angular))

    assert accepted >= 5
    assert all(linear > 0 and angular == 0 for now, linear, angular in frames if 3.0 <= now < 10.0)
    assert all(linear == 0 and angular == 0 for now, linear, angular in frames if 10.0 <= now < 10.3)
    assert all(linear > 0 for now, linear, _ in frames if 10.5 <= now < 14.5)


def test_stop_decision_is_honored_without_a_fresh_sensor_sample():
    worker = load("limo_mapless_worker")
    controller_module = load("limo_mapless_controller")
    controller = controller_module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    accepted, anchor = worker.apply_revalidated_decision(
        controller, "stop", 10.1, {}, None, 10.2
    )
    assert accepted and anchor is None and controller.choice == "stop"


def test_stop_is_honored_even_when_sensor_minimum_timestamp_repeats():
    worker = load("limo_mapless_worker")
    controller_module = load("limo_mapless_controller")
    controller = controller_module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    accepted, _ = worker.apply_revalidated_decision(controller, "stop", 10.0, {}, None, 10.2)
    assert accepted and controller.choice == "stop"


def test_model_stop_immediately_writes_a_sequenced_zero_frame():
    worker = load("limo_mapless_worker")
    controller_module = load("limo_mapless_controller")
    controller = controller_module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    decision = {"count": 1, "pose": None, "expired": False}
    sequence = {"next": 5}
    pipe = io.BytesIO()
    assert worker.accept_and_dispatch_decision(
        controller, decision, "stop", 10.0, {}, None, pipe, sequence,
        threading.Lock(), lambda: 10.2
    )
    frame = json.loads(pipe.getvalue())
    assert frame == {"sequence": 5, "stamp": 10.2, "linear": 0.0, "angular": 0.0}
    assert sequence["next"] == 6 and decision["count"] == 2


def test_failed_stop_write_signals_a_fatal_control_pipe_failure():
    worker = load("limo_mapless_worker")
    controller_module = load("limo_mapless_controller")
    controller = controller_module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    decision = {"count": 1, "pose": None, "expired": False}
    failed = threading.Event()

    class FullPipe:
        def write(self, payload):
            raise BlockingIOError("full")

    with pytest.raises(OSError):
        worker.accept_and_dispatch_decision(
            controller, decision, "stop", 10.0, {}, None, FullPipe(), {"next": 5},
            threading.Lock(), lambda: 10.2, on_write_failure=failed.set
        )
    assert failed.is_set()


def test_budget_stop_immediately_writes_zero_even_when_model_chose_forward():
    worker = load("limo_mapless_worker")
    controller_module = load("limo_mapless_controller")
    controller = controller_module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    controller.linear = 0.1
    observed = {"pose": (0.0, 0.0, 0.0), "pose_uncertainty": (0.0, 0.0)}
    current = {
        "pose": (0.13, 0.0, 0.0), "odom_at": 10.2, "scan_at": 10.2,
        "status_at": 10.2, "depth_at": 10.2, "front": 3.0, "left": 2.0,
        "right": 2.0, "rear": 2.0, "depth_min": 1.0, "depth_valid_ratio": 0.2,
        "error_code": 0, "motion_mode": 0, "battery_voltage": 12.0,
        "linear_x_mps": 0.1, "angular_z_radps": 0.0,
    }
    pipe, sequence = io.BytesIO(), {"next": 5}
    decision = {"count": 1, "pose": None, "expired": False}
    assert worker.accept_and_dispatch_decision(
        controller, decision, "forward", 10.1, observed, current, pipe, sequence,
        threading.Lock(), lambda: 10.2
    )
    assert json.loads(pipe.getvalue())["linear"] == 0.0
    assert controller.choice == "stop"


def test_pose_distance_already_used_reduces_the_remaining_motion_budget():
    worker = load("limo_mapless_worker")
    controller_module = load("limo_mapless_controller")
    controller = controller_module.MaplessController(started_at=10.0, deadline=30.0)
    observed = {"pose": (0.0, 0.0, 0.0), "pose_uncertainty": (0.0, 0.0)}
    current = {
        "pose": (0.14, 0.0, 0.0), "odom_at": 10.2, "scan_at": 10.2,
        "status_at": 10.2, "depth_at": 10.2, "front": 3.0, "left": 2.0,
        "right": 2.0, "rear": 2.0, "depth_min": 1.0, "depth_valid_ratio": 0.2,
        "error_code": 0, "motion_mode": 0, "battery_voltage": 12.0,
        "linear_x_mps": 0.0, "angular_z_radps": 0.0,
    }
    accepted, _ = worker.apply_revalidated_decision(
        controller, "forward", 10.0, observed, current, 10.2
    )
    assert accepted
    assert controller.tick(current, 10.2) == (0.0, 0.0)


def test_exhausted_pose_budget_requests_zero_without_a_deceleration_ramp():
    worker = load("limo_mapless_worker")
    controller_module = load("limo_mapless_controller")
    controller = controller_module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    controller.linear = 0.1
    observed = {"pose": (0.0, 0.0, 0.0), "pose_uncertainty": (0.0, 0.0)}
    current = {
        "pose": (0.13, 0.0, 0.0), "odom_at": 10.2, "scan_at": 10.2,
        "status_at": 10.2, "depth_at": 10.2, "front": 3.0, "left": 2.0,
        "right": 2.0, "rear": 2.0, "depth_min": 1.0, "depth_valid_ratio": 0.2,
        "error_code": 0, "motion_mode": 0, "battery_voltage": 12.0,
        "linear_x_mps": 0.1, "angular_z_radps": 0.0,
    }
    accepted, _ = worker.apply_revalidated_decision(
        controller, "forward", 10.1, observed, current, 10.2
    )
    assert accepted
    assert controller.tick(current, 10.25) == (0.0, 0.0)


def test_exhausted_turn_budget_does_not_turn_a_left_choice_into_forward_motion():
    worker = load("limo_mapless_worker")
    controller_module = load("limo_mapless_controller")
    controller = controller_module.MaplessController(started_at=10.0, deadline=30.0)
    observed = {"pose": (0.0, 0.0, 0.0), "pose_uncertainty": (0.0, 0.0)}
    current = {
        "pose": (0.0, 0.0, 0.31), "odom_at": 10.2, "scan_at": 10.2,
        "status_at": 10.2, "depth_at": 10.2, "front": 3.0, "left": 2.0,
        "right": 2.0, "rear": 2.0, "depth_min": 1.0, "depth_valid_ratio": 0.2,
        "error_code": 0, "motion_mode": 0, "battery_voltage": 12.0,
        "linear_x_mps": 0.0, "angular_z_radps": 0.0,
    }
    accepted, _ = worker.apply_revalidated_decision(controller, "left", 10.0, observed, current, 10.2)
    assert accepted
    assert controller.tick(current, 10.25) == (0.0, 0.0)


def test_new_decision_cannot_use_a_sample_taken_before_it_was_accepted():
    worker = load("limo_mapless_worker")
    controller_module = load("limo_mapless_controller")
    controller = controller_module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    decision = {"count": 2, "pose": ((0.0, 0.0, 0.0), 0.0, 0.0), "expired": False}
    sample = {
        "pose": (0.0, 0.0, 0.0), "odom_at": 10.2, "scan_at": 10.2,
        "status_at": 10.2, "depth_at": 10.2, "front": 3.0, "left": 2.0,
        "right": 2.0, "rear": 2.0, "depth_min": 1.0, "depth_valid_ratio": 0.2,
        "error_code": 0, "motion_mode": 0, "battery_voltage": 12.0,
        "linear_x_mps": 0.0, "angular_z_radps": 0.0,
    }
    pipe = io.BytesIO()
    linear, angular = worker.send_control_frame(
        controller, decision, sample, 10.2, {"next": 1}, pipe, threading.Lock(), 8.0,
        sampled_generation=1
    )
    assert (linear, angular) == (None, None)
    assert pipe.getvalue() == b""


def test_stop_acceptance_cannot_overtake_a_pending_control_frame():
    worker = load("limo_mapless_worker")
    controller_module = load("limo_mapless_controller")
    controller = controller_module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    decision = {"count": 1, "pose": ((0.0, 0.0, 0.0), 0.0, 0.0), "expired": False}
    sample = {
        "pose": (0.0, 0.0, 0.0), "odom_at": 10.2, "scan_at": 10.2,
        "status_at": 10.2, "depth_at": 10.2, "front": 3.0, "left": 2.0,
        "right": 2.0, "rear": 2.0, "depth_min": 1.0, "depth_valid_ratio": 0.2,
        "error_code": 0, "motion_mode": 0, "battery_voltage": 12.0,
        "linear_x_mps": 0.0, "angular_z_radps": 0.0,
    }
    started, release, stopped = threading.Event(), threading.Event(), threading.Event()
    lock = threading.Lock()

    class Pipe(io.BytesIO):
        def write(self, payload):
            started.set()
            assert release.wait(2.0)
            return super().write(payload)

    pipe = Pipe()
    sequence = {"next": 1}
    sender = threading.Thread(target=worker.send_control_frame,
                              args=(controller, decision, sample, 10.2, sequence, pipe, lock, 8.0, 1))

    def stop():
        assert worker.accept_and_dispatch_decision(
            controller, decision, "stop", 10.1, {}, None, pipe, sequence, lock, lambda: 10.2
        )
        stopped.set()

    sender.start()
    assert started.wait(2.0)
    stopper = threading.Thread(target=stop)
    stopper.start()
    assert not stopped.wait(0.05)
    release.set()
    sender.join(timeout=2.0)
    stopper.join(timeout=2.0)
    assert stopped.is_set()
    frames = [json.loads(line) for line in pipe.getvalue().splitlines()]
    assert [item["sequence"] for item in frames] == [1, 2]
    assert frames[0]["linear"] == pytest.approx(0.06) and frames[1]["linear"] == 0.0
    pipe.seek(0)
    pipe.truncate()
    worker.send_control_frame(controller, decision, sample, 10.25, sequence, pipe, lock, 8.0, 2)
    assert json.loads(pipe.getvalue())["linear"] == 0.0


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
    monkeypatch.setattr(worker, "NonblockingPipeWriter", lambda stream: stream)

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
