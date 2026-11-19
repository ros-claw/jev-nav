#!/usr/bin/env python2
"""Review candidate for a live trial. Importing or running this file does no I/O.

The default session is diagnostic-only. Live motion additionally requires
separate command-caller and serial-ownership monitors. ROS graph enumeration
is not atomic, and TIOCEXCL cannot exclude previously open descriptors. The
read-only monitor candidates live in ``live_monitors.py``. These adapters do not claim
to be simulation-only, and cannot be passed to the offline ForwardTrial class.
None of their status messages prove physical stopping or clearance.
"""
from __future__ import division

import copy
import errno
import fcntl
import hashlib
import json
import math
import os
import select
import subprocess
import sys
import threading

import forward_trial as checks


def _copy_scene(scene):
    """Own a fusion snapshot without recursively copying every numeric ray."""
    result = {key: copy.deepcopy(value) for key, value in scene.items()
              if key not in ("rays", "occupied_hits")}
    result["rays"] = [dict(ray, start_base=tuple(ray["start_base"]),
                            hit_base=tuple(ray["hit_base"])) for ray in scene["rays"]]
    result["occupied_hits"] = [dict(hit, point_base=tuple(hit["point_base"]))
                               for hit in scene["occupied_hits"]]
    return result


class ObservationAdapter(object):
    """Translate an existing RosObservation without creating ROS connections."""
    simulation_only = False

    def __init__(self, observation, jpeg_encoder):
        self.observation = observation
        self.jpeg_encoder = jpeg_encoder
        self.last_issue = None

    def snapshot(self):
        try:
            sample = self.observation.snapshot()
            if not isinstance(sample, dict):
                self.last_issue = "source_unavailable"
                return None
            if not self.observation.is_current_sample(sample):
                self.last_issue = "current_before_copy"
                return None
            scan, color = sample["scan"], sample["color"]
            result = {key: _copy_scene(value) if key == "scene" else copy.deepcopy(value)
                      for key, value in sample.items() if key not in ("scan", "color")}
            # These fields must come from the actual observation, never defaults.
            result["depth_pose"] = copy.deepcopy(sample["depth_pose"])
            result["state"]["vehicle_state"] = sample["state"]["vehicle_state"]
            result["state"]["control_mode"] = sample["state"]["control_mode"]
            result["image_jpeg"] = self.jpeg_encoder(color)
            result["raw_scan"] = {"stamp": sample["scan_at"], "frame_id": scan.header.frame_id,
                                  "angle_min": scan.angle_min, "angle_increment": scan.angle_increment,
                                  "range_min": scan.range_min, "range_max": scan.range_max,
                                  "time_increment": scan.time_increment, "ranges": list(scan.ranges)}
            if not self.observation.is_current_sample(result):
                self.last_issue = "current_after_copy"
                return None
            self.last_issue = None
            return result
        except Exception:  # malformed observations are unavailable
            self.last_issue = "adapter_error"
            return None

    def is_current_sample(self, sample):
        try:
            return self.observation.is_current_sample(sample) is True
        except Exception:
            return False

    def is_current_epoch(self, epoch):
        try:
            return self.observation.is_current_epoch(epoch) is True
        except Exception:
            return False

    def stationary_trace(self, start, end):
        try:
            return copy.deepcopy(self.observation.stationary_trace(start, end))
        except Exception:
            return []


class CvJpegEncoder(object):
    """Explicitly supplied bridge; CV import/conversion happens only on call."""
    def __init__(self, bridge):
        self.bridge = bridge

    def __call__(self, color):
        import cv2
        image = self.bridge.imgmsg_to_cv2(color, desired_encoding="bgr8")
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return encoded.tostring()


def make_evidence(sample):
    """Bind the original trial evidence plus decision state and extrinsics."""
    evidence = checks._evidence(sample)
    extra = {"state": copy.deepcopy(sample["state"]),
             "laser_extrinsics": list(sample["laser_extrinsics"]),
             "odom_at": sample["odom_at"], "status_at": sample["status_at"],
             "depth_at": sample["depth_at"]}
    encoded = json.dumps(extra, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(evidence["observation_id"].encode("ascii") + encoded).hexdigest()
    evidence.update(extra)
    evidence["observation_id"] = digest
    return evidence


class DecisionsAdapter(object):
    """Bind a single helper call's matching response to its local evidence ID."""
    simulation_only = False

    def __init__(self, invoke, timeout_sec=6.0):
        if not callable(invoke) or not checks._finite(timeout_sec) or not 0 < timeout_sec <= 6.0:
            raise ValueError("bounded Decisions invoker required")
        self.invoke, self.timeout_sec = invoke, timeout_sec

    def decide(self, evidence):
        rejected = {"ok": False, "choice": "stop"}
        try:
            from explore import build_decision_evidence, physical_forward_view
            raw = evidence["raw_scan"]
            ranges = [value if checks._finite(value) else float("nan") for value in raw["ranges"]]
            view = physical_forward_view(ranges, raw["angle_min"], raw["angle_increment"],
                                         raw["range_min"], raw["range_max"], *evidence["laser_extrinsics"])
            sample = dict(evidence)
            sample["forward_view"] = view
            payload = build_decision_evidence(sample, evidence["image_jpeg"])
            if payload is None:
                return rejected
            result = self.invoke(payload, self.timeout_sec)
            if (not isinstance(result, dict) or result.get("ok") is not True
                    or result.get("captured_at") != payload["captured_at"]
                    or result.get("choice") not in ("forward", "left", "right", "stop")):
                return rejected
            return {"ok": True, "choice": result["choice"], "observation_id": evidence["observation_id"]}
        except Exception:
            return rejected


class ExistingDecisionHelper(object):
    """Invoke the existing bounded helper only when explicitly called."""
    def __init__(self, clock, cancelled):
        self.clock, self.cancelled = clock, cancelled

    def __call__(self, payload, timeout):
        from explore import WORKER, run_decision_process
        command = ["/usr/bin/python3", os.path.join(WORKER, "limo_mapless_decision.py")]
        return run_decision_process(command, payload, cancelled=self.cancelled,
                                    clock=self.clock.now, timeout=timeout)


class RelayProcessTransport(object):
    """Bounded pipe transport; starts the relay only when start() is called."""
    def __init__(self, clock, popen=None):
        self.clock = clock
        self.popen = popen if popen is not None else subprocess.Popen
        self.process = None
        self.buffer = b""
        self.exited = None

    def start(self, max_motion_sec):
        if self.process is not None or max_motion_sec != 0.3:
            raise ValueError("invalid relay startup")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relay.py")
        command = ["/usr/bin/python2", path, "--max-motion-sec", "0.300000"]
        with open(os.devnull, "wb") as errors:
            self.process = self.popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=errors, close_fds=True)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is None:
                continue
            fd = stream.fileno()
            fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)

    def read_line(self, timeout):
        if self.process is None or not checks._finite(timeout) or not 0 <= timeout <= 3.5:
            raise ValueError("invalid relay read")
        deadline = self.clock.now() + timeout
        while True:
            if b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                return line + b"\n"
            if len(self.buffer) > 512:
                raise IOError("relay status line too long")
            remaining = max(0.0, deadline - self.clock.now())
            readable, _, _ = select.select([self.process.stdout.fileno()], [], [], remaining)
            if not readable:
                return None
            try:
                chunk = os.read(self.process.stdout.fileno(), 513)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    return None
                raise
            if not chunk:
                raise IOError("relay status pipe closed")
            self.buffer += chunk

    def write_line(self, data, timeout):
        if (self.process is None or not isinstance(data, bytes) or len(data) > 512
                or not data.endswith(b"\n") or not checks._finite(timeout) or not 0 <= timeout <= 0.1):
            return False
        fd = self.process.stdin.fileno()
        _, writable, _ = select.select([], [fd], [], timeout)
        if not writable:
            return False
        try:
            # One nonblocking PIPE_BUF-sized write; any short write fails closed.
            return os.write(fd, data) == len(data)
        except (OSError, IOError):
            return False

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def close(self):
        if self.process is None:
            return
        try:
            self.process.stdin.close()  # EOF is another relay stop request.
        except Exception:
            pass
        if self.process.poll() is None:
            try:
                self.process.terminate()
            except OSError:
                pass
        # Relay shutdown sends zeros for 0.5 s before unregistering its ROS
        # publisher. Allow that cleanup to finish before escalating to kill.
        for _ in range(160):
            if self.process.poll() is not None:
                break
            self.clock.sleep(0.01)
        if self.process.poll() is None:
            self.process.kill()
            for _ in range(30):
                if self.process.poll() is not None:
                    break
                self.clock.sleep(0.01)
        self.exited = self.process.poll() is not None
        if self.exited:
            self.process.wait()
        for stream in (self.process.stdout,):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass


class RelayAdapter(object):
    """One calling thread writes bounded relay JSON; no start until arm()."""
    simulation_only = False

    def __init__(self, transport, clock):
        self.transport, self.clock = transport, clock
        self.owner = None
        self.armed = False
        self.faulted = False
        self.sequence = -1
        self.last_stamp = None
        self.had_motion = False
        self.evidence = {"zero_write_completed": False, "stop_line": None,
                         "physical_stop_verified": False}

    def _owner(self):
        return self.owner == threading.current_thread().ident

    def arm(self, max_motion_sec):
        if self.owner is not None or max_motion_sec != 0.3:
            return None
        self.owner = threading.current_thread().ident
        try:
            self.transport.start(max_motion_sec)
            ready = self.transport.read_line(3.5)
            if ready != checks.RELAY_READY_LINE:
                self.faulted = True
                return None
            self.armed = True
            return {"ready_line": ready, "relative_motion_limit": True}
        except Exception:
            self.faulted = True
            return None

    def healthy(self):
        if not self._owner() or not self.armed or self.faulted:
            return False
        try:
            line = self.transport.read_line(0.0)
            if line is not None:
                self.evidence["stop_line"] = line
                self.faulted = True
            return not self.faulted and self.transport.alive() is True
        except Exception:
            self.faulted = True
            return False

    def _write(self, sequence, stamp, linear, angular):
        frame = {"sequence": sequence, "stamp": stamp, "linear": linear, "angular": angular}
        data = (json.dumps(frame, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
        if self.transport.write_line(data, 0.05) is not True:
            self.faulted = True
            return False
        self.sequence, self.last_stamp = sequence, stamp
        return True

    def send_frame(self, sequence, stamp, linear, angular):
        if (not self._owner() or not self.armed or self.faulted
                or isinstance(sequence, bool) or not isinstance(sequence, checks.INTEGER_TYPES)
                or sequence != self.sequence + 1 or not checks._finite(stamp)
                or (self.last_stamp is not None and stamp <= self.last_stamp)
                or not checks._finite(linear) or not 0 < linear <= checks.MAX_SPEED_MPS
                or not checks._finite(angular) or angular != 0.0):
            return False
        try:
            written = self._write(sequence, stamp, linear, angular)
            if written:
                self.had_motion = True
            return written
        except Exception:
            self.faulted = True
            return False

    def send_zero_heartbeat(self):
        if not self._owner() or not self.armed or self.faulted or self.had_motion:
            return False
        try:
            now = self.clock.now()
            if self.last_stamp is not None and now == self.last_stamp:
                self.clock.sleep(0.001)
                now = self.clock.now()
            return (checks._finite(now) and (self.last_stamp is None or now > self.last_stamp)
                    and self._write(self.sequence + 1, now, 0.0, 0.0))
        except Exception:
            self.faulted = True
            return False

    def stop_and_latch(self):
        if not self._owner() or not self.armed:
            return False
        try:
            now = self.clock.now()
            if self.last_stamp is not None and now == self.last_stamp:
                self.clock.sleep(0.001)
                now = self.clock.now()
            if not checks._finite(now) or (self.last_stamp is not None and now <= self.last_stamp):
                return False
            sent = self._write(self.sequence + 1, now, 0.0, 0.0)
            self.evidence["zero_write_completed"] = sent
            if sent:
                try:
                    self.evidence["stop_line"] = self.transport.read_line(0.15)
                except Exception:
                    pass  # zero was requested; missing status is not proof of a stop
            return sent
        except Exception:
            return False

    def close(self):
        if self.owner is not None and not self._owner():
            return False
        try:
            self.transport.close()
            exited = self.transport.alive() is False
        except BaseException:
            exited = False
        self.evidence["process_exited"] = exited
        return exited


class RosGraphGuard(object):
    """A ROS graph listing cannot prevent a new publisher from racing the check."""
    monitoring_enabled = False

    def acquire(self):
        return False


def _depth_reason(sample):
    try:
        depth = sample["depth"]
        grid = depth["valid_grid"]
        if (not isinstance(grid, (tuple, list)) or len(grid) != 3
                or any(not isinstance(row, (tuple, list)) or len(row) != 3 for row in grid)
                or any(not checks._finite(v) or not 0.5 <= v <= 1.0 for row in grid for v in row)):
            return "depth_grid_incomplete"
        if not checks._finite(depth["min_m"]) or depth["min_m"] <= 0.5:
            return "depth_near_or_unknown"
    except (KeyError, TypeError, ValueError):
        return "depth_unavailable"
    return None


def _live_timestamps_fresh(sample, now):
    """Allow bounded processing latency; callers also recheck current odom and scan."""
    try:
        ages = {"snapshot_at": 0.25, "color_at": 0.5, "scan_at": 0.5,
                "odom_at": 0.25, "status_at": 1.0, "depth_at": 0.5}
        return isinstance(sample, dict) and all(checks._fresh(sample[name], now, age)
                                                for name, age in ages.items())
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


class StrictDepthGate(object):
    """Default legacy image-grid gate; it does not certify clearance."""
    def assess(self, sample, now):
        reason = _depth_reason(sample)
        return {"candidate": reason is None, "reason": reason or "depth_grid_accepted",
                "depth_at": sample.get("depth_at"), "geometric_clearance_verified": False}


def _snapshot_without_depth_gate(sample, now, diagnostic=None):
    """The offline sample gates, with only depth interpretation delegated."""
    def reject(check, reason="sensor_unavailable"):
        if diagnostic is not None:
            diagnostic["check"] = check
        return reason
    try:
        if not isinstance(sample, dict) or not _live_timestamps_fresh(sample, now):
            return reject("timestamps")
        epoch = sample["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, checks.INTEGER_TYPES) or epoch < 0:
            return reject("epoch")
        image = sample["image_jpeg"]
        if (not isinstance(image, bytes) or not image.startswith(b"\xff\xd8")
                or not image.endswith(b"\xff\xd9") or not 4 < len(image) <= 2000000):
            return reject("image")
        pose, image_pose, depth_pose = sample["pose"], sample["image_pose"], sample["depth_pose"]
        if not all(checks._pose(item) for item in (pose, image_pose, depth_pose)):
            return reject("pose")
        scene = sample["scene"]
        if (not isinstance(scene, dict) or not isinstance(scene.get("coverage"), dict)
                or scene["coverage"].get("scan_stamp") != sample["scan_at"]
                or not checks._pose(scene["coverage"].get("scan_pose"))):
            return reject("scene_metadata")
        for other in (image_pose, depth_pose, scene["coverage"]["scan_pose"]):
            distance, rotation = checks._delta(pose, other)
            if not checks._finite(distance) or not checks._finite(rotation) or distance > 0.01 or rotation > 0.02:
                return reject("pose_alignment")
        # The pose is deliberately refreshed after sensor capture. Its age and
        # displacement are checked separately; comparing that newer odometry
        # stamp as if it were a camera/scan exposure creates false skew.
        stamps = [sample[name] for name in ("color_at", "scan_at", "depth_at")]
        if max(stamps) - min(stamps) > 0.25:
            return reject("sensor_skew")
        reason, _ = checks._raw_scan(sample)
        if reason:
            return reject("raw_scan", reason)
        state = sample["state"]
        if (not isinstance(state, dict) or not checks._finite(state.get("stamp"))
                or state.get("stamp") != min(sample["odom_at"], sample["status_at"])
                or type(state.get("motion_mode")) is not int or state["motion_mode"] != 0
                or type(state.get("vehicle_state")) is not int or state["vehicle_state"] != 0
                or type(state.get("control_mode")) is not int or state["control_mode"] != 1
                or type(state.get("error_code")) is not int or state["error_code"] != 0
                or not checks._finite(state.get("battery_voltage")) or state["battery_voltage"] < 10.0
                or not checks._finite(state.get("linear_x_mps")) or not 0 <= state["linear_x_mps"] <= 0.02
                or not checks._finite(state.get("linear_y_mps")) or abs(state["linear_y_mps"]) > 0.005
                or not checks._finite(state.get("angular_z_radps")) or abs(state["angular_z_radps"]) > 0.02):
            return reject("base_state")
        depth = sample["depth"]
        if (not isinstance(depth, dict) or not checks._finite(depth.get("stamp"))
                or depth["stamp"] != sample["depth_at"]):
            return reject("depth_stamp")
        # The fusion snapshot is assessed at its actual capture time. The
        # caller separately checks current odometry and vetoes new scan hits
        # before every command, so an arriving scan does not invalidate older
        # ray coverage solely because processing took time.
        front = checks.assess_front(scene, sample["snapshot_at"], latest_age_limit=0.5,
                                    history_age_limit=0.65, history_span_limit=0.4)
        if not front["candidate"]:
            if diagnostic is not None:
                diagnostic.update({"scene_status": scene.get("status"),
                                   "sensor_fresh": scene["coverage"].get("sensor_fresh"),
                                   "scan_age_at_snapshot_sec": sample["snapshot_at"] - sample["scan_at"],
                                   "scan_age_now_sec": now - sample["scan_at"],
                                   "rays": len(scene.get("rays", [])),
                                   "hits": len(scene.get("occupied_hits", []))})
            return reject("front_" + front["reason"],
                          "front_blocked" if front["reason"] == "forward_obstacle" else "sensor_unavailable")
        return None
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return reject("malformed_sample")


class LiveTrial(object):
    """Candidate entry point; external ROS and serial monitors are required."""
    def __init__(self, source, model, relay, arbiter, clock, enable_motion=False,
                 duration_sec=0.6, speed_mps=0.02, period_sec=0.05, serial_guard=None, depth_gate=None,
                 operator_site_report=None):
        if (not all(checks._finite(x) for x in (duration_sec, speed_mps, period_sec))
                or not 0 < duration_sec <= 0.6 or not 0 < speed_mps <= 0.02
                or not 0 < period_sec <= 0.05 or type(enable_motion) is not bool):
            raise ValueError("invalid trial bounds")
        self.source, self.model, self.relay, self.arbiter, self.clock = source, model, relay, arbiter, clock
        self.serial_guard = serial_guard
        self.depth_gate = depth_gate if depth_gate is not None else StrictDepthGate()
        self.operator_site_report = copy.deepcopy(operator_site_report)
        self._last_depth_assessment = None
        self._last_diagnostic = None
        self._validation_detail = {}
        self._stationary_proof_at = None
        self._stationary_proof_key = None
        self.enable_motion = enable_motion
        self.duration_sec, self.speed_mps, self.period_sec = duration_sec, speed_mps, period_sec
        self._used = False
        self._lock = threading.Lock()
        self._last_time = None

    def _now(self):
        now = self.clock.now()
        if not checks._finite(now) or now < 0 or (self._last_time is not None and now < self._last_time):
            raise checks._StopError("clock_invalid")
        self._last_time = now
        return now

    def _sample(self):
        sample = self.source.snapshot()
        now = self._now()
        if sample is None:
            self._last_diagnostic = {"step": "source_snapshot",
                                     "detail": getattr(self.source, "last_issue", None)}
            raise checks._StopError("sensor_unavailable")
        reason = self._validate(sample, now)
        if reason:
            self._last_diagnostic = {"step": "snapshot_validation", "reason": reason,
                                     "check": self._validation_detail.get("check")}
            self._last_diagnostic.update(self._validation_detail)
            raise checks._StopError(reason)
        if (self.source.is_current_epoch(sample["epoch"]) is not True
                or self.source.is_current_sample(sample) is not True):
            self._last_diagnostic = {"step": "latest_sample_recheck"}
            raise checks._StopError("sensor_unavailable")
        self._last_diagnostic = None
        return sample, now

    def _validate(self, sample, now):
        self._validation_detail = {}
        reason = _snapshot_without_depth_gate(sample, now, self._validation_detail)
        if reason:
            return reason
        try:
            assessment = self.depth_gate.assess(sample, now)
            if (not isinstance(assessment, dict) or type(assessment.get("candidate")) is not bool
                    or not isinstance(assessment.get("reason"), checks.STRING_TYPES)
                    or not assessment["reason"] or len(assessment["reason"]) > 128
                    or not checks._finite(assessment.get("depth_at"))
                    or assessment["depth_at"] != sample["depth_at"]
                    or assessment.get("geometric_clearance_verified") is not False):
                return "depth_gate_invalid"
            self._last_depth_assessment = copy.deepcopy(assessment)
            return None if assessment["candidate"] else assessment["reason"]
        except Exception:
            return "depth_gate_invalid"

    def _model_fresh(self, before, now):
        if (not checks._fresh(before["color_at"], now, checks.MODEL_MAX_AGE_SEC)
                or not checks._fresh(before["scan_at"], now, checks.MODEL_MAX_AGE_SEC)):
            raise checks._StopError("model_stale")

    def _stationary(self, before, now):
        key = (before["epoch"], before["color_at"], before["scan_at"], tuple(before["image_pose"]))
        if self._stationary_proof_key is not None and self._stationary_proof_key != key:
            raise checks._StopError("sensor_unavailable")
        start = (self._stationary_proof_at if self._stationary_proof_at is not None
                 else min(before["color_at"], before["scan_at"]))
        first_end = now
        trace = self.source.stationary_trace(start, first_end)
        tail_end = self._now()
        self._model_fresh(before, tail_end)
        if (self.source.is_current_epoch(before["epoch"]) is not True
                or not checks._stationary_trace_valid(trace, start, first_end,
                                                       before["epoch"], before["image_pose"], max_gap_sec=0.15)):
            self._trace_failed(trace, start, first_end, "stationary_trace_head")
        # The first history can take time to copy. Read the intervening tail so
        # odometry during that processing interval is checked as well.
        tail = self.source.stationary_trace(first_end, tail_end)
        now = self._now()
        self._model_fresh(before, now)
        if (self.source.is_current_epoch(before["epoch"]) is not True
                or not checks._stationary_trace_valid(tail, first_end, now,
                                                       before["epoch"], before["image_pose"], max_gap_sec=0.15)):
            self._trace_failed(tail, first_end, now, "stationary_trace_tail")
        self._stationary_proof_at = now
        self._stationary_proof_key = key
        return now

    def _trace_failed(self, trace, start, now, step):
        self._last_diagnostic = {"step": step}
        try:
            gaps = [trace[i]["stamp"] - trace[i - 1]["stamp"] for i in range(1, len(trace))]
            stats = {"samples": len(trace), "max_gap_sec": max(gaps) if gaps else None,
                     "start_gap_sec": start - trace[0]["stamp"] if trace else None,
                     "end_gap_sec": now - trace[-1]["stamp"] if trace else None}
            if all(value is None or checks._finite(value) for value in stats.values()):
                self._last_diagnostic.update(stats)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
        raise checks._StopError("sensor_unavailable")

    def _aligned_stationary_sample(self, before):
        deadline = self._now() + 3.0
        if self.relay.armed and (self.relay.healthy() is not True
                                 or self.relay.send_zero_heartbeat() is not True):
            raise checks._StopError("relay_failure")
        # Check the potentially long inference history before obtaining the
        # next sample. Later checks extend this proof with the short new tail.
        self._stationary(before, self._now())
        for _ in range(301):
            now = self._now()
            self._model_fresh(before, now)
            if self.relay.armed and (self.relay.healthy() is not True
                                     or self.relay.send_zero_heartbeat() is not True):
                raise checks._StopError("relay_failure")
            try:
                sample, now = self._sample()
            except checks._StopError as exc:
                if str(exc) != "sensor_unavailable":
                    raise
                now = self._now()
                if now >= deadline:
                    raise
                self.clock.sleep(min(0.01, deadline - now))
                if self._now() <= now:
                    raise checks._StopError("clock_invalid")
                continue
            self._model_fresh(before, now)
            distance, yaw = checks._delta(before["image_pose"], sample["pose"])
            if (sample["epoch"] != before["epoch"] or distance > 0.005 or yaw > 0.01
                    or sample["state"]["linear_x_mps"] > 0.005
                    or abs(sample["state"]["linear_y_mps"]) > 0.005
                    or abs(sample["state"]["angular_z_radps"]) > 0.01):
                raise checks._StopError("sensor_unavailable")
            now = self._stationary(before, now)
            fresh = _live_timestamps_fresh(sample, now)
            current = self.source.is_current_sample(sample) is True
            if not fresh or not current:
                self._last_diagnostic = {"step": "post_trace_recheck", "timestamps_fresh": fresh,
                                         "current_sample": current,
                                         "snapshot_age_sec": now - sample["snapshot_at"]}
                if now >= deadline:
                    raise checks._StopError("sensor_unavailable")
                self.clock.sleep(min(0.01, deadline - now))
                continue
            return sample, now
        raise checks._StopError("sensor_unavailable")

    def _guard_evidence(self, result):
        """Record best-effort evidence without claiming impossible guarantees."""
        try:
            ros = self.arbiter.evidence()
            serial = self.serial_guard.evidence()
            if (not isinstance(ros, dict) or ros.get("driver_caller_bound") is not True
                    or ros.get("observed_competitors") != []
                    or type(ros.get("graph_query_atomic")) is not bool):
                raise checks._StopError("exclusive_arbitration_lost")
            if (not isinstance(serial, dict) or serial.get("tioc_exclusive_reported") is not False
                    or serial.get("initial_open_errno") != "EBUSY"
                    or serial.get("probe_repeated") is not False
                    or serial.get("observed_other_fds") != []
                    or type(serial.get("previous_fd_excluded")) is not bool):
                raise checks._StopError("serial_exclusivity_lost")
            result["ros_guard_evidence"] = copy.deepcopy(ros)
            result["serial_guard_evidence"] = copy.deepcopy(serial)
        except checks._StopError:
            raise
        except BaseException:
            raise checks._StopError("guard_evidence_unavailable")

    def run(self):
        with self._lock:
            if self._used:
                return {"reason": "already_used", "physical_stop_verified": False}
            self._used = True
        result = {"reason": "exception", "stage": "preflight", "motion_requested": False, "stop_requested": False,
                  "physical_stop_verified": False, "site_conditions_verified": False,
                  "atomic_exclusivity_verified": False, "serial_sole_writer_verified": False,
                  "site_conditions_operator_report": copy.deepcopy(self.operator_site_report)}
        acquired = serial_acquired = attempted_arm = armed = False
        try:
            started = self._now()
            if not self.enable_motion:
                sample, now = self._sample()
                result["reason"] = "dry_run"
                return result
            # Wait briefly for post-session image and scan; no motion occurs here.
            sample = None
            for _ in range(301):
                try:
                    candidate, now = self._sample()
                    if all(candidate[name] >= started for name in
                           ("color_at", "scan_at", "odom_at", "status_at", "depth_at")):
                        sample = candidate
                        break
                except checks._StopError as exc:
                    if str(exc) != "sensor_unavailable":
                        raise
                now = self._now()
                if now - started >= 3.0:
                    break
                self.clock.sleep(min(0.01, started + 3.0 - now))
                if self._now() <= now:
                    raise checks._StopError("clock_invalid")
            if sample is None:
                raise checks._StopError("pre_session_sensor_frame")
            if (sample["state"]["linear_x_mps"] > 0.005
                    or abs(sample["state"]["linear_y_mps"]) > 0.005
                    or abs(sample["state"]["angular_z_radps"]) > 0.01):
                raise checks._StopError("sensor_unavailable")
            if (self.arbiter is None or getattr(self.arbiter, "monitoring_enabled", None) is not True
                    or self.arbiter.acquire() is not True):
                raise checks._StopError("exclusive_arbitration_unavailable")
            acquired = True
            if (self.serial_guard is None or getattr(self.serial_guard, "monitoring_enabled", None) is not True
                    or self.serial_guard.acquire() is not True):
                raise checks._StopError("serial_exclusivity_unavailable")
            serial_acquired = True
            result["stage"] = "decision"
            evidence = make_evidence(sample)
            decision = self.model.decide(copy.deepcopy(evidence))
            now = self._now()
            self._model_fresh(sample, now)
            if (not isinstance(decision, dict) or decision.get("ok") is not True
                    or decision.get("observation_id") != evidence["observation_id"]):
                raise checks._StopError("model_invalid")
            result["model_choice"] = decision.get("choice")
            if decision.get("choice") != "forward":
                raise checks._StopError("model_not_forward")
            result["stage"] = "post_decision_observation"
            current, now = self._aligned_stationary_sample(sample)
            attempted_arm = True
            result["stage"] = "relay_startup"
            contract = self.relay.arm(checks.RELAY_MOTION_LIMIT_SEC)
            armed = self.relay.armed is True
            if (not isinstance(contract, dict) or contract.get("ready_line") != checks.RELAY_READY_LINE
                    or contract.get("relative_motion_limit") is not True):
                raise checks._StopError("relay_contract")
            if self.relay.send_zero_heartbeat() is not True:
                raise checks._StopError("relay_failure")
            if (self.arbiter.bind_relay(self.relay) is not True or self.arbiter.verify() is not True
                    or self.serial_guard.verify() is not True):
                raise checks._StopError("exclusive_arbitration_lost")
            self._guard_evidence(result)
            if self.relay.send_zero_heartbeat() is not True:
                raise checks._StopError("relay_failure")
            result["stage"] = "relay_ready_observation"
            current, now = self._aligned_stationary_sample(sample)
            if self.relay.send_zero_heartbeat() is not True:
                raise checks._StopError("relay_failure")
            # No nonzero has been sent yet. Start the upper bound after the
            # fresh, stationary observation and retain the relay's independent
            # first-motion deadline as the actual motion limit.
            ready_at = self._now()
            deadline = ready_at + self.duration_sec
            if not checks._finite(deadline) or deadline <= ready_at:
                raise checks._StopError("clock_invalid")
            anchor = previous = current["pose"]
            previous_odom = current["odom_at"]
            first_sent_at = last_sent_at = None
            path, sequence = 0.0, self.relay.sequence + 1
            result["stage"] = "motion_loop"
            while True:
                now = self._now()
                if now >= deadline:
                    raise checks._StopError("time_limit")
                if first_sent_at is not None and now >= first_sent_at + checks.RELAY_MOTION_LIMIT_SEC:
                    raise checks._StopError("motion_limit")
                self._model_fresh(sample, now)
                if first_sent_at is None:
                    # The post-relay sample was just checked; revalidate it
                    # below against the latest scan, pose and guard state.
                    now = self._now()
                else:
                    current, now = self._sample()
                if current["epoch"] != sample["epoch"] or current["odom_at"] < previous_odom:
                    raise checks._StopError("odom_discontinuity")
                step, rotation = checks._delta(previous, current["pose"])
                displacement, yaw = checks._delta(anchor, current["pose"])
                elapsed = current["odom_at"] - previous_odom
                dx, dy = current["pose"][0] - previous[0], current["pose"][1] - previous[1]
                progress = math.cos(previous[2]) * dx + math.sin(previous[2]) * dy
                if (not all(checks._finite(v) for v in (step, rotation, displacement, yaw, progress))
                        or elapsed == 0 and (step > 0 or rotation > 0)
                        or progress < 0 or step > checks.MAX_SPEED_MPS * elapsed + 0.002 or yaw > 0.03):
                    raise checks._StopError("odom_discontinuity")
                path += step
                previous, previous_odom = current["pose"], current["odom_at"]
                result["odom_displacement_m"], result["odom_path_m"] = displacement, path
                if max(displacement, path) + self.speed_mps * self.period_sec >= checks.MAX_DISTANCE_M:
                    raise checks._StopError("distance_limit")
                if self.arbiter.verify() is not True:
                    raise checks._StopError("exclusive_arbitration_lost")
                if self.serial_guard.verify() is not True:
                    raise checks._StopError("serial_exclusivity_lost")
                self._guard_evidence(result)
                if self.relay.healthy() is not True:
                    raise checks._StopError("relay_failure")
                if (self.source.is_current_epoch(current["epoch"]) is not True
                        or self.source.is_current_sample(current) is not True):
                    raise checks._StopError("sensor_unavailable")
                now = self._now()
                reason = self._validate(current, now)
                if reason:
                    raise checks._StopError(reason)
                if (self.source.is_current_epoch(current["epoch"]) is not True
                        or self.source.is_current_sample(current) is not True):
                    raise checks._StopError("sensor_unavailable")
                now = self._now()
                if now >= deadline:
                    raise checks._StopError("time_limit")
                if first_sent_at is not None and now >= first_sent_at + checks.RELAY_MOTION_LIMIT_SEC:
                    raise checks._StopError("motion_limit")
                self._model_fresh(sample, now)
                if not _live_timestamps_fresh(current, now):
                    raise checks._StopError("sensor_unavailable")
                if last_sent_at is None and self.relay.last_stamp is not None and now == self.relay.last_stamp:
                    self.clock.sleep(0.001)
                    now = self._now()
                    if now >= deadline or not _live_timestamps_fresh(current, now):
                        raise checks._StopError("sensor_unavailable")
                    if (self.source.is_current_epoch(current["epoch"]) is not True
                            or self.source.is_current_sample(current) is not True):
                        raise checks._StopError("sensor_unavailable")
                    self._model_fresh(sample, now)
                if last_sent_at is not None and now <= last_sent_at:
                    raise checks._StopError("clock_invalid")
                if first_sent_at is None:
                    if now + checks.RELAY_MOTION_LIMIT_SEC <= now:
                        raise checks._StopError("clock_invalid")
                    first_sent_at = now
                result["motion_requested"] = True
                if self.relay.send_frame(sequence, now, self.speed_mps, 0.0) is not True:
                    raise checks._StopError("relay_failure")
                sequence += 1
                last_sent_at = now
                now = self._now()
                if now >= deadline:
                    raise checks._StopError("time_limit")
                self.clock.sleep(min(self.period_sec, deadline - now))
        except checks._StopError as exc:
            result["reason"] = str(exc)
        except BaseException:
            result["reason"] = "exception"
        finally:
            relay_exited = not attempted_arm
            try:
                if armed:
                    result["stop_requested"] = self.relay.stop_and_latch() is True
            except BaseException:
                result["stop_requested"] = False
            try:
                if attempted_arm:
                    relay_exited = self.relay.close() is True
            except BaseException:
                relay_exited = False
            if attempted_arm:
                result["relay_evidence"] = copy.deepcopy(self.relay.evidence)
                if not relay_exited:
                    result["trigger_reason"] = result["reason"]
                    result["reason"] = "relay_unreaped"
            try:
                if serial_acquired and relay_exited:
                    self.serial_guard.release()
            except BaseException:
                pass
            try:
                if acquired and relay_exited:
                    self.arbiter.release()
            except BaseException:
                pass
            if self._last_depth_assessment is not None:
                result["depth_assessment"] = copy.deepcopy(self._last_depth_assessment)
            if self._last_diagnostic is not None:
                result["diagnostic"] = copy.deepcopy(self._last_diagnostic)
            for name, guard in (("ros_monitor_diagnostic", self.arbiter),
                                ("serial_monitor_diagnostic", self.serial_guard)):
                try:
                    diagnose = getattr(guard, "diagnostic", None)
                    if callable(diagnose):
                        result[name] = diagnose()
                except Exception:
                    result[name] = {"reason": "diagnostic_unavailable"}
        return result


if __name__ == "__main__":
    sys.stderr.write("Review-only adapter; no robot connection or motion was started.\n")
    sys.exit(2)
