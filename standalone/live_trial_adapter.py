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

import collections
import copy
import errno
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import select
import subprocess
import sys
import threading
import time

import forward_trial as checks
from async_safety_sampler import WAITING_REPLACEMENT
from relay import _monotonic
from split_sensor_validation import PENDING_PROOF

_INITIAL_COVERAGE_PENDING = object()


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

    def __init__(self, observation, jpeg_encoder, take_scene_ownership=False, projection_only_after_model=False,
                 recheck_snapshot=True, predecision_warmup=False, serialize_processing=False):
        if (type(take_scene_ownership) is not bool or type(projection_only_after_model) is not bool
                or type(recheck_snapshot) is not bool or type(predecision_warmup) is not bool
                or type(serialize_processing) is not bool):
            raise ValueError("scene ownership must be explicit")
        self.observation = observation
        self.jpeg_encoder = jpeg_encoder
        # Only use this when the source allocates a new unshared scene on
        # every snapshot, as RosObservation's fusion snapshot does.
        self.take_scene_ownership = take_scene_ownership
        self.projection_only_after_model = projection_only_after_model
        # Only a consumer that separately rechecks current inputs before
        # every motion command may explicitly defer this expensive check.
        self.recheck_snapshot = recheck_snapshot
        self.predecision_warmup = predecision_warmup
        self.serialize_processing = serialize_processing
        self.last_issue = None
        self._snapshot_issues = threading.local()
        self._processing_issue = None
        self._processing_blocked = False
        self._processing_latched = False
        self._processing_blocked_reason = None
        self._requires_validated_recovery = False
        self._processing_lock = threading.RLock() if serialize_processing else None
        self._encode_images = True

    def _acquire_processing(self):
        if self._processing_lock is None:
            return True
        try:
            started = _monotonic()
            if not checks._finite(started) or started < 0:
                return False
            deadline = started + .1
            for _ in range(100):
                if self._processing_lock.acquire(False):
                    return True
                now = _monotonic()
                if not checks._finite(now) or now < started:
                    return False
                remaining = deadline - now
                if remaining <= 0:
                    return False
                time.sleep(min(.002, remaining))
            return False
        except Exception:  # noqa: BLE001 - inability to lock cannot authorize motion
            return False

    def _release_processing(self):
        if self._processing_lock is not None:
            self._processing_lock.release()

    def _set_snapshot_issue(self, issue):
        self._snapshot_issues.value = issue
        self.last_issue = issue

    def snapshot_issue(self):
        return getattr(self._snapshot_issues, 'value', self.last_issue)

    def begin_command_check(self):
        """Serialize the final proof and command submission with perception."""
        if not self._acquire_processing():
            self._processing_issue = 'processing_busy'
            return False
        if self._processing_blocked:
            self._processing_issue = self._processing_blocked_reason or 'snapshot_unavailable'
            self._release_processing()
            return False
        return True

    def end_command_check(self):
        self._release_processing()

    def enable_safety_snapshots(self):
        """Skip repeated JPEG encoding once model evidence has been captured."""
        if self.projection_only_after_model:
            enable_projection_only = getattr(self.observation, "enable_projection_only", None)
            disable_projection_only = getattr(self.observation, "disable_projection_only", None)
            if not callable(enable_projection_only) or not callable(disable_projection_only):
                raise ValueError("observation cannot provide projected depth")
            enable_projection_only()
        self._encode_images = False

    def restore_model_snapshots(self):
        if self.projection_only_after_model:
            self.observation.disable_projection_only()
        self._encode_images = True

    def snapshot(self):
        if not self._acquire_processing():
            self._set_snapshot_issue('source_processing_busy')
            return None
        try:
            result = self._snapshot_unlocked()
            if self.serialize_processing:
                if not isinstance(result, dict):
                    self._processing_blocked = True
                    self._processing_blocked_reason = self.snapshot_issue() or 'snapshot_unavailable'
                    if not _transient_snapshot_gap(self):
                        self._processing_latched = True
                elif not self._processing_latched and not self._requires_validated_recovery:
                    self._processing_blocked = False
                    self._processing_blocked_reason = None
            return result
        except BaseException:
            if self.serialize_processing:
                self._processing_blocked = True
                self._processing_latched = True
                self._processing_blocked_reason = 'snapshot_exception'
            raise
        finally:
            self._release_processing()

    def validated_snapshot(self, validate, now):
        """Keep snapshot acquisition and its background validation atomic."""
        if not self._acquire_processing():
            self._set_snapshot_issue('source_processing_busy')
            return None, None, None
        try:
            sample = self._snapshot_unlocked()
            if not isinstance(sample, dict):
                self._processing_blocked = True
                self._processing_blocked_reason = self.snapshot_issue() or 'snapshot_unavailable'
                if not _transient_snapshot_gap(self):
                    self._processing_latched = True
                return None, None, None
            reason, assessment = validate(sample, now())
            if reason is WAITING_REPLACEMENT:
                return sample, reason, None
            if reason is _INITIAL_COVERAGE_PENDING:
                self._processing_blocked = True
                self._requires_validated_recovery = True
                self._processing_blocked_reason = 'source_initial_coverage_pending'
                self._set_snapshot_issue('source_initial_coverage_pending')
                return None, None, None
            if reason is not None or not isinstance(assessment, dict):
                self._processing_latched = True
                self._processing_blocked = True
                self._processing_blocked_reason = (reason if isinstance(reason, str) and 0 < len(reason) <= 128
                                                   else 'assessment_invalid')
            elif not self._processing_latched:
                self._processing_blocked = False
                self._processing_blocked_reason = None
                self._requires_validated_recovery = False
            return sample, reason, assessment
        except BaseException:
            self._processing_blocked = True
            self._processing_latched = True
            self._processing_blocked_reason = 'snapshot_exception'
            raise
        finally:
            self._release_processing()

    def _snapshot_unlocked(self):
        try:
            sample = self.observation.snapshot()
            if not isinstance(sample, dict):
                issue = getattr(self.observation, "snapshot_last_issue", None)
                self._set_snapshot_issue("source_" + issue if isinstance(issue, str) else "source_unavailable")
                return None
            scan, color = sample["scan"], sample["color"]
            result = {key: (value if self.take_scene_ownership else _copy_scene(value))
                      if key == "scene" else copy.deepcopy(value)
                      for key, value in sample.items() if key not in ("scan", "color")}
            # These fields must come from the actual observation, never defaults.
            result["depth_pose"] = copy.deepcopy(sample["depth_pose"])
            result["state"]["vehicle_state"] = sample["state"]["vehicle_state"]
            result["state"]["control_mode"] = sample["state"]["control_mode"]
            if self._encode_images:
                result["image_jpeg"] = self.jpeg_encoder(color)
            result["raw_scan"] = {"stamp": sample["scan_at"], "frame_id": scan.header.frame_id,
                                  "angle_min": scan.angle_min, "angle_increment": scan.angle_increment,
                                  "range_min": scan.range_min, "range_max": scan.range_max,
                                  "time_increment": scan.time_increment, "ranges": list(scan.ranges)}
            if self.recheck_snapshot and not self.observation.is_current_sample(result):
                self._set_snapshot_issue("current_after_copy")
                return None
            self._set_snapshot_issue(None)
            return result
        except Exception:  # malformed observations are unavailable
            self._set_snapshot_issue("adapter_error")
            return None

    def is_current_sample(self, sample):
        if not self._acquire_processing():
            self._processing_issue = 'processing_busy'
            return False
        try:
            self._processing_issue = None
            if self._processing_blocked:
                self._processing_issue = self._processing_blocked_reason or 'snapshot_unavailable'
                return False
            return self.observation.is_current_sample(sample) is True
        except Exception:
            return False
        finally:
            self._release_processing()

    def current_issue(self):
        return self._processing_issue or getattr(self.observation, "current_last_issue", None)

    def current_limit_kind(self):
        return getattr(self.observation, "current_limit_kind", None)

    def prepare_command_evidence(self, sample):
        return self.observation.prepare_command_evidence(sample)

    def read_command_evidence(self, sample):
        split = getattr(self.observation, '_split_validation', None)
        if split is not None:
            self._split_revision = split.revision
            proof, issue = split.read(sample)
            return (proof['valid_until'], None) if proof is not None else (None, issue)
        return self.observation.read_command_evidence(sample, sample.get('_command_evidence'))

    def wait_for_prepared_evidence(self, timeout):
        split = getattr(self.observation, '_split_validation', None)
        if split is not None:
            split.wait_for_change(getattr(self, '_split_revision', None), timeout)
        else:
            time.sleep(timeout)

    def updates_prepared_evidence(self):
        return getattr(self.observation, '_split_validation', None) is not None

    def prepare_initial_command_evidence(self, sample):
        return self.observation.prepare_initial_command_evidence(sample)

    def seed_split_validation(self, sample, proof):
        self.observation.seed_split_validation(sample, proof)

    def commit_split_validation(self, sample):
        split = getattr(self.observation, '_split_validation', None)
        if split is not None:
            split.commit(sample)

    def current_valid_until(self, sample):
        if not self._acquire_processing():
            self._processing_issue = 'processing_busy'
            return None
        try:
            self._processing_issue = None
            if self._processing_blocked:
                self._processing_issue = self._processing_blocked_reason or 'snapshot_unavailable'
                return None
            method = getattr(self.observation, "current_valid_until", None)
            if callable(method):
                return method(sample)
            if self.observation.is_current_sample(sample) is not True:
                return None
            # Injectable fixtures can expose only sample timestamps. A real
            # ROS observer supplies the latest odom and scan deadlines above.
            return min(sample['odom_at'] + 0.1, sample['scan_at'] + 0.3,
                       sample['status_at'] + 1.0, sample['snapshot_at'] + 0.25,
                       sample['depth_at'] + 0.5, sample['color_at'] + 0.5)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return None
        finally:
            self._release_processing()

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
        except Exception as exc:
            rejected['error_class'] = type(exc).__name__
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


def relay_ready_line(motion_limit_sec):
    """Return the exact handshake for one of the supported bounded trials."""
    if not checks._finite(motion_limit_sec) or motion_limit_sec not in (0.3, 2.0):
        raise ValueError("unsupported motion limit")
    return ("READY motion_limit_sec=%.6f\n" % motion_limit_sec).encode("ascii")


class RelayProcessTransport(object):
    """Bounded pipe transport; starts the relay only when start() is called."""
    def __init__(self, clock, popen=None):
        self.clock = clock
        self.popen = popen if popen is not None else subprocess.Popen
        self.process = None
        self.buffer = b""
        self.exited = None

    def start(self, max_motion_sec):
        if self.process is not None:
            raise ValueError("invalid relay startup")
        relay_ready_line(max_motion_sec)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relay.py")
        command = ["/usr/bin/python2", path, "--require-intent-deadline",
                   "--max-motion-sec", "%.6f" % max_motion_sec]
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

    def __init__(self, transport, clock, motion_limit_sec=0.3):
        relay_ready_line(motion_limit_sec)
        self.transport, self.clock = transport, clock
        self.motion_limit_sec = motion_limit_sec
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
        if self.owner is not None or max_motion_sec != self.motion_limit_sec:
            return None
        self.owner = threading.current_thread().ident
        try:
            self.transport.start(max_motion_sec)
            ready = self.transport.read_line(3.5)
            if ready != relay_ready_line(self.motion_limit_sec):
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

    def _write(self, sequence, stamp, linear, angular, valid_until):
        frame = {"sequence": sequence, "stamp": stamp, "linear": linear, "angular": angular,
                 "expires_at": valid_until}
        data = (json.dumps(frame, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
        remaining = valid_until - self.clock.now()
        if not checks._finite(remaining) or remaining <= 0:
            self.faulted = True
            return False
        timeout = min(0.01 if linear else 0.05, remaining)
        if self.transport.write_line(data, timeout) is not True:
            self.faulted = True
            return False
        self.sequence, self.last_stamp = sequence, stamp
        return True

    def send_frame(self, sequence, stamp, linear, angular, valid_until=None):
        if valid_until is None and checks._finite(stamp):
            valid_until = stamp + 0.16
        if (not self._owner() or not self.armed or self.faulted
                or isinstance(sequence, bool) or not isinstance(sequence, checks.INTEGER_TYPES)
                or sequence != self.sequence + 1 or not checks._finite(stamp)
                or (self.last_stamp is not None and stamp <= self.last_stamp)
                or not checks._finite(linear) or not 0 < linear <= checks.MAX_SPEED_MPS
                or not checks._finite(angular) or angular != 0.0
                or not checks._finite(valid_until) or not stamp < valid_until <= stamp + 0.2):
            return False
        try:
            written = self._write(sequence, stamp, linear, angular, valid_until)
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
                    and self._write(self.sequence + 1, now, 0.0, 0.0, now + 0.19))
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
            sent = self._write(self.sequence + 1, now, 0.0, 0.0, now + 0.19)
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


def _live_timestamps_fresh(sample, now, latest_odom_rechecked=False):
    """Allow bounded processing latency; a waived odom age requires a live recheck before sending."""
    try:
        ages = {"snapshot_at": 0.25, "color_at": 0.5, "scan_at": 0.5,
                "odom_at": 0.25, "status_at": 1.0, "depth_at": 0.5}
        if latest_odom_rechecked:
            del ages["odom_at"]
        return isinstance(sample, dict) and all(checks._fresh(sample[name], now, age)
                                                for name, age in ages.items())
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _transient_snapshot_gap(source):
    """Retry only timing gaps; the previous proof keeps its original expiry."""
    issue_method = getattr(source, 'snapshot_issue', None)
    issue = issue_method() if callable(issue_method) else getattr(source, 'last_issue', None)
    return issue in ('source_sensor_alignment', 'source_missing_color', 'source_missing_info',
                     'source_refreshed_odom', 'source_snapshot_exception_ExtrapolationException',
                     'source_processing_busy', 'source_initial_coverage_pending')


class StrictDepthGate(object):
    """Default legacy image-grid gate; it does not certify clearance."""
    def assess(self, sample, now):
        reason = _depth_reason(sample)
        return {"candidate": reason is None, "reason": reason or "depth_grid_accepted",
                "depth_at": sample.get("depth_at"), "geometric_clearance_verified": False}


def _snapshot_without_depth_gate(sample, now, diagnostic=None, require_image=True):
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
        if require_image:
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
        reason, _ = checks._raw_scan(sample, need_canonical=False)
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


def _validated_front_expiry(sample):
    """Expiry of the already validated 0.6 m forward cross-section.

    The caller must first validate the complete scene with ``assess_front``.
    Removing expired rays cannot increase coverage. Find the first acquisition
    time whose removal would break the accepted maximum-gap bound, keeping the
    latest scan's separate freshness limit. Hits were checked independently.
    """
    try:
        scan_at, snapshot_at = sample['scan_at'], sample['snapshot_at']
        native = getattr(checks, '_native_scan', None)
        if native is not None:
            try:
                accelerated = native.fast_front_expiry(sample['scene']['rays'], scan_at, snapshot_at)
                if (type(accelerated) is float and checks._finite(accelerated)
                        and 0 <= accelerated <= scan_at + .5):
                    return accelerated
            except Exception:  # noqa: BLE001 - use the existing checked scalar path
                pass
        crossings = []
        for ray in sample['scene']['rays']:
            stamp = ray['stamp']
            if (not checks._finite(stamp) or not 0 <= snapshot_at - stamp <= 0.65
                    or not 0 <= scan_at - stamp <= 0.4):
                continue
            start, end = ray['start_base'], ray['hit_base']
            if start[0] <= 0.60 < end[0]:
                y = start[1] + (0.60 - start[0]) / (end[0] - start[0]) * (end[1] - start[1])
                if not checks._finite(y):
                    return None
                if -0.15 <= y <= 0.15:
                    crossings.append((y, stamp))
        if not crossings or not any(stamp == scan_at for _, stamp in crossings):
            return None
        def largest_gap(values):
            ordered = sorted(values)
            if not ordered:
                return 0.3
            gaps = [ordered[0] + 0.15, 0.15 - ordered[-1]]
            gaps.extend(ordered[index] - ordered[index - 1] for index in range(1, len(ordered)))
            return max(gaps)
        if (largest_gap([y for y, _ in crossings]) > 0.035 + 1e-12
                or largest_gap([y for y, stamp in crossings if stamp == scan_at]) > 0.07 + 1e-12):
            return None
        deadline = scan_at + 0.5
        for stamp in sorted(set(stamp for _, stamp in crossings)):
            expires = stamp + 0.65
            if expires >= deadline:
                break
            remaining = [y for y, other in crossings if other > stamp]
            if largest_gap(remaining) > 0.035 - 1e-12:
                return expires
        return deadline
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None


def _assess_sample(sample, now, depth_gate, require_image):
    """Validate one owned sample without mutating another thread's state."""
    detail = {}
    reason = _snapshot_without_depth_gate(sample, now, detail, require_image=require_image)
    if reason:
        return reason, None, detail
    try:
        assessment = depth_gate.assess(sample, now)
        if (not isinstance(assessment, dict) or type(assessment.get("candidate")) is not bool
                or not isinstance(assessment.get("reason"), checks.STRING_TYPES)
                or not assessment["reason"] or len(assessment["reason"]) > 128
                or not checks._finite(assessment.get("depth_at"))
                or assessment["depth_at"] != sample["depth_at"]
                or assessment.get("geometric_clearance_verified") is not False):
            return "depth_gate_invalid", None, detail
        allowed = (('depth_grid_accepted', 'sampled_depth_corridor') if assessment['candidate'] else
                   ('depth_unavailable', 'depth_grid_incomplete', 'depth_near_or_unknown',
                    'depth_obstacle', 'depth_unclassified'))
        if assessment['reason'] not in allowed:
            return 'depth_gate_invalid', None, detail
        keep = ('candidate', 'reason', 'depth_at', 'geometric_clearance_verified',
                'valid_until', 'valid_count', 'ground_count', 'unobserved_space',
                'unknown_space', 'calibration_verified')
        assessment = {name: assessment[name] for name in keep if name in assessment}
        if 'valid_until' in assessment and not checks._finite(assessment['valid_until']):
            return 'depth_gate_invalid', None, detail
        for name in ('unobserved_space', 'unknown_space', 'calibration_verified'):
            if name in assessment and type(assessment[name]) is not bool:
                return 'depth_gate_invalid', None, detail
        for name in ('valid_count', 'ground_count'):
            if name in assessment and (isinstance(assessment[name], bool)
                    or not isinstance(assessment[name], checks.INTEGER_TYPES) or assessment[name] < 0):
                return 'depth_gate_invalid', None, detail
        if not assessment['candidate']:
            return assessment['reason'], assessment, detail
        # Do not extend a gap-filled front proof beyond the first expiry that
        # actually breaks the coverage. Redundant older rays need not shorten
        # a proof that is already supported by newer readings.
        coverage_until = _validated_front_expiry(sample)
        if not checks._finite(coverage_until):
            detail['check'] = 'front_coverage_expiry'
            return 'sensor_unavailable', None, detail
        previous_until = assessment.get('valid_until', coverage_until)
        if not checks._finite(previous_until) or min(previous_until, coverage_until) <= now:
            detail['check'] = 'front_coverage_expiry'
            return 'sensor_unavailable', None, detail
        assessment['depth_valid_until'] = assessment.get('valid_until')
        assessment['front_coverage_valid_until'] = coverage_until
        assessment['valid_until'] = min(previous_until, coverage_until)
        return None, assessment, detail
    except Exception:
        return "depth_gate_invalid", None, detail


class LiveTrial(object):
    """Candidate entry point; external ROS and serial monitors are required."""
    MAX_DURATION_SEC = 0.6
    MOTION_LIMIT_SEC = 0.3
    MAX_DISTANCE_M = 0.02
    ASYNC_SAMPLING = False
    ARM_BEFORE_MODEL = False
    IMMEDIATE_SYNC_SAMPLING = False
    DISTANCE_RESERVE_SEC = 0.0
    PREPARED_COMMANDS = False

    def __init__(self, source, model, relay, arbiter, clock, enable_motion=False,
                 duration_sec=0.6, speed_mps=0.02, period_sec=0.05, serial_guard=None, depth_gate=None,
                 operator_site_report=None):
        if (not all(checks._finite(x) for x in (duration_sec, speed_mps, period_sec))
                or not 0 < duration_sec <= self.MAX_DURATION_SEC or not 0 < speed_mps <= 0.02
                or not 0 < period_sec <= 0.05 or type(enable_motion) is not bool):
            raise ValueError("invalid trial bounds")
        self.source, self.model, self.relay, self.arbiter, self.clock = source, model, relay, arbiter, clock
        self.serial_guard = serial_guard
        self.depth_gate = depth_gate if depth_gate is not None else StrictDepthGate()
        self.operator_site_report = copy.deepcopy(operator_site_report)
        self._last_depth_assessment = None
        self._last_diagnostic = None
        self._validation_detail = {}
        self._image_required = True
        self._stationary_proof_at = None
        self._stationary_proof_key = None
        self.enable_motion = enable_motion
        self.duration_sec, self.speed_mps, self.period_sec = duration_sec, speed_mps, period_sec
        self._used = False
        self._lock = threading.Lock()
        self._last_time = None
        self._motion_started = False

    def _distance_limit_reached(self, displacement, path):
        reserve = self.speed_mps * max(self.period_sec, self.DISTANCE_RESERVE_SEC)
        return max(displacement, path) + reserve >= self.MAX_DISTANCE_M

    def _now(self):
        now = self.clock.now()
        if not checks._finite(now) or now < 0 or (self._last_time is not None and now < self._last_time):
            raise checks._StopError("clock_invalid")
        self._last_time = now
        return now

    def _sample(self, defer_current_check=False):
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
        if (not self._motion_started and (self.source.is_current_epoch(sample["epoch"]) is not True
                or not defer_current_check and self.source.is_current_sample(sample) is not True)):
            issue = getattr(self.source, "current_issue", None)
            self._last_diagnostic = {"step": "latest_sample_recheck",
                                     "detail": issue() if callable(issue) else None}
            raise checks._StopError("sensor_unavailable")
        self._last_diagnostic = None
        return sample, now

    def _validate(self, sample, now):
        reason, assessment, self._validation_detail = _assess_sample(
            sample, now, self.depth_gate, self._image_required)
        if assessment is not None:
            self._last_depth_assessment = assessment
        return reason

    def _model_fresh(self, before, now):
        if (not checks._fresh(before["color_at"], now, checks.MODEL_MAX_AGE_SEC)
                or not checks._fresh(before["scan_at"], now, checks.MODEL_MAX_AGE_SEC)):
            raise checks._StopError("model_stale")

    def _request_forward_decision(self, sample, result):
        result["stage"] = "decision"
        evidence = make_evidence(sample)
        decision = self.model.decide(copy.deepcopy(evidence))
        now = self._now()
        self._model_fresh(sample, now)
        if (not isinstance(decision, dict) or decision.get("ok") is not True
                or decision.get("observation_id") != evidence["observation_id"]):
            name = decision.get('error_class') if isinstance(decision, dict) else None
            if isinstance(name, str) and re.match(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$', name):
                self._last_diagnostic = {'step': 'decision', 'error_class': name}
            raise checks._StopError("model_invalid")
        result["model_choice"] = decision.get("choice")
        if decision.get("choice") != "forward":
            raise checks._StopError("model_not_forward")

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
            self._trace_failed(trace, start, first_end, "stationary_trace_head", before)
        # The first history can take time to copy. Read the intervening tail so
        # odometry during that processing interval is checked as well.
        tail = self.source.stationary_trace(first_end, tail_end)
        now = self._now()
        self._model_fresh(before, now)
        if (self.source.is_current_epoch(before["epoch"]) is not True
                or not checks._stationary_trace_valid(tail, first_end, now,
                                                       before["epoch"], before["image_pose"], max_gap_sec=0.15)):
            self._trace_failed(tail, first_end, now, "stationary_trace_tail", before)
        self._stationary_proof_at = now
        self._stationary_proof_key = key
        return now

    def _trace_failed(self, trace, start, now, step, before):
        self._last_diagnostic = {"step": step, "retryable": False}
        try:
            gaps = [trace[i]["stamp"] - trace[i - 1]["stamp"] for i in range(1, len(trace))]
            stats = {"samples": len(trace), "max_gap_sec": max(gaps) if gaps else None,
                     "start_gap_sec": start - trace[0]["stamp"] if trace else None,
                     "end_gap_sec": now - trace[-1]["stamp"] if trace else None}
            if all(value is None or checks._finite(value) for value in stats.values()):
                self._last_diagnostic.update(stats)
            # Retry only an absent history or a complete, stationary prefix
            # whose tail may still be queued. A bad pose, speed, timestamp,
            # sequence or epoch must not disappear on a later read.
            if self.source.is_current_epoch(before['epoch']) is True:
                empty_history = isinstance(trace, (tuple, list)) and not trace
                delayed_tail = (trace and 0.15 < now - trace[-1]['stamp'] <= 0.3
                                and checks._stationary_trace_valid(trace, start, trace[-1]['stamp'],
                                                                   before['epoch'], before['image_pose'],
                                                                   max_gap_sec=0.15))
                if empty_history or delayed_tail:
                    self._last_diagnostic['retryable'] = True
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
        raise checks._StopError("sensor_unavailable")

    def _stationary_with_retry(self, before):
        """While stopped, allow delayed odometry to complete the same proof."""
        deadline = self._now() + 0.3
        for _ in range(31):
            try:
                now = self._stationary(before, self._now())
                self._last_diagnostic = None
                return now
            except checks._StopError as exc:
                now = self._now()
                if (str(exc) != 'sensor_unavailable' or not isinstance(self._last_diagnostic, dict)
                        or self._last_diagnostic.get('step') not in ('stationary_trace_head',
                                                                     'stationary_trace_tail')
                        or self._last_diagnostic.get('retryable') is not True
                        or now >= deadline):
                    raise
                if self.relay.armed and (self.relay.healthy() is not True
                                         or self.relay.send_zero_heartbeat() is not True):
                    raise checks._StopError('relay_failure')
                self.clock.sleep(min(0.01, deadline - now))
                if self._now() <= now:
                    raise checks._StopError('clock_invalid')
        raise checks._StopError('sensor_unavailable')

    def _aligned_stationary_sample(self, before):
        deadline = self._now() + 3.0
        if self.relay.armed and (self.relay.healthy() is not True
                                 or self.relay.send_zero_heartbeat() is not True):
            raise checks._StopError("relay_failure")
        # Check the potentially long inference history before obtaining the
        # next sample. Later checks extend this proof with the short new tail.
        self._stationary_with_retry(before)
        for _ in range(301):
            now = self._now()
            self._model_fresh(before, now)
            if self.relay.armed and (self.relay.healthy() is not True
                                     or self.relay.send_zero_heartbeat() is not True):
                raise checks._StopError("relay_failure")
            try:
                # A final current-sensor check follows the stationary history;
                # avoid doing the same expensive check twice before moving.
                sample, now = self._sample(defer_current_check=True)
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
            now = self._stationary_with_retry(before)
            fresh = _live_timestamps_fresh(sample, now)
            current = (self.source.is_current_epoch(sample['epoch']) is True
                       and self.source.is_current_sample(sample) is True)
            if not fresh or not current:
                self._last_diagnostic = {"step": "post_trace_recheck", "timestamps_fresh": fresh,
                                         "current_sample": current,
                                         "snapshot_age_sec": now - sample["snapshot_at"]}
                issue = getattr(self.source, "current_issue", None)
                if not current and callable(issue):
                    self._last_diagnostic["detail"] = issue()
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
        acquired = serial_acquired = attempted_arm = armed = gc_suspended = safety_mode_enabled = False
        sampler = None
        sampler_started = False
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
            if not self.ARM_BEFORE_MODEL:
                self._request_forward_decision(sample, result)
                enable_safety_snapshots = getattr(self.source, "enable_safety_snapshots", None)
                if callable(enable_safety_snapshots):
                    safety_mode_enabled = True
                    enable_safety_snapshots()
                self._image_required = False
                result["stage"] = "post_decision_observation"
                current, now = self._aligned_stationary_sample(sample)
            # CPython's full cyclic collection can block the sensor and relay
            # threads longer than a control-intent lifetime on this host.
            # Collect while stopped, defer only during the bounded trial, then
            # restore the caller's original setting after the relay exits.
            if gc.isenabled():
                gc.collect()
                gc_suspended = True
                gc.disable()
            attempted_arm = True
            result["stage"] = "relay_startup"
            contract = self.relay.arm(self.MOTION_LIMIT_SEC)
            armed = self.relay.armed is True
            if (not isinstance(contract, dict) or contract.get("ready_line") != relay_ready_line(self.MOTION_LIMIT_SEC)
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
            if self.ARM_BEFORE_MODEL:
                # Capture the model evidence only after the relay is primed
                # at zero, so startup latency does not age the decision.
                sample = current
                self._stationary_proof_at = self._stationary_proof_key = None
                enable_safety_snapshots = getattr(self.source, "enable_safety_snapshots", None)
                if callable(enable_safety_snapshots):
                    safety_mode_enabled = True
                    enable_safety_snapshots()
                warmup = None
                try:
                    if getattr(self.source, 'predecision_warmup', False) is True:
                        from async_safety_sampler import DecisionWarmup
                        warmup = DecisionWarmup(self.source)
                        warmup.start()
                    self._request_forward_decision(sample, result)
                finally:
                    if warmup is not None and not warmup.close(2.0):
                        # Do not concurrently restore or reuse a source whose
                        # outstanding snapshot did not finish.
                        safety_mode_enabled = False
                        raise checks._StopError('predecision_warmup_busy')
                self._image_required = False
                result["stage"] = "post_decision_observation"
                current, now = self._aligned_stationary_sample(sample)
            if self.relay.send_zero_heartbeat() is not True:
                raise checks._StopError("relay_failure")
            # No nonzero has been sent yet. Start the upper bound after the
            # fresh, stationary observation and retain the relay's independent
            # first-motion deadline as the actual motion limit.
            ready_at = self._now()
            # While still stationary, permit extra fresh-sample attempts.
            # Model freshness and every per-command expiry remain unchanged.
            pre_motion_retry_until = ready_at + 4.0
            deadline = ready_at + self.duration_sec
            if not checks._finite(deadline) or deadline <= ready_at:
                raise checks._StopError("clock_invalid")
            anchor = previous = current["pose"]
            previous_odom = current["odom_at"]
            first_sent_at = last_sent_at = None
            last_valid_until = None
            short_window_rechecked = False
            refresh_prepared = self.PREPARED_COMMANDS and self.source.updates_prepared_evidence()
            initial_sample_used = None
            path, sequence = 0.0, self.relay.sequence + 1
            result["stage"] = "motion_loop"
            result["motion_timing"] = {"last_sample_sec": None, "max_sample_sec": 0.0,
                                       "last_submit_interval_sec": None}
            renewal_checks = collections.deque(maxlen=128)
            if self.ASYNC_SAMPLING and getattr(self.source, 'serialize_processing', False) is True:
                result['initial_preparation'] = {'waiting': 0, 'same_sample': 0, 'ready': 0,
                                                 'guard_retries': 0, 'short_window': 0,
                                                 'largest_window_sec': None, 'last_guard_issue': None,
                                                 'sensor_rejections': {}}
            if self.ASYNC_SAMPLING:
                from async_safety_sampler import AsyncSafetySampler
                stationary_until = [self._stationary_proof_at]

                def validate_background(candidate, captured_at):
                    reason, assessment, detail = _assess_sample(candidate, captured_at, self.depth_gate, False)
                    if (refresh_prepared and reason == 'sensor_unavailable'
                            and detail.get('check') == 'front_forward_coverage_incomplete'):
                        return WAITING_REPLACEMENT, None
                    if (not self._motion_started and reason == 'sensor_unavailable'
                            and detail.get('check') == 'front_forward_coverage_incomplete'):
                        return _INITIAL_COVERAGE_PENDING, None
                    if reason is None and self.PREPARED_COMMANDS:
                        if refresh_prepared:
                            candidate['_front_coverage_valid_until'] = assessment['front_coverage_valid_until']
                        if not self._motion_started:
                            start = stationary_until[0]
                            end = self.clock.now()
                            trace = self.source.stationary_trace(start, end)
                            if not self._motion_started:
                                if not checks._stationary_trace_valid(
                                        trace, start, end, sample['epoch'], sample['image_pose'], max_gap_sec=.15):
                                    return 'stationary_trace_invalid', None
                                stationary_until[0] = end
                                candidate['_stationary_checked_at'] = end
                        began = self.clock.now()
                        proof, issue = self.source.prepare_command_evidence(candidate)
                        if refresh_prepared and proof is None and issue == 'split_proof_pending':
                            return WAITING_REPLACEMENT, None
                        if proof is None and issue not in (
                                'scan_generation_changed', 'depth_generation_changed',
                                'depth_replaced_during_retry', 'depth_preprojection_pending',
                                'stale_snapshot', 'timestamps', 'split_proof_pending'):
                            if self._motion_started or issue not in ('new_scan', 'new_depth_unchecked'):
                                return issue or 'sensor_unavailable', None
                        candidate['_command_evidence'] = proof
                        candidate['_command_proof_info'] = (began, self.clock.now(), issue)
                    return reason, assessment
                sampler = AsyncSafetySampler(self.source, validate_background, self.clock,
                                             transient_unavailable=_transient_snapshot_gap)
                if getattr(self.source, 'serialize_processing', False) is True and not refresh_prepared:
                    # Prepare the next checked observation while the relay is
                    # still sending zeros, before any short-lived motion intent.
                    sampler.start()
                    sampler_started = True
            while True:
                now = self._now()
                if now >= deadline:
                    raise checks._StopError("time_limit")
                if first_sent_at is not None and now >= first_sent_at + self.MOTION_LIMIT_SEC:
                    raise checks._StopError("motion_limit")
                self._model_fresh(sample, now)
                if first_sent_at is None:
                    if sampler is not None and sampler_started:
                        pending = sampler.poll()
                        if pending['state'] == 'fault':
                            self._last_diagnostic = {'step': 'background_sample', 'detail': pending['reason']}
                            raise checks._StopError(pending['reason'])
                        if pending['state'] != 'ready':
                            result['initial_preparation']['waiting'] += 1
                            if self.relay.send_zero_heartbeat() is not True:
                                raise checks._StopError('relay_failure')
                            if self.PREPARED_COMMANDS:
                                sampler.wait_next(None, .01)
                            else:
                                self.clock.sleep(.01)
                            continue
                        if pending['sample'] is initial_sample_used and not refresh_prepared:
                            result['initial_preparation']['same_sample'] += 1
                            if self.relay.send_zero_heartbeat() is not True:
                                raise checks._StopError('relay_failure')
                            if self.PREPARED_COMMANDS:
                                sampler.wait_next(current, .01)
                            else:
                                self.clock.sleep(.01)
                            continue
                        current = pending['sample']
                        initial_sample_used = current
                        result['initial_preparation']['ready'] += 1
                        self._last_depth_assessment = pending['assessment']
                        now = self._now() if self.PREPARED_COMMANDS else self._stationary_with_retry(sample)
                        distance, yaw = checks._delta(sample['image_pose'], current['pose'])
                        if (current['epoch'] != sample['epoch'] or distance > .005 or yaw > .01
                                or current['state']['linear_x_mps'] > .005
                                or abs(current['state']['linear_y_mps']) > .005
                                or abs(current['state']['angular_z_radps']) > .01):
                            self._last_diagnostic = {'step': 'background_initial_stationary',
                                                     'epoch_matches': current['epoch'] == sample['epoch'],
                                                     'distance_m': distance, 'yaw_rad': yaw}
                            raise checks._StopError('sensor_unavailable')
                        if not _live_timestamps_fresh(current, now):
                            self.clock.sleep(.01)
                            continue
                        anchor = previous = current['pose']
                        previous_odom = current['odom_at']
                        sequence = self.relay.sequence + 1
                    else:
                        # The post-relay sample was just checked; revalidate it
                        # below against the latest scan, pose and guard state.
                        now = self._now()
                elif sampler is not None:
                    now = self._now()
                    # The producer keeps refreshing in parallel. Poll close
                    # to the publication slot so a valid sample does not
                    # spend most of its odometry lifetime waiting here.
                    next_slot = max(last_sent_at + 0.005,
                                    min(last_sent_at + self.period_sec, last_valid_until - 0.015))
                    early_wait = max(0.0, next_slot - now - 0.005)
                    if early_wait:
                        if refresh_prepared:
                            self.source.wait_for_prepared_evidence(early_wait)
                            next_slot = min(next_slot, max(self._now(), last_sent_at + .012))
                        else:
                            self.clock.sleep(early_wait)
                    pending = sampler.poll()
                    if pending['state'] == 'fault':
                        self._last_diagnostic = {'step': 'background_sample', 'detail': pending['reason']}
                        raise checks._StopError(pending['reason'])
                    if pending['state'] == 'ready':
                        candidate = pending['sample']
                        use_candidate = True
                        if refresh_prepared and candidate is not current:
                            candidate_until, _ = self.source.read_command_evidence(candidate)
                            if candidate_until is None or candidate_until <= last_valid_until:
                                # A staged scene can age while being handed
                                # over. Keep the active scene available until
                                # the replacement can advance its deadline.
                                current_until, _ = self.source.read_command_evidence(current)
                                use_candidate = current_until is None or (
                                    candidate_until is not None and candidate_until > current_until)
                        if use_candidate:
                            current = candidate
                            self._last_depth_assessment = pending['assessment']
                    now = self._now()
                else:
                    # Do not capture the next sample immediately after a
                    # command: its short odometry lifetime would mostly be
                    # consumed while waiting for the next publication slot.
                    # A small early wait still leaves most of the period for
                    # slower real sensing and validation.
                    now = self._now()
                    next_slot = max(last_sent_at + 0.005,
                                    min(last_sent_at + self.period_sec, last_valid_until - 0.015))
                    early_wait = (0.0 if self.IMMEDIATE_SYNC_SAMPLING else
                                  min(0.015, max(0.0, next_slot - now)))
                    if early_wait:
                        self.clock.sleep(early_wait)
                    sample_started = self._now()
                    try:
                        current, now = self._sample()
                    finally:
                        sample_elapsed = self._now() - sample_started
                        result["motion_timing"]["last_sample_sec"] = sample_elapsed
                        result["motion_timing"]["max_sample_sec"] = max(
                            result["motion_timing"]["max_sample_sec"], sample_elapsed)
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
                if self._distance_limit_reached(displacement, path):
                    raise checks._StopError("distance_limit")
                # Prepare the next checked observation while the previous
                # command is active, then wait only for the remaining frame
                # interval. Recheck the live inputs after any wait.
                if last_sent_at is not None:
                    now = self._now()
                    remaining = next_slot - now
                    if remaining > 0:
                        self.clock.sleep(remaining)
                local_serial_limit = getattr(self.serial_guard, "local_command_deadline", None)
                cached_guard_check = refresh_prepared and first_sent_at is not None and callable(local_serial_limit)
                if not cached_guard_check:
                    if self.arbiter.verify() is not True:
                        raise checks._StopError("exclusive_arbitration_lost")
                    if self.serial_guard.verify() is not True:
                        raise checks._StopError("serial_exclusivity_lost")
                    self._guard_evidence(result)
                if self.relay.healthy() is not True:
                    raise checks._StopError("relay_failure")
                current_limit = getattr(self.source, "current_valid_until", None)
                now = self._now()
                if now >= deadline:
                    raise checks._StopError("time_limit")
                if first_sent_at is not None and now >= first_sent_at + self.MOTION_LIMIT_SEC:
                    raise checks._StopError("motion_limit")
                self._model_fresh(sample, now)
                # The final live-sensor check below is mandatory for every
                # command, including the first. It validates the newer odom
                # age and pose, so the older snapshot odom need not also be
                # artificially capped while the full snapshot remains fresh.
                if not _live_timestamps_fresh(current, now, latest_odom_rechecked=True):
                    self._last_diagnostic = {"step": "motion_final_timestamps"}
                    if first_sent_at is None and now < pre_motion_retry_until:
                        self.clock.sleep(0.01)
                        if sampler is not None and sampler_started:
                            continue
                        current, now = self._aligned_stationary_sample(sample)
                        if self.relay.send_zero_heartbeat() is not True:
                            raise checks._StopError("relay_failure")
                        ready_at = self._now()
                        deadline = ready_at + self.duration_sec
                        anchor = previous = current["pose"]
                        previous_odom = current["odom_at"]
                        path, sequence = 0.0, self.relay.sequence + 1
                        continue
                    raise checks._StopError("sensor_unavailable")
                if last_sent_at is None and self.relay.last_stamp is not None and now == self.relay.last_stamp:
                    self.clock.sleep(0.001)
                    now = self._now()
                    if now >= deadline or not _live_timestamps_fresh(current, now, latest_odom_rechecked=True):
                        raise checks._StopError("sensor_unavailable")
                    if (self.source.is_current_epoch(current["epoch"]) is not True
                            or self.source.is_current_sample(current) is not True):
                        raise checks._StopError("sensor_unavailable")
                    self._model_fresh(sample, now)
                if last_sent_at is not None and now <= last_sent_at:
                    raise checks._StopError("clock_invalid")
                if last_sent_at is not None:
                    result["motion_timing"]["last_submit_interval_sec"] = now - last_sent_at
                proposed_first = now if first_sent_at is None else first_sent_at
                if proposed_first + self.MOTION_LIMIT_SEC <= proposed_first:
                    raise checks._StopError("clock_invalid")
                ros_limit = getattr(self.arbiter, "valid_until", None)
                serial_limit = local_serial_limit if cached_guard_check else getattr(self.serial_guard, "valid_until", None)
                limits = {"trial": deadline,
                          "relay": proposed_first + self.MOTION_LIMIT_SEC,
                          "model_color": sample["color_at"] + checks.MODEL_MAX_AGE_SEC,
                          "model_scan": sample["scan_at"] + checks.MODEL_MAX_AGE_SEC,
                          "snapshot": current["snapshot_at"] + 0.25,
                          "scan": current["scan_at"] + 0.5, "status": current["status_at"] + 1.0,
                          "color": current["color_at"] + 0.5, "depth": current["depth_at"] + 0.5}
                gate_limit = self._last_depth_assessment.get(
                    "depth_valid_until" if refresh_prepared and first_sent_at is not None else "valid_until")
                if gate_limit is not None:
                    limits["depth_gate"] = gate_limit
                missing = [name for name, limit in limits.items() if not checks._finite(limit)]
                if missing:
                    self._last_diagnostic = {"step": "intent_deadline", "missing": missing}
                    raise checks._StopError("expiry_unavailable")
                guarded = getattr(self.source, 'serialize_processing', False) is True and not self.PREPARED_COMMANDS
                if guarded:
                    begin_guard = getattr(self.source, 'begin_command_check', None)
                    end_guard = getattr(self.source, 'end_command_check', None)
                    if not callable(begin_guard) or not callable(end_guard) or begin_guard() is not True:
                        issue = getattr(self.source, 'current_issue', None)
                        detail = issue() if callable(issue) else None
                        self._last_diagnostic = {'step': 'command_guard', 'detail': detail}
                        if (first_sent_at is None and sampler is not None and sampler_started
                                and detail in ('source_initial_coverage_pending', 'source_sensor_alignment',
                                               'source_missing_color', 'source_missing_info',
                                               'source_refreshed_odom', 'source_snapshot_exception_ExtrapolationException',
                                               'processing_busy')):
                            result['initial_preparation']['guard_retries'] += 1
                            result['initial_preparation']['last_guard_issue'] = detail
                            self.clock.sleep(.01)
                            continue
                        raise checks._StopError('sensor_unavailable')
                try:
                    check_started = self._now()
                    initial_split = refresh_prepared and first_sent_at is None
                    initial_evidence, prepared_issue = None, None
                    if initial_split:
                        initial_evidence, prepared_issue = self.source.prepare_initial_command_evidence(current)
                        final_sensor_until = initial_evidence['valid_until'] if initial_evidence is not None else None
                        sensor_checked = self._now()
                    elif not self.PREPARED_COMMANDS:
                        final_sensor_until = current_limit(current) if callable(current_limit) else None
                        sensor_checked = self._now()
                    # These calls revalidate the monitor caches. Use the
                    # current deadlines after sensor work, not those read
                    # before it while a new watchdog packet could arrive.
                    limits["ros_monitor"] = ros_limit() if callable(ros_limit) else None
                    limits["serial_monitor"] = serial_limit() if callable(serial_limit) else None
                    if self.PREPARED_COMMANDS and not initial_split:
                        final_sensor_until, prepared_issue = self.source.read_command_evidence(current)
                        if first_sent_at is None and final_sensor_until is not None:
                            start = current.get('_stationary_checked_at')
                            end = self._now()
                            if (not checks._finite(start) or not checks._stationary_trace_valid(
                                    self.source.stationary_trace(start, end), start, end,
                                    sample['epoch'], sample['image_pose'], max_gap_sec=.15)):
                                raise checks._StopError('stationary_trace_invalid')
                            final_sensor_until, prepared_issue = self.source.read_command_evidence(current)
                        sensor_checked = self._now()
                    now = self._now()
                    check = {"sample_at": current["snapshot_at"], "sensor_check_started_at": check_started,
                             "sensor_checked_at": sensor_checked, "checked_at": now,
                             "previous_valid_until": last_valid_until,
                             "limits": dict(limits, latest_sensors=final_sensor_until
                                            if checks._finite(final_sensor_until) else None),
                             "submitted": False}
                    if self.PREPARED_COMMANDS:
                        check["prepared_proof"] = current.get('_command_proof_info')
                        check["prepared_issue"] = prepared_issue
                    renewal_checks.append(check)
                    if last_valid_until is not None and now >= last_valid_until:
                        raise checks._StopError("intent_expired")
                    missing = [name for name in ("ros_monitor", "serial_monitor")
                               if not checks._finite(limits[name])]
                    if missing:
                        self._last_diagnostic = {"step": "intent_deadline", "missing": missing}
                        raise checks._StopError("expiry_unavailable")
                    if not checks._finite(final_sensor_until):
                        issue = getattr(self.source, "current_issue", None)
                        detail = prepared_issue if self.PREPARED_COMMANDS else issue() if callable(issue) else None
                        self._last_diagnostic = {"step": "motion_final_latest", "detail": detail}
                        if first_sent_at is None and 'initial_preparation' in result:
                            label = detail if isinstance(detail, str) and 0 < len(detail) <= 128 else 'unavailable'
                            reasons = result['initial_preparation']['sensor_rejections']
                            reasons[label] = reasons.get(label, 0) + 1
                        if self.PREPARED_COMMANDS and not initial_split and (detail in PENDING_PROOF
                                or first_sent_at is None and detail in ('new_scan', 'new_depth_unchecked')):
                            # Withhold a new command while waiting for a
                            # complete proof. The previous deadline stands.
                            if refresh_prepared:
                                self.source.wait_for_prepared_evidence(.005)
                            else:
                                sampler.wait_next(current, .005)
                            continue
                        if first_sent_at is not None or self.PREPARED_COMMANDS and not initial_split:
                            raise checks._StopError("sensor_unavailable")
                        limiting, valid_until = "latest_sensors", now
                    else:
                        limits["latest_sensors"] = final_sensor_until
                        limiting = min(limits, key=limits.get)
                        valid_until = limits[limiting]
                    # Starting with only one short frame leaves no time for
                    # the next observation. While stopped, wait for a longer
                    # *already valid* window; never extend an expiry.
                    min_window = .08 if (guarded or self.PREPARED_COMMANDS) and first_sent_at is None else .03
                    if first_sent_at is None and 'initial_preparation' in result:
                        window = valid_until - now
                        largest = result['initial_preparation']['largest_window_sec']
                        if checks._finite(window) and (largest is None or window > largest):
                            result['initial_preparation']['largest_window_sec'] = window
                    if (refresh_prepared and last_valid_until is not None
                            and now + .005 < last_valid_until
                            and valid_until <= last_valid_until
                            and limiting in ("latest_sensors", "depth_gate", "snapshot",
                                             "scan", "depth", "color")):
                        # A renewal with an earlier deadline would shorten
                        # the live intent. Hold its original expiry while
                        # waiting for genuinely newer checked evidence.
                        timing = result["motion_timing"]
                        timing["nonadvancing_window_polls"] = timing.get("nonadvancing_window_polls", 0) + 1
                        self.source.wait_for_prepared_evidence(.005)
                        continue
                    if valid_until - now <= min_window:
                        if first_sent_at is None and 'initial_preparation' in result:
                            result['initial_preparation']['short_window'] += 1
                        detail = (self._last_diagnostic.get('detail') if limiting == 'latest_sensors'
                                  and isinstance(self._last_diagnostic, dict) else None)
                        self._last_diagnostic = {"step": "intent_deadline", "limiting": limiting,
                                                 "remaining_sec": valid_until - now}
                        if detail is not None:
                            self._last_diagnostic['detail'] = detail
                        if (refresh_prepared and last_valid_until is not None
                                and now + .005 < last_valid_until
                                and limiting in ("latest_sensors", "depth_gate", "snapshot",
                                                 "scan", "depth", "color")):
                            # A cached read is cheap. Keep polling for an
                            # independent update through the already issued
                            # deadline; never replace it with a short intent.
                            timing = result["motion_timing"]
                            timing["pending_window_polls"] = timing.get("pending_window_polls", 0) + 1
                            self.source.wait_for_prepared_evidence(.005)
                            continue
                        # A still-valid but too-short proposal sends nothing.
                        # Try one fresh observation while the last intent
                        # can still run under its independent relay deadline.
                        if (last_valid_until is not None and not short_window_rechecked
                                and now < valid_until and now + .005 < last_valid_until
                                and limiting in ("latest_sensors", "depth_gate", "snapshot",
                                                 "scan", "depth", "color")):
                            short_window_rechecked = True
                            timing = result["motion_timing"]
                            timing["short_window_rechecks"] = timing.get("short_window_rechecks", 0) + 1
                            continue
                        if (first_sent_at is None and now < pre_motion_retry_until
                                and limiting not in ("relay", "trial", "model_color", "model_scan")):
                            if sampler is not None and sampler_started:
                                if refresh_prepared:
                                    sampler.wait_next(current, .005)
                                continue
                            self.clock.sleep(0.01)
                            current, now = self._aligned_stationary_sample(sample)
                            if self.relay.send_zero_heartbeat() is not True:
                                raise checks._StopError("relay_failure")
                            ready_at = self._now()
                            deadline = ready_at + self.duration_sec
                            anchor = previous = current["pose"]
                            previous_odom = current["odom_at"]
                            path, sequence = 0.0, self.relay.sequence + 1
                            continue
                        if limiting == "relay":
                            raise checks._StopError("motion_limit")
                        if limiting == "trial":
                            raise checks._StopError("time_limit")
                        raise checks._StopError("expiry_unavailable")
                    result["motion_timing"]["last_intent_valid_sec"] = valid_until - now
                    result["motion_timing"]["last_intent_limit"] = limiting
                    if limiting == "latest_sensors":
                        sensor_kind = getattr(self.source, "current_limit_kind", None)
                        if callable(sensor_kind):
                            result["motion_timing"]["last_sensor_limit"] = sensor_kind()
                    if sampler is not None:
                        pending = sampler.poll()
                        if pending['state'] == 'fault':
                            self._last_diagnostic = {'step': 'background_sample', 'detail': pending['reason']}
                            raise checks._StopError(pending['reason'])
                    if first_sent_at is None:
                        first_sent_at = now
                    result["motion_requested"] = True
                    if self.relay.send_frame(sequence, now, self.speed_mps, 0.0,
                                             valid_until=valid_until) is not True:
                        raise checks._StopError("relay_failure")
                    check.update({"submitted": True, "sequence": sequence, "valid_until": valid_until})
                    check["submitted_returned_at"] = self._now()
                    if refresh_prepared and not initial_split:
                        self.source.commit_split_validation(current)
                    self._motion_started = True
                    if guarded:
                        wait_for_write = getattr(self.relay, 'wait_for_frame_write', None)
                        if callable(wait_for_write) and wait_for_write(sequence) is not True:
                            raise checks._StopError('relay_failure')
                finally:
                    if guarded:
                        end_guard()
                if sampler is not None and not sampler_started:
                    # Let the pipe writer finish the first short-lived intent
                    # before starting CPU-heavy perception in another thread.
                    # The child still enforces the unmodified absolute expiry.
                    wait_for_write = getattr(self.relay, 'wait_for_frame_write', None)
                    if not guarded and callable(wait_for_write) and wait_for_write(sequence) is not True:
                        raise checks._StopError('relay_failure')
                    check["first_write_confirmed_at"] = self._now() if callable(wait_for_write) else None
                    if initial_split:
                        current['_front_coverage_valid_until'] = self._last_depth_assessment['front_coverage_valid_until']
                        initial_evidence['valid_until'] = min(initial_evidence['valid_until'], valid_until)
                        self.source.seed_split_validation(current, initial_evidence)
                        check["seed_installed_at"] = self._now()
                        sampler.start(current, self._last_depth_assessment)
                    else:
                        sampler.start()
                    check["sampler_started_at"] = self._now()
                    sampler_started = True
                sequence += 1
                last_sent_at = now
                last_valid_until = valid_until
                short_window_rechecked = False
                now = self._now()
                if now >= deadline:
                    raise checks._StopError("time_limit")
        except checks._StopError as exc:
            result["reason"] = str(exc)
        except BaseException as exc:
            result["reason"] = "exception"
            result["error_class"] = type(exc).__name__
        finally:
            if "motion_timing" in result:
                result["renewal_checks"] = list(renewal_checks)
            if sampler is not None:
                sampler.close(0)
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
            if sampler is not None:
                result['sampler_closed'] = sampler.close(0.2)
            if gc_suspended:
                gc.enable()
            if safety_mode_enabled:
                try:
                    restore = getattr(self.source, "restore_model_snapshots", None)
                    if callable(restore):
                        restore()
                except Exception:
                    result["source_restore_failed"] = True
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


class LongerForwardTrial(LiveTrial):
    """A separately opted-in, bounded forward trial; not full exploration."""
    MAX_DURATION_SEC = 2.0
    MOTION_LIMIT_SEC = 2.0
    MAX_DISTANCE_M = 0.04
    ASYNC_SAMPLING = True
    ARM_BEFORE_MODEL = True


class SynchronousLongerForwardTrial(LongerForwardTrial):
    """Experimental single-threaded sensing with the same motion bounds."""
    ASYNC_SAMPLING = False
    IMMEDIATE_SYNC_SAMPLING = True
    # Commanded-speed distance over one TTL; excludes odometry lag, stopping
    # latency, speed error and braking, and is not a physical travel bound.
    DISTANCE_RESERVE_SEC = 0.16


class PreparedLongerForwardTrial(LongerForwardTrial):
    """Overlap perception with renewal while retaining the relay's deadlines."""
    PREPARED_COMMANDS = True
    DISTANCE_RESERVE_SEC = 0.16


if __name__ == "__main__":
    sys.stderr.write("Review-only adapter; no robot connection or motion was started.\n")
    sys.exit(2)
