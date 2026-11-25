#!/usr/bin/env python2
"""Read-only ROS caller and serial-owner monitors for a reviewed trial.

No ROS connection, process inspection, serial open, or subprocess starts on
import or construction. These monitors sample asynchronously; verify() only
reads a bounded-age cache. ROS graph queries are not atomic. fuser may miss
inaccessible processes and TIOCEXCL cannot revoke already open descriptors.
An open probe that unexpectedly succeeds can have tty driver side effects; it
is immediately closed without reading, writing, or changing terminal settings.
"""
from __future__ import division

import copy
import errno
import hashlib
import math
import os
import socket
import stat
import subprocess
import tempfile
import threading
import time

from relay import _monotonic

try:
    import httplib as httpclient
    import xmlrpclib as xmlrpc
except ImportError:
    import http.client as httpclient
    import xmlrpc.client as xmlrpc

try:
    INTS = (int, long)  # noqa: F821
    STRINGS = (str, unicode)  # noqa: F821
except NameError:
    INTS = (int,)
    STRINGS = (str,)


def _number(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, INTS + (float,))
                and not math.isnan(value) and not math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _valid_pid(value):
    return not isinstance(value, bool) and isinstance(value, INTS) and 1 < value <= 4194304


class _AsyncMonitor(object):
    monitoring_enabled = True

    def __init__(self, backend, clock, poll_interval, max_age, acquire_timeout):
        if (not all(_number(v) for v in (poll_interval, max_age, acquire_timeout))
                or not 0.005 <= poll_interval <= 0.2 or not 0 < max_age <= 0.6
                or not 0 < acquire_timeout <= 2.0):
            raise ValueError("invalid monitoring limits")
        self.backend, self.clock = backend, clock
        self.poll_interval, self.max_age, self.acquire_timeout = poll_interval, max_age, acquire_timeout
        self._lock = threading.RLock()
        self._updated = threading.Event()
        self._wake = threading.Event()
        self._stop = threading.Event()
        if callable(getattr(self.backend, "set_cancel_event", None)):
            self.backend.set_cancel_event(self._stop)
        self._thread = None
        self._sample = None
        self._sample_at = None
        self._sample_mono_at = None
        self._command_evidence = None
        self._last_clock = None
        self._clock_changed_mono_at = None
        self._faulted = False
        self._acquired = False
        self._confirmed = False
        self._fault_reason = None
        self._fault_details = None
        self._read_reason = "sample_unavailable"
        self._sampling_started_mono = None
        self._terminal_reason = None
        self._terminal_details = None

    def _diagnose_assessment(self, sample, now):
        return "sample_rejected"

    def _diagnostic_details(self, sample):
        return {}

    def _late_sample(self, value, now):
        self._latch_fault("sample_time_invalid_or_expired")

    def _latch_fault(self, reason):
        if not self._faulted:
            self._faulted = True
            self._command_evidence = None
            self._fault_reason = reason
            try:
                self._fault_details = self._diagnostic_details(self._sample)
            except BaseException:
                self._fault_details = {}

    def _remember_terminal_reason(self):
        if not self._faulted:
            self._terminal_reason = self._read_reason
            try:
                if (self._sampling_started_mono is not None
                        and _monotonic() - self._sampling_started_mono > self.max_age):
                    self._terminal_reason = "background_sample_timeout"
            except BaseException:
                pass
            try:
                self._terminal_details = self._diagnostic_details(self._sample)
            except BaseException:
                self._terminal_details = {}

    def _loop(self):
        stage = "start"
        try:
            self.backend.start()
            while not self._stop.is_set():
                stage = "clock"
                started = self.clock.now()
                sample_started_mono = _monotonic()
                with self._lock:
                    self._sampling_started_mono = sample_started_mono
                stage = "sample"
                value = self.backend.sample()
                stage = "clock"
                now = self.clock.now()
                mono_now = _monotonic()
                with self._lock:
                    self._sampling_started_mono = None
                    if self._stop.is_set():
                        break
                    if not _number(started) or not _number(now) or now < started:
                        self._latch_fault("sample_time_invalid_or_expired")
                    else:
                        if self._last_clock is not None and now < self._last_clock:
                            self._latch_fault("clock_regressed")
                        if self._last_clock is None or now > self._last_clock:
                            self._clock_changed_mono_at = mono_now
                        elif mono_now - self._clock_changed_mono_at > self.max_age:
                            self._latch_fault("clock_frozen")
                        self._last_clock = now
                        if now - started > self.max_age:
                            self._late_sample(value, now)
                        else:
                            self._sample = value
                            self._sample_at = now
                            self._sample_mono_at = mono_now
                            # Once confirmed, any observed violation is permanent.
                            if self._confirmed:
                                if self._assess(value, now) is None:
                                    self._latch_fault(self._diagnose_assessment(value, now))
                                else:
                                    self._read_reason = "ok"
                                    self._publish_command_deadline(now, mono_now)
                    self._updated.set()
                remaining = self.poll_interval - (_monotonic() - sample_started_mono)
                self._wake.wait(max(.005, remaining))
                self._wake.clear()
        except BaseException:
            with self._lock:
                self._sampling_started_mono = None
                reason = "backend_start_exception" if stage == "start" else "clock_exception"
                if stage == "sample":
                    try:
                        reason = getattr(self.backend, "_diagnostic_rpc_failure", None)
                    except BaseException:
                        reason = None
                    if reason not in ("rpc_exception", "rpc_timeout"):
                        reason = "backend_sample_exception"
                self._latch_fault(reason)
                self._updated.set()
        finally:
            try:
                self.backend.stop()
            except BaseException:
                pass

    def _read(self):
        with self._lock:
            now = self.clock.now()
            if self._stop.is_set():
                self._read_reason = "stopped"
            elif self._faulted:
                self._read_reason = self._fault_reason or "fault_latched"
            elif not _number(now):
                self._read_reason = "clock_invalid"
            elif self._sample_at is None or self._sample_mono_at is None:
                if self._read_reason != "sample_deadline_missed":
                    self._read_reason = "sample_unavailable"
            elif (not 0 <= now - self._sample_at <= self.max_age
                  or not 0 <= _monotonic() - self._sample_mono_at <= self.max_age):
                if (self._sampling_started_mono is not None
                        and _monotonic() - self._sampling_started_mono > self.max_age):
                    self._read_reason = "background_sample_timeout"
                else:
                    self._read_reason = "cache_expired"
            else:
                result = self._assess(self._sample, now)
                self._read_reason = ("ok" if result is not None
                                     else self._diagnose_assessment(self._sample, now))
                return result
            if self._read_reason != "ok":
                return None

    def diagnostic(self):
        """Return sanitized cached state only; never query the backend or re-assess.

        reason is an observational code, not a safety or motion authorization.
        Once fault_latched is true, reason and details describe its first cause.
        Ages use the injected clock and monotonic clock; caller names, URIs, raw
        messages and exception text are intentionally omitted.
        """
        with self._lock:
            try:
                try:
                    now = self.clock.now()
                except BaseException:
                    now = None
                try:
                    mono_now = _monotonic()
                except BaseException:
                    mono_now = None
                age = (now - self._sample_at if _number(now) and self._sample_at is not None
                       else None)
                mono_age = (mono_now - self._sample_mono_at if _number(mono_now)
                            and self._sample_mono_at is not None else None)
                sampling_age = (mono_now - self._sampling_started_mono
                                if _number(mono_now) and self._sampling_started_mono is not None else None)
                age = age if _number(age) else None
                mono_age = mono_age if _number(mono_age) else None
                sampling_age = sampling_age if _number(sampling_age) else None
                if self._faulted:
                    reason = self._fault_reason or "fault_latched"
                elif self._thread is None:
                    reason = "not_started"
                elif self._stop.is_set():
                    reason = self._terminal_reason or "stopped"
                elif not _number(now) or not _number(mono_now):
                    reason = "clock_invalid"
                elif sampling_age is not None and sampling_age > self.max_age:
                    reason = "background_sample_timeout"
                elif age is None or mono_age is None:
                    reason = self._read_reason
                elif not 0 <= age <= self.max_age or not 0 <= mono_age <= self.max_age:
                    reason = "cache_expired"
                else:
                    reason = self._read_reason
                if self._faulted:
                    details = self._fault_details
                elif self._stop.is_set() and self._terminal_reason is not None:
                    details = self._terminal_details
                else:
                    details = self._diagnostic_details(self._sample)
                details = details or {}
                report = dict(details)
                report.update({"reason": reason, "fault_latched": self._faulted,
                               "sampling": sampling_age is not None,
                               "sampling_age_sec": sampling_age,
                               "cache_age_sec": age, "cache_mono_age_sec": mono_age,
                               "acquired": self._acquired, "confirmed": self._confirmed})
                return report
            except BaseException:
                return {"reason": "diagnostic_unavailable", "fault_latched": self._faulted}

    def _wait_valid(self, timeout):
        deadline = _monotonic() + timeout
        while _monotonic() < deadline:
            with self._lock:
                self._updated.clear()
                if self._read() is not None:
                    return True
                if self._faulted or self._stop.is_set():
                    return False
            self._updated.wait(min(0.02, max(0.0, deadline - _monotonic())))
        return False

    def acquire(self):
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                return False
            self._thread = threading.Thread(target=self._loop)
            self._thread.daemon = True
            self._thread.start()
        if not self._wait_valid(self.acquire_timeout):
            with self._lock:
                self._remember_terminal_reason()
            self.release()
            return False
        with self._lock:
            self._acquired = True
        return True

    def verify(self):
        with self._lock:
            if not self._acquired or not self._confirmed:
                return False
            result = self._read()
            if result is None:
                self._latch_fault(self._read_reason)
                return False
            return True

    def evidence(self):
        with self._lock:
            if not self._acquired or not self._confirmed:
                return {}
            result = self._read()
            if result is None:
                self._latch_fault(self._read_reason)
                return {}
            return copy.deepcopy(result)

    def _additional_deadlines(self, sample, now, mono_now):
        return []

    def _deadline_for_sample(self, now, mono_now):
        deadlines = [self._sample_at + self.max_age,
                     now + self.max_age - (mono_now - self._sample_mono_at)]
        deadlines.extend(self._additional_deadlines(self._sample, now, mono_now))
        if not all(_number(item) for item in deadlines):
            return None
        deadline = min(deadlines)
        return deadline if now < deadline else None

    def _publish_command_deadline(self, now, mono_now):
        if self._faulted or self._stop.is_set():
            return
        deadline = self._deadline_for_sample(now, mono_now)
        self._command_evidence = (self._sample, now, mono_now, deadline) if deadline is not None else None

    def _command_sample_current(self, sample):
        return True

    def cached_command_deadline(self):
        """Read already checked monitor data without its sampler lock."""
        evidence = self._command_evidence
        if evidence is None or self._faulted or self._stop.is_set():
            return None
        sample, checked, checked_mono, deadline = evidence
        now, mono_now = self.clock.now(), _monotonic()
        if (not _number(now) or now < checked or mono_now < checked_mono
                or not self._command_sample_current(sample)):
            return None
        deadline = min(deadline, now + (deadline - checked) - (mono_now - checked_mono))
        if self._faulted or self._stop.is_set() or not now < deadline:
            return None
        return deadline

    def valid_until(self):
        """Return the earliest cache/evidence expiry in the injected clock domain."""
        with self._lock:
            if not self._acquired or not self._confirmed or self._read() is None:
                return None
            try:
                now, mono_now = self.clock.now(), _monotonic()
                return self._deadline_for_sample(now, mono_now)
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                return None

    def release(self):
        self._stop.set()
        self._wake.set()
        self._updated.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(0.05)


def _parse_watchdog(packet, now, driver_pid):
    try:
        if not isinstance(packet, dict) or not _number(packet.get("received_at")):
            return None
        data = packet["data"]
        if not isinstance(data, STRINGS) or len(data) > 256:
            return None
        fields = data.split()
        if (len(fields) != 9 or fields[0] != "v2"
                or not fields[1].endswith("_" + str(driver_pid))
                or not all(x.isdigit() for x in fields[2:6])):
            return None
        count, mode, primed, latched = [int(x) for x in fields[2:6]]
        zero_at, generated = float(fields[7]), float(fields[8])
        if (count > 18446744073709551615 or mode != 0 or primed not in (0, 1) or latched != 0
                or not _number(zero_at) or not _number(generated)
                or not 0 <= zero_at <= generated
                or not 0 <= now - generated < 0.25
                or not 0 <= now - packet["received_at"] < 0.25):
            return None
        return {"boot_id": fields[1], "count": count, "primed": primed, "source": fields[6],
                "zero_at": zero_at, "generated_at": generated, "publisher": packet.get("publisher")}
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None


class RosCallerMonitor(_AsyncMonitor):
    """Bind one /cmd_vel node to the relay process and driver's v2 status."""
    def __init__(self, backend, clock, driver_pid, poll_interval=0.05, max_age=0.25,
                 acquire_timeout=0.5, bind_timeout=0.12, retry_late_samples=False):
        if (not _valid_pid(driver_pid) or not _number(bind_timeout) or not 0 < bind_timeout <= 0.4
                or not _number(max_age) or not 0 < max_age <= 0.3 or type(retry_late_samples) is not bool):
            raise ValueError("invalid ROS monitor settings")
        _AsyncMonitor.__init__(self, backend, clock, poll_interval, max_age, acquire_timeout)
        self.driver_pid, self.bind_timeout = driver_pid, bind_timeout
        self._relay = None
        self._relay_pid = None
        self._last_status = None
        self._retry_late_samples = retry_late_samples
        self._late_samples_discarded = 0

    def _late_sample(self, value, now):
        if not self._retry_late_samples or self._faulted:
            _AsyncMonitor._late_sample(self, value, now)
            return
        self._command_evidence = None
        if self._confirmed and self._assess(value, now) is None:
            self._sample = value
            self._latch_fault(self._diagnose_assessment(value, now))
            return
        # A fresh graph is required for initial binding as well as renewed authority.
        self._sample = self._sample_at = self._sample_mono_at = None
        self._read_reason = "sample_deadline_missed"
        self._late_samples_discarded += 1
        self.backend.invalidate_graph()

    def _graph_fresh(self, sample, now):
        get_generation = getattr(self.backend, "graph_generation", None)
        if not callable(get_generation):
            return True  # Existing injectable backends retain their contract.
        try:
            observed = sample["graph_observed_at"]
            observed_mono = sample["graph_observed_mono_at"]
            generation = sample["graph_generation"]
            return (not isinstance(generation, bool) and isinstance(generation, INTS)
                    and generation == get_generation()
                    and _number(now) and _number(observed) and _number(observed_mono)
                    and 0 <= now - observed <= 1.2
                    and 0 <= _monotonic() - observed_mono <= 1.2)
        except BaseException:
            return False

    def _additional_deadlines(self, sample, now, mono_now):
        packet = sample['watchdog']
        status = _parse_watchdog(packet, now, self.driver_pid)
        if status is None:
            return [None]
        limits = [status['generated_at'] + 0.25, packet['received_at'] + 0.25]
        if callable(getattr(self.backend, 'graph_generation', None)):
            limits.extend([sample['graph_observed_at'] + 1.2,
                           now + 1.2 - (mono_now - sample['graph_observed_mono_at'])])
        return limits

    def _diagnostic_details(self, sample):
        # Never include raw graph entries, caller names, status text or URIs.
        details = {"driver_pid_expected": self.driver_pid, "relay_pid_expected": self._relay_pid,
                   "late_samples_discarded": self._late_samples_discarded,
                   "driver_pid_observed": None, "relay_pid_observed": None,
                   "cmd_publisher_count": None, "watchdog_publisher_count": None,
                   "watchdog_source": "unavailable"}
        try:
            pubs, drivers, pids = (sample.get("cmd_vel_publishers"),
                                   sample.get("watchdog_publishers"), sample.get("node_pids"))
            if isinstance(pubs, list):
                details["cmd_publisher_count"] = len(pubs)
            if isinstance(drivers, list):
                details["watchdog_publisher_count"] = len(drivers)
            if isinstance(pids, dict):
                if isinstance(drivers, list) and len(drivers) == 1:
                    pid = pids.get(drivers[0])
                    if _valid_pid(pid):
                        details["driver_pid_observed"] = pid
                if isinstance(pubs, list) and len(pubs) == 1:
                    pid = pids.get(pubs[0])
                    if _valid_pid(pid):
                        details["relay_pid_observed"] = pid
            packet = sample.get("watchdog")
            data = packet.get("data") if isinstance(packet, dict) else None
            if isinstance(data, STRINGS) and len(data) <= 256:
                fields = data.split()
                if len(fields) == 9:
                    if fields[6] == "none":
                        details["watchdog_source"] = "none"
                    elif isinstance(pubs, list) and len(pubs) == 1:
                        details["watchdog_source"] = ("matches_cmd_publisher" if fields[6] == pubs[0]
                                                      else "other")
                    else:
                        details["watchdog_source"] = "unmatched_or_multiple_publishers"
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            pass
        return details

    def _diagnose_assessment(self, sample, now):
        """Explain a failed assessment without mutating status or querying ROS."""
        try:
            if not self._graph_fresh(sample, now):
                return "graph_expired_or_invalidated"
            pubs = sample["cmd_vel_publishers"]
            drivers = sample["watchdog_publishers"]
            pids = sample["node_pids"]
            if (not isinstance(pubs, list) or len(set(pubs)) != len(pubs)
                    or not all(isinstance(v, STRINGS) and v.startswith("/") for v in pubs)):
                return "cmd_publishers_malformed"
            if not isinstance(drivers, list) or len(drivers) != 1:
                return "watchdog_publisher_count"
            if not isinstance(pids, dict):
                return "graph_malformed"
            if pids.get(drivers[0]) != self.driver_pid:
                return "driver_pid_mismatch"
            packet = sample["watchdog"]
            status = _parse_watchdog(packet, now, self.driver_pid)
            if status is None:
                if isinstance(packet, dict):
                    data = packet.get("data")
                    fields = data.split() if isinstance(data, STRINGS) and len(data) <= 256 else []
                    if len(fields) == 9 and fields[5].isdigit() and int(fields[5]) != 0:
                        return "watchdog_latched"
                    received = packet.get("received_at")
                    if _number(received) and not 0 <= now - received < 0.25:
                        return "watchdog_expired"
                    if len(fields) == 9:
                        try:
                            generated = float(fields[8])
                            if _number(generated) and not 0 <= now - generated < 0.25:
                                return "watchdog_expired"
                        except (TypeError, ValueError, OverflowError):
                            pass
                return "watchdog_missing_or_malformed"
            if status["publisher"] != drivers[0]:
                return "watchdog_publisher_mismatch"
            last = self._last_status
            if last is not None and (status["boot_id"] != last["boot_id"]
                                     or status["count"] < last["count"]
                                     or status["generated_at"] < last["generated_at"]
                                     or status["zero_at"] < last["zero_at"]):
                return "watchdog_regressed"
            if self._relay_pid is None:
                return "pre_arm_publisher_present" if pubs else "sample_rejected"
            if len(pubs) != 1:
                return "cmd_publisher_count"
            if pids.get(pubs[0]) != self._relay_pid:
                return "relay_pid_mismatch"
            if status["source"] != pubs[0]:
                return "watchdog_source_mismatch"
            if status["primed"] != 1:
                return "watchdog_unprimed"
            if status["count"] == 0:
                return "watchdog_no_commands"
            return "relay_not_running"
        except BaseException:
            return "graph_or_watchdog_malformed"

    def _assess(self, sample, now):
        try:
            if not self._graph_fresh(sample, now):
                return None
            pubs = sample["cmd_vel_publishers"]
            drivers = sample["watchdog_publishers"]
            pids = sample["node_pids"]
            if (not isinstance(pubs, list) or len(set(pubs)) != len(pubs)
                    or not all(isinstance(v, STRINGS) and v.startswith("/") for v in pubs)
                    or not isinstance(drivers, list) or len(drivers) != 1
                    or not isinstance(pids, dict) or pids.get(drivers[0]) != self.driver_pid):
                return None
            status = _parse_watchdog(sample["watchdog"], now, self.driver_pid)
            if status is None or status["publisher"] != drivers[0]:
                return None
            last = self._last_status
            if last is not None and (status["boot_id"] != last["boot_id"]
                                     or status["count"] < last["count"]
                                     or status["generated_at"] < last["generated_at"]
                                     or status["zero_at"] < last["zero_at"]):
                return None
            self._last_status = status
            if self._relay_pid is None:
                return {"pre_arm": True} if pubs == [] else None
            if (len(pubs) != 1 or pids.get(pubs[0]) != self._relay_pid
                    or status["source"] != pubs[0] or status["primed"] != 1
                    or status["count"] == 0 or not self._relay_alive()):
                return None
            result = {"driver_caller_bound": True, "observed_competitors": [], "graph_query_atomic": False,
                      "relay_node": pubs[0], "relay_pid": self._relay_pid, "driver_node": drivers[0],
                      "driver_pid": self.driver_pid, "watchdog_source": status["source"],
                      "watchdog_boot_id": status["boot_id"], "watchdog_generated_at": status["generated_at"],
                      "checked_at": self._sample_at}
            if callable(getattr(self.backend, "graph_generation", None)):
                result.update({"graph_observed_at": sample["graph_observed_at"],
                               "graph_age_sec": now - sample["graph_observed_at"],
                               "graph_max_age_sec": 1.2})
            return result
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return None

    def _relay_alive(self):
        try:
            process = self._relay.transport.process
            return process.pid == self._relay_pid and process.poll() is None
        except (AttributeError, TypeError, ValueError):
            return False

    def _command_sample_current(self, sample):
        generation = getattr(self.backend, 'graph_generation', None)
        return (self._relay_alive() and
                (not callable(generation) or sample.get('graph_generation') == generation()))

    def bind_relay(self, relay):
        try:
            process = relay.transport.process
            if not _valid_pid(process.pid) or process.poll() is not None:
                return False
            with self._lock:
                if not self._acquired or self._confirmed or self._faulted:
                    return False
                invalidate = getattr(self.backend, "invalidate_graph", None)
                if callable(invalidate):
                    invalidate()
                self._relay, self._relay_pid = relay, process.pid
            self._wake.set()
            if not self._wait_valid(self.bind_timeout):
                with self._lock:
                    self._remember_terminal_reason()
                return False
            with self._lock:
                self._terminal_reason = None
                self._terminal_details = None
                self._confirmed = True
                if self._read() is not None:
                    self._publish_command_deadline(self.clock.now(), _monotonic())
            return True
        except (AttributeError, TypeError, ValueError):
            return False


class SerialOwnershipMonitor(_AsyncMonitor):
    """Check process identity, visible holders, and the read-only EBUSY probe."""
    def __init__(self, backend, clock, pid, expected_exe, expected_sha256,
                 poll_interval=0.05, max_age=0.25, acquire_timeout=0.5, ros_monitor=None,
                 retry_late_samples=False):
        if (not _valid_pid(pid) or not isinstance(expected_exe, STRINGS) or not expected_exe.startswith("/")
                or not isinstance(expected_sha256, STRINGS) or len(expected_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in expected_sha256)):
            raise ValueError("invalid driver identity")
        if ros_monitor is not None and getattr(ros_monitor, "driver_pid", None) != pid:
            raise ValueError("ROS and serial monitors must identify the same driver")
        if (type(retry_late_samples) is not bool
                or retry_late_samples and not callable(getattr(backend, "invalidate_holders", None))):
            raise ValueError("serial retry requires fresh holder queries")
        _AsyncMonitor.__init__(self, backend, clock, poll_interval, max_age, acquire_timeout)
        self.pid, self.expected_exe, self.expected_sha256 = pid, expected_exe, expected_sha256
        self.ros_monitor = ros_monitor
        self._start_ticks = None
        self._probe_at = None
        self._retry_late_samples = retry_late_samples
        self._late_samples_discarded = 0

    def _late_sample(self, value, now):
        if not self._retry_late_samples or not self._confirmed or self._faulted:
            _AsyncMonitor._late_sample(self, value, now)
            return
        self._command_evidence = None
        if self._assess(value, now) is None:
            self._sample = value
            self._latch_fault(self._diagnose_assessment(value, now))
            return
        # An over-budget check cannot renew a command, even with the expected owner.
        self._sample = self._sample_at = self._sample_mono_at = None
        self._read_reason = "sample_deadline_missed"
        self._late_samples_discarded += 1
        self.backend.invalidate_holders()

    def _diagnostic_details(self, sample):
        if not isinstance(sample, dict):
            return {'late_samples_discarded': self._late_samples_discarded}
        holders = sample.get('fuser_pids')
        return {'visible_holder_count': len(holders) if isinstance(holders, list) else None,
                'expected_holder_only': holders == [self.pid],
                'holder_query_complete': sample.get('fuser_complete') is True,
                'exclusive_open_probe': sample.get('open_errno') == errno.EBUSY,
                'late_samples_discarded': self._late_samples_discarded}

    def _diagnose_assessment(self, sample, now):
        if not isinstance(sample, dict):
            return 'serial_sample_missing'
        if sample.get('before') != sample.get('after'):
            return 'serial_driver_identity_changed'
        if sample.get('fuser_complete') is not True:
            return 'serial_holder_query_unavailable'
        if sample.get('fuser_pids') != [self.pid]:
            return 'serial_holder_mismatch'
        if not _number(sample.get('fuser_at')) or not 0 <= now - sample['fuser_at'] <= 1.2:
            return 'serial_holder_query_expired'
        return 'serial_identity_or_exclusivity_unverified'

    def _assess(self, sample, now):
        try:
            before, after = sample["before"], sample["after"]
            if (not isinstance(before, dict) or before != after
                    or before.get("pid") != self.pid or before.get("exe") != self.expected_exe
                    or before.get("sha256") != self.expected_sha256
                    or not _valid_pid(before.get("pid")) or not isinstance(before.get("start_ticks"), INTS)
                    or isinstance(before.get("start_ticks"), bool) or before["start_ticks"] <= 0
                    or self._start_ticks is not None and before["start_ticks"] != self._start_ticks
                    or sample.get("fuser_complete") is not True
                    or not _number(sample.get("fuser_at")) or not 0 <= now - sample["fuser_at"] <= 1.2
                    or sample.get("fuser_pids") != [self.pid] or sample.get("open_errno") != errno.EBUSY
                    or sample.get("probe_repeated") is not False or not _number(sample.get("probe_at"))
                    or not 0 <= sample["probe_at"] <= now
                    or self._probe_at is not None and self._probe_at != sample["probe_at"]):
                return None
            self._start_ticks = before["start_ticks"]
            self._probe_at = sample["probe_at"]
            return {"tioc_exclusive_reported": False, "initial_open_errno": "EBUSY", "probe_repeated": False,
                    "observed_other_fds": [], "previous_fd_excluded": False, "fuser_pids": [self.pid],
                    "driver_pid": self.pid, "driver_exe": self.expected_exe,
                    "driver_sha256": self.expected_sha256, "driver_start_ticks": before["start_ticks"],
                    "checked_at": self._sample_at, "initial_probe_at": sample.get("probe_at"),
                    "fuser_checked_at": sample["fuser_at"],
                    "fuser_visibility_complete": False}
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return None

    def _additional_deadlines(self, sample, now, mono_now):
        return [sample['fuser_at'] + 1.2]

    def valid_until(self):
        own = self.local_command_deadline()
        if own is None:
            return None
        if self.ros_monitor is None:
            return own
        linked = self.ros_monitor.valid_until()
        return min(own, linked) if _number(linked) else None

    def local_command_deadline(self):
        """Check serial evidence when the caller separately checks the ROS guard."""
        return _AsyncMonitor.valid_until(self)

    def acquire(self):
        if not _AsyncMonitor.acquire(self):
            return False
        with self._lock:
            self._confirmed = True
            if self._read() is not None:
                self._publish_command_deadline(self.clock.now(), _monotonic())
        return True

    def verify(self):
        if not _AsyncMonitor.verify(self):
            return False
        try:
            if self.ros_monitor is not None and self.ros_monitor.verify() is not True:
                with self._lock:
                    self._latch_fault("linked_ros_monitor_failed")
                return False
            return True
        except BaseException:
            with self._lock:
                self._latch_fault("linked_ros_monitor_exception")
            return False


def parse_fuser(code, stdout, stderr, device):
    """Accept only complete-looking fuser output; visibility may still be limited."""
    try:
        if (code != 0 or not isinstance(stdout, bytes) or not isinstance(stderr, bytes)
                or not isinstance(device, STRINGS)):
            return None
        out = stdout.decode("ascii").strip()
        err = stderr.decode("ascii").strip()
        if err not in ("", device + ":"):
            return None
        values = out.split()
        if not values or not all(item.isdigit() for item in values):
            return None
        pids = [int(item) for item in values]
        if any(not _valid_pid(pid) for pid in pids) or len(set(pids)) != len(pids):
            return None
        return sorted(pids)
    except (TypeError, ValueError, UnicodeError, OverflowError):
        return None


class _TimeoutTransport(xmlrpc.Transport):
    def __init__(self, timeout):
        xmlrpc.Transport.__init__(self)
        self.timeout = timeout

    def make_connection(self, host):
        return httpclient.HTTPConnection(host, timeout=self.timeout)


class Ros1ReadOnlyBackend(object):
    """ROS graph XML-RPC plus a watchdog subscription; no velocity publisher."""
    def __init__(self, master_uri, clock, rpc_timeout=0.05, proxy_factory=None,
                 subscriber_factory=None):
        if (not isinstance(master_uri, STRINGS) or not master_uri.startswith("http://")
                or not _number(rpc_timeout) or not 0 < rpc_timeout <= 0.2):
            raise ValueError("invalid ROS master or RPC timeout")
        self.master_uri, self.clock, self.rpc_timeout = master_uri, clock, rpc_timeout
        self.proxy_factory = proxy_factory
        self.subscriber_factory = subscriber_factory
        self.lock = threading.Lock()
        self.watchdog = None
        self.subscriber = None
        self.caller_id = "/standalone_trial_monitor"
        self._diagnostic_rpc_failure = None
        self._graph_lock = threading.Lock()
        self._graph_generation = 0
        self._graph_cache = None

    def graph_generation(self):
        with self._graph_lock:
            return self._graph_generation

    def invalidate_graph(self):
        """Require queries started after this call; do not wait for in-flight RPC."""
        with self._graph_lock:
            self._graph_generation += 1
            self._graph_cache = None
            return self._graph_generation

    def _proxy(self, uri):
        if not isinstance(uri, STRINGS) or not uri.startswith("http://"):
            raise ValueError("unexpected ROS node URI")
        if self.proxy_factory is not None:
            return self.proxy_factory(uri)
        return xmlrpc.ServerProxy(uri, transport=_TimeoutTransport(self.rpc_timeout))

    def _receive(self, message):
        try:
            record = {"data": message.data, "received_at": self.clock.now(),
                      "publisher": message._connection_header["callerid"]}
            with self.lock:
                self.watchdog = record
        except (AttributeError, KeyError, TypeError, ValueError):
            with self.lock:
                self.watchdog = None

    def start(self):
        if self.subscriber_factory is None:
            import rospy
            from std_msgs.msg import String
            self.subscriber = rospy.Subscriber("/rosclaw_limo/command_watchdog", String,
                                               self._receive, queue_size=1)
        else:
            self.subscriber = self.subscriber_factory(self._receive)

    @staticmethod
    def _value(response):
        if (not isinstance(response, (tuple, list)) or len(response) != 3
                or response[0] != 1):
            raise ValueError("ROS XML-RPC query failed")
        return response[2]

    def _rpc_value(self, method, *args):
        try:
            return self._value(method(*args))
        except socket.timeout:  # noqa: UP041 - Python 2
            self._diagnostic_rpc_failure = "rpc_timeout"
            raise
        except BaseException:
            self._diagnostic_rpc_failure = "rpc_exception"
            raise

    def _query_graph(self):
        master = self._proxy(self.master_uri)
        state = self._rpc_value(master.getSystemState, self.caller_id)
        if not isinstance(state, (tuple, list)) or len(state) != 3:
            raise ValueError("malformed ROS graph")
        publications = state[0]
        if not isinstance(publications, (tuple, list)):
            raise ValueError("malformed ROS publishers")
        by_topic = {}
        for entry in publications:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2 or entry[0] in by_topic:
                raise ValueError("malformed ROS publication")
            if not isinstance(entry[1], (tuple, list)):
                raise ValueError("malformed ROS caller list")
            by_topic[entry[0]] = list(entry[1])
        cmd = by_topic.get("/cmd_vel", [])
        status_nodes = by_topic.get("/rosclaw_limo/command_watchdog", [])
        node_pids = {}
        for node in set(cmd + status_nodes):
            if not isinstance(node, STRINGS) or not node.startswith("/"):
                raise ValueError("malformed ROS node")
            uri = self._rpc_value(master.lookupNode, self.caller_id, node)
            pid = self._rpc_value(self._proxy(uri).getPid, self.caller_id)
            if not _valid_pid(pid):
                raise ValueError("invalid ROS node PID")
            node_pids[node] = pid
        return {"cmd_vel_publishers": cmd, "watchdog_publishers": status_nodes,
                "node_pids": node_pids}

    def sample(self):
        """Reuse graph/PIDs for <0.75 s; always read the latest watchdog packet.

        Graph time is the beginning of the non-atomic RPC sequence. Invalidation
        discards in-flight older generations, and a slower concurrent query
        cannot overwrite a newer cache. Clock regressions and RPC errors fail
        closed; a failed refresh never silently reuses stale graph data.
        """
        self._diagnostic_rpc_failure = None
        graph = None
        for _ in range(3):
            now, mono = self.clock.now(), _monotonic()
            if not _number(now) or not _number(mono):
                raise ValueError("invalid graph clock")
            with self._graph_lock:
                generation = self._graph_generation
                cached = self._graph_cache
                if cached is not None:
                    age = now - cached["graph_observed_at"]
                    mono_age = mono - cached["graph_observed_mono_at"]
                    if age < 0 or mono_age < 0:
                        raise ValueError("graph clock regressed")
                    if age < 0.75 and mono_age < 0.75:
                        graph = copy.deepcopy(cached)
                        break
            graph = self._query_graph()
            after, after_mono = self.clock.now(), _monotonic()
            if (not _number(after) or not _number(after_mono)
                    or not 0 <= after - now <= 1.2 or not 0 <= after_mono - mono <= 1.2):
                raise ValueError("graph observation expired or clock changed")
            graph.update({"graph_observed_at": now, "graph_observed_mono_at": mono,
                          "graph_generation": generation})
            with self._graph_lock:
                if generation != self._graph_generation:
                    graph = None
                    continue
                cached = self._graph_cache
                if cached is None or cached["graph_observed_mono_at"] <= mono:
                    self._graph_cache = copy.deepcopy(graph)
                else:
                    # A newer concurrent query has already completed.
                    if (not 0 <= after - cached["graph_observed_at"] <= 1.2
                            or not 0 <= after_mono - cached["graph_observed_mono_at"] <= 1.2):
                        raise ValueError("concurrent graph observation expired")
                    graph = copy.deepcopy(cached)
                break
        if graph is None:
            raise ValueError("graph invalidated repeatedly")
        with self.lock:
            graph["watchdog"] = copy.deepcopy(self.watchdog)
        return graph

    def stop(self):
        if self.subscriber is not None:
            self.subscriber.unregister()
            self.subscriber = None


class ProcSerialBackend(object):
    """Inspect /proc, fuser and a read-only nonblocking open of a tty device."""
    def __init__(self, clock, pid, device, fuser_timeout=0.16):
        if (not _valid_pid(pid) or not isinstance(device, STRINGS) or not device.startswith("/dev/")
                or not _number(fuser_timeout) or not 0 < fuser_timeout <= 0.5):
            raise ValueError("invalid serial inspection settings")
        self.clock, self.pid, self.device, self.fuser_timeout = clock, pid, device, fuser_timeout
        self._probe_done = False
        self._probe_errno = None
        self._probe_at = None
        self._fuser_at = None
        self._fuser_pids = None
        self._cancel_event = None
        self._started = False

    def set_cancel_event(self, event):
        if self._started:
            raise ValueError("serial inspector already started")
        self._cancel_event = event

    def start(self):
        if self._started:
            raise ValueError("serial inspector is single-use")
        self._started = True

    def read_identity(self):
        prefix = "/proc/%d" % self.pid
        with open(prefix + "/stat", "rb") as stat_file:
            stat_line = stat_file.read(8192).decode("ascii")
        end = stat_line.rfind(")")
        if end < 1 or not stat_line.startswith(str(self.pid) + " ("):
            raise ValueError("invalid process stat")
        fields = stat_line[end + 2:].split()
        if len(fields) < 20 or not fields[19].isdigit():
            raise ValueError("missing process start time")
        start_ticks = int(fields[19])
        exe = os.readlink(prefix + "/exe")
        digest = hashlib.sha256()
        with open(prefix + "/exe", "rb") as binary:
            while True:
                chunk = binary.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
        return {"pid": self.pid, "start_ticks": start_ticks, "exe": exe, "sha256": digest.hexdigest()}

    def fuser(self):
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(["/bin/fuser", self.device], stdout=output, stderr=errors,
                                       close_fds=True)
            deadline = _monotonic() + self.fuser_timeout
            while process.poll() is None and _monotonic() < deadline:
                time.sleep(0.005)
            if process.poll() is None:
                process.kill()
                for _ in range(20):
                    if process.poll() is not None:
                        break
                    time.sleep(0.005)
                if process.poll() is None:
                    return None
            code = process.poll()
            output.seek(0)
            errors.seek(0)
            stdout, stderr = output.read(4097), errors.read(4097)
            if len(stdout) > 4096 or len(stderr) > 4096:
                return None
            return parse_fuser(code, stdout, stderr, self.device)

    def _query_holders(self):
        return self.fuser()

    def invalidate_holders(self):
        """Require a new descriptor scan on the next sampling iteration."""
        self._fuser_at = self._fuser_pids = None

    def probe(self):
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise ValueError("serial inspection cancelled")
        try:
            fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)
        except OSError as exc:
            return exc.errno
        else:
            os.close(fd)
            return 0

    def sample(self):
        before = self.read_identity()
        now = self.clock.now()
        if not _number(now) or (self._fuser_at is not None and now < self._fuser_at):
            raise ValueError("serial monitor clock invalid")
        if self._fuser_at is None or now - self._fuser_at >= 0.75:
            self._fuser_pids = self._query_holders()
            self._fuser_at = self.clock.now()
            if not _number(self._fuser_at) or self._fuser_at < now:
                raise ValueError("serial monitor clock invalid")
        holders = self._fuser_pids
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise ValueError("serial inspection cancelled")
        if not self._probe_done and holders == [self.pid]:
            # Minimize, but cannot eliminate, the exit-before-open race.
            if self.read_identity() != before:
                raise ValueError("driver identity changed before probe")
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise ValueError("serial inspection cancelled")
            self._probe_done = True
            self._probe_errno = self.probe()
            self._probe_at = self.clock.now()
        after = self.read_identity()
        return {"before": before, "after": after, "fuser_pids": holders,
                "fuser_complete": holders is not None, "open_errno": self._probe_errno,
                "fuser_at": self._fuser_at, "probe_at": self._probe_at, "probe_repeated": False}

    def stop(self):
        return None


class ProcFdSerialBackend(ProcSerialBackend):
    """Inspect visible tty descriptors directly, without starting fuser."""

    def _query_holders(self):
        device = os.stat(self.device)
        if not stat.S_ISCHR(device.st_mode):
            raise ValueError('serial device is not a character device')
        holders = []
        for name in os.listdir('/proc'):
            if not name.isdigit():
                continue
            pid = int(name)
            directory = '/proc/{}/fd'.format(pid)
            # Disappearing processes and descriptors are normal. Visibility
            # of other users' descriptors is limited to the caller's privileges.
            unavailable = (errno.ENOENT, errno.ESRCH)
            if pid != self.pid:
                unavailable += (errno.EACCES, errno.EPERM)
            try:
                descriptors = os.listdir(directory)
            except OSError as error:
                if error.errno in unavailable:
                    continue
                raise
            for descriptor in descriptors:
                try:
                    fd = os.stat(os.path.join(directory, descriptor))
                except OSError as error:
                    if error.errno in unavailable:
                        continue
                    raise
                if stat.S_ISCHR(fd.st_mode) and fd.st_rdev == device.st_rdev:
                    holders.append(pid)
                    break
        return sorted(holders)
