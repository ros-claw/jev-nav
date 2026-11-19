"""Offline tests with fake ROS, process and device observations only."""
import copy
import errno
import socket
import threading
import time

import pytest


def module():
    try:
        import live_monitors
        return live_monitors
    except ImportError:
        pytest.fail("live monitors are not implemented", pytrace=False)


class Clock(object):
    def __init__(self):
        self.t = 100.0

    def now(self):
        return self.t


class Backend(object):
    def __init__(self, data):
        self.data = data
        self.lock = threading.Lock()
        self.sampled = threading.Event()
        self.block = None
        self.stopped = False

    def start(self):
        return None

    def sample(self):
        if self.block:
            self.block.wait(1.0)
        with self.lock:
            value = copy.deepcopy(self.data)
        self.sampled.set()
        return value

    def set(self, data):
        with self.lock:
            self.data = data
        self.sampled.clear()
        assert self.sampled.wait(0.5)

    def stop(self):
        self.stopped = True


def ros_sample(clock, publishers=None, caller="none", primed=0, relay_pid=222, driver_pid=11139,
               watchdog_pub="/limo_base", latched=0, boot_pid=None):
    publishers = [] if publishers is None else publishers
    boot_pid = driver_pid if boot_pid is None else boot_pid
    return {"cmd_vel_publishers": publishers, "watchdog_publishers": [watchdog_pub],
            "node_pids": {"/relay": relay_pid, watchdog_pub: driver_pid},
            "watchdog": {"data": "v2 boot_%s 4 0 %s %s %s 99.900000 100.000000" %
                                  (boot_pid, primed, latched, caller),
                         "publisher": watchdog_pub, "received_at": clock.now()}}


class Relay(object):
    def __init__(self, pid):
        self.transport = Obj(process=Obj(pid=pid, poll=lambda: None))

    def healthy(self):
        return True


class Obj(object):
    def __init__(self, **values):
        self.__dict__.update(values)


def ros_guard(clock, backend):
    return module().RosCallerMonitor(backend, clock, driver_pid=11139,
                                     poll_interval=0.01, max_age=0.25,
                                     acquire_timeout=0.4, bind_timeout=0.4)


def test_ros_guard_binds_relay_pid_caller_and_watchdog_identity():
    clock = Clock()
    backend = Backend(ros_sample(clock))
    guard = ros_guard(clock, backend)
    try:
        assert guard.acquire()
        backend.set(ros_sample(clock, ["/relay"], caller="/relay", primed=1))
        assert guard.bind_relay(Relay(222))
        assert guard.verify()
        evidence = guard.evidence()
        assert evidence["driver_caller_bound"] is True
        assert evidence["observed_competitors"] == []
        assert evidence["graph_query_atomic"] is False
        assert evidence["relay_node"] == "/relay" and evidence["relay_pid"] == 222
        assert evidence["driver_pid"] == 11139 and evidence["watchdog_source"] == "/relay"
    finally:
        guard.release()
    assert backend.stopped


@pytest.mark.parametrize("problem", ["other_publisher", "relay_pid", "wrong_source", "driver_pid",
                                       "boot_pid", "latched", "unprimed", "status_stale"])
def test_ros_guard_rejects_wrong_graph_pid_or_watchdog(problem):
    clock = Clock()
    backend = Backend(ros_sample(clock))
    guard = ros_guard(clock, backend)
    try:
        assert guard.acquire()
        sample = ros_sample(clock, ["/relay"], caller="/relay", primed=1)
        if problem == "other_publisher":
            sample["cmd_vel_publishers"].append("/other")
        elif problem == "relay_pid":
            sample["node_pids"]["/relay"] = 333
        elif problem == "wrong_source":
            sample["watchdog"]["data"] = sample["watchdog"]["data"].replace(" /relay ", " /other ")
        elif problem == "driver_pid":
            sample["node_pids"]["/limo_base"] = 11140
        elif problem == "boot_pid":
            sample["watchdog"]["data"] = sample["watchdog"]["data"].replace("boot_11139", "boot_11140")
        elif problem == "latched":
            sample = ros_sample(clock, ["/relay"], caller="/relay", primed=1, latched=1)
        elif problem == "unprimed":
            sample = ros_sample(clock, ["/relay"], caller="/relay", primed=0)
        else:
            sample["watchdog"]["received_at"] = 99.0
        backend.set(sample)
        assert not guard.bind_relay(Relay(222))
    finally:
        guard.release()


def serial_sample():
    return {"before": {"pid": 11139, "start_ticks": 12345, "exe": "/tmp/candidate/limo_base_node",
                       "sha256": "a" * 64},
            "after": {"pid": 11139, "start_ticks": 12345, "exe": "/tmp/candidate/limo_base_node",
                      "sha256": "a" * 64},
            "fuser_pids": [11139], "fuser_complete": True, "open_errno": errno.EBUSY,
            "probe_at": 100.0, "probe_repeated": False, "fuser_at": 100.0}


def serial_guard(clock, backend):
    return module().SerialOwnershipMonitor(backend, clock, pid=11139,
        expected_exe="/tmp/candidate/limo_base_node", expected_sha256="a" * 64,
        poll_interval=0.01, max_age=0.25, acquire_timeout=0.4)


def test_serial_monitor_can_accommodate_slow_fuser_without_relaxing_ros_monitor():
    mod, clock = module(), Clock()
    serial = mod.SerialOwnershipMonitor(Backend(serial_sample()), clock, pid=11139,
        expected_exe='/tmp/candidate/limo_base_node', expected_sha256='a' * 64,
        max_age=0.6, acquire_timeout=1.0)
    assert serial.max_age == 0.6
    assert mod.ProcSerialBackend(clock, 11139, '/dev/fake-serial', fuser_timeout=0.4).fuser_timeout == 0.4
    with pytest.raises(ValueError):
        mod.RosCallerMonitor(Backend(None), clock, driver_pid=11139, max_age=0.31)


def test_serial_guard_checks_process_ownership_busy_probe_and_records_limits():
    clock = Clock()
    backend = Backend(serial_sample())
    guard = serial_guard(clock, backend)
    try:
        assert guard.acquire() and guard.verify()
        evidence = guard.evidence()
        assert evidence["tioc_exclusive_reported"] is False
        assert evidence["initial_open_errno"] == "EBUSY" and evidence["probe_repeated"] is False
        assert evidence["observed_other_fds"] == []
        assert evidence["previous_fd_excluded"] is False
        assert evidence["fuser_pids"] == [11139]
        assert evidence["initial_probe_at"] == 100.0
    finally:
        guard.release()


def test_ebusy_is_recorded_as_one_observation_without_claiming_ioctl_proof():
    clock = Clock()
    backend = Backend(serial_sample())
    guard = module().SerialOwnershipMonitor(backend, clock, pid=11139,
        expected_exe="/tmp/candidate/limo_base_node", expected_sha256="a" * 64,
        poll_interval=0.01, max_age=0.25, acquire_timeout=0.2)
    try:
        assert guard.acquire() and guard.verify()
        proof = guard.evidence()
        assert proof["initial_open_errno"] == "EBUSY"
        assert proof["probe_repeated"] is False
        assert proof["tioc_exclusive_reported"] is False
        assert proof["previous_fd_excluded"] is False
    finally:
        guard.release()


@pytest.mark.parametrize("problem", ["pid_reuse", "exe", "hash", "competitor", "fuser_partial",
                                       "open_succeeds", "wrong_error", "probe_repeated"])
def test_serial_guard_rejects_invalid_identity_ownership_or_busy_proof(problem):
    clock = Clock()
    value = serial_sample()
    if problem == "pid_reuse":
        value["after"]["start_ticks"] += 1
    elif problem == "exe":
        value["before"]["exe"] = "/tmp/other"
    elif problem == "hash":
        value["before"]["sha256"] = "b" * 64
    elif problem == "competitor":
        value["fuser_pids"].append(999)
    elif problem == "fuser_partial":
        value["fuser_complete"] = False
    elif problem == "open_succeeds":
        value["open_errno"] = 0
    elif problem == "wrong_error":
        value["open_errno"] = errno.EACCES
    else:
        value["probe_repeated"] = True
    backend = Backend(value)
    guard = serial_guard(clock, backend)
    try:
        assert not guard.acquire()
        assert not guard.verify()
    finally:
        guard.release()


def test_expired_monitor_fails_closed_without_waiting_for_stalled_worker():
    clock = Clock()
    backend = Backend(serial_sample())
    guard = serial_guard(clock, backend)
    blocked = threading.Event()
    try:
        assert guard.acquire()
        backend.block = blocked
        clock.t += 0.3
        assert not guard.verify()
        blocked.set()
        assert not guard.verify()
    finally:
        blocked.set()
        guard.release()


def test_frozen_clock_cannot_keep_a_stalled_monitor_fresh():
    clock = Clock()
    entered, blocked = threading.Event(), threading.Event()
    class StallBackend(Backend):
        def sample(self):
            if self.block:
                entered.set()
                self.block.wait(1.0)
            return super(StallBackend, self).sample()
    backend = StallBackend(serial_sample())
    guard = serial_guard(clock, backend)
    try:
        assert guard.acquire() and guard.verify()
        backend.block = blocked
        assert entered.wait(0.5)
        time.sleep(0.3)
        assert not guard.verify()
        blocked.set()
        assert not guard.verify()
    finally:
        blocked.set()
        guard.release()


def test_frozen_clock_faults_even_if_worker_keeps_sampling():
    clock = Clock()
    guard = serial_guard(clock, Backend(serial_sample()))
    try:
        assert guard.acquire() and guard.verify()
        time.sleep(0.3)
        assert not guard.verify()
    finally:
        guard.release()


@pytest.mark.parametrize("problem", ["fuser_timeout", "same_pid_new_process"])
def test_active_serial_monitor_latches_on_failed_sample_or_pid_reuse(problem):
    clock = Clock()
    backend = Backend(serial_sample())
    guard = serial_guard(clock, backend)
    try:
        assert guard.acquire() and guard.verify()
        bad = serial_sample()
        if problem == "fuser_timeout":
            bad["fuser_complete"] = False
            bad["fuser_pids"] = None
        else:
            bad["before"]["start_ticks"] += 1
            bad["after"]["start_ticks"] += 1
        backend.set(bad)
        assert diagnostic_wait(guard, "sample_rejected")["fault_latched"]
        assert not guard.verify()
        backend.set(serial_sample())
        assert not guard.verify()
    finally:
        guard.release()


def test_probe_timestamp_must_be_present_and_match_the_original_observation():
    clock = Clock()
    backend = Backend(serial_sample())
    guard = serial_guard(clock, backend)
    try:
        assert guard.acquire() and guard.verify()
        bad = serial_sample()
        bad["probe_at"] = 100.01
        backend.set(bad)
        assert diagnostic_wait(guard, "sample_rejected")["fault_latched"]
        assert not guard.verify()
    finally:
        guard.release()


def test_serial_guard_requires_linked_watchdog_to_stay_fresh_and_match_pid():
    mod = module()
    clock = Clock()
    class FakeWatchdog(object):
        def __init__(self, pid):
            self.driver_pid, self.fresh = pid, True
        def verify(self):
            return self.fresh
    with pytest.raises(ValueError):
        mod.SerialOwnershipMonitor(Backend(serial_sample()), clock, 11139,
            "/tmp/candidate/limo_base_node", "a" * 64, ros_monitor=FakeWatchdog(11140))
    watchdog = FakeWatchdog(11139)
    guard = mod.SerialOwnershipMonitor(Backend(serial_sample()), clock, 11139,
        "/tmp/candidate/limo_base_node", "a" * 64, ros_monitor=watchdog,
        poll_interval=0.01, acquire_timeout=0.4)
    try:
        assert guard.acquire() and guard.verify()
        watchdog.fresh = False
        assert not guard.verify()
        watchdog.fresh = True
        assert not guard.verify()
    finally:
        guard.release()


def test_fuser_parser_rejects_warnings_or_non_pid_output():
    parse = module().parse_fuser
    assert parse(0, b" 11139\n", b"/dev/ttyTHS1:", "/dev/ttyTHS1") == [11139]
    for code, out, err in ((0, b"11139 bad", b"/dev/ttyTHS1:"),
                           (0, b"11139", b"/dev/ttyTHS1: Permission denied"),
                           (1, b"", b"/dev/ttyTHS1:")):
        assert parse(code, out, err, "/dev/ttyTHS1") is None


def test_ros_backend_samples_fake_master_pid_and_watchdog_without_network():
    mod = module()
    clock = Clock()
    subscription = []
    class Subscriber(object):
        def unregister(self):
            self.closed = True
    def subscribe(callback):
        subscription.append(callback)
        return Subscriber()
    class Proxy(object):
        def __init__(self, uri):
            self.uri = uri
        def getSystemState(self, caller):  # noqa: N802 - ROS XML-RPC name
            return [1, "ok", [[("/cmd_vel", ["/relay"]),
                               ("/rosclaw_limo/command_watchdog", ["/limo_base"])], [], []]]
        def lookupNode(self, caller, name):  # noqa: N802 - ROS XML-RPC name
            return [1, "ok", "http://" + name.lstrip("/")]
        def getPid(self, caller):  # noqa: N802 - ROS XML-RPC name
            return [1, "ok", 222 if self.uri.endswith("relay") else 11139]
    backend = mod.Ros1ReadOnlyBackend("http://fake-master", clock, proxy_factory=Proxy,
                                       subscriber_factory=subscribe)
    backend.start()
    try:
        subscription[0](Obj(data="v2 boot_11139 4 0 1 0 /relay 99.9 100.0",
                            _connection_header={"callerid": "/limo_base"}))
        sample = backend.sample()
        assert sample["cmd_vel_publishers"] == ["/relay"]
        assert sample["watchdog_publishers"] == ["/limo_base"]
        assert sample["node_pids"] == {"/relay": 222, "/limo_base": 11139}
        assert sample["watchdog"]["publisher"] == "/limo_base"
    finally:
        backend.stop()


def test_serial_backend_uses_two_identity_checks_and_captures_busy_without_hardware():
    mod = module()
    class FakeProcBackend(mod.ProcSerialBackend):
        def __init__(self):
            super(FakeProcBackend, self).__init__(Clock(), 11139, "/dev/fake-serial")
            self.calls = []
        def read_identity(self):
            self.calls.append("identity")
            return serial_sample()["before"]
        def fuser(self):
            self.calls.append("fuser")
            return [11139]
        def probe(self):
            self.calls.append("probe")
            return errno.EBUSY
    backend = FakeProcBackend()
    snapshot = backend.sample()
    assert backend.calls == ["identity", "fuser", "identity", "probe", "identity"]
    assert snapshot["before"] == snapshot["after"]
    assert snapshot["fuser_pids"] == [11139] and snapshot["open_errno"] == errno.EBUSY


def test_serial_backend_never_repeats_the_open_probe():
    mod = module()
    class FakeProcBackend(mod.ProcSerialBackend):
        def __init__(self):
            super(FakeProcBackend, self).__init__(Clock(), 11139, "/dev/fake-serial")
            self.probe_calls = 0
        def read_identity(self):
            return serial_sample()["before"]
        def fuser(self):
            return [11139]
        def probe(self):
            self.probe_calls += 1
            return errno.EBUSY
    backend = FakeProcBackend()
    assert backend.sample()["open_errno"] == errno.EBUSY
    assert backend.sample()["open_errno"] == errno.EBUSY
    assert backend.probe_calls == 1


def test_serial_backend_reuses_recent_fuser_then_detects_new_holder():
    mod, clock = module(), Clock()
    class FakeProcBackend(mod.ProcSerialBackend):
        def __init__(self):
            super(FakeProcBackend, self).__init__(clock, 11139, '/dev/fake-serial')
            self.fuser_calls = 0
        def read_identity(self):
            return serial_sample()['before']
        def fuser(self):
            self.fuser_calls += 1
            return [11139] if self.fuser_calls == 1 else [11139, 222]
        def probe(self):
            return errno.EBUSY
    backend = FakeProcBackend()
    assert backend.sample()['fuser_pids'] == [11139]
    clock.t += 0.1
    assert backend.sample()['fuser_pids'] == [11139]
    assert backend.fuser_calls == 1
    clock.t += 0.8
    assert backend.sample()['fuser_pids'] == [11139, 222]
    assert backend.fuser_calls == 2


def test_serial_backend_skips_open_if_fuser_failed_or_found_competitor():
    mod = module()
    for holders in (None, [11139, 999]):
        class FakeProcBackend(mod.ProcSerialBackend):
            def __init__(self):
                super(FakeProcBackend, self).__init__(Clock(), 11139, "/dev/fake-serial")
                self.probe_calls = 0
            def read_identity(self):
                return serial_sample()["before"]
            def fuser(self, expected=holders):
                return expected
            def probe(self):
                self.probe_calls += 1
                return errno.EBUSY
        backend = FakeProcBackend()
        assert backend.sample()["open_errno"] is None
        assert backend.probe_calls == 0


def test_aborted_acquire_cannot_probe_after_a_stalled_fuser_returns():
    mod = module()
    clock = Clock()
    entered, finish, completed = threading.Event(), threading.Event(), threading.Event()
    class StalledProcBackend(mod.ProcSerialBackend):
        def __init__(self):
            super(StalledProcBackend, self).__init__(clock, 11139, "/dev/fake-serial")
            self.probe_calls = 0
        def read_identity(self):
            return serial_sample()["before"]
        def fuser(self):
            entered.set()
            finish.wait(1.0)
            return [11139]
        def probe(self):
            self.probe_calls += 1
            return errno.EBUSY
        def stop(self):
            completed.set()
    backend = StalledProcBackend()
    guard = mod.SerialOwnershipMonitor(backend, clock, 11139, "/tmp/candidate/limo_base_node", "a" * 64,
                                       poll_interval=0.01, acquire_timeout=0.05)
    try:
        assert not guard.acquire() and entered.is_set()
        finish.set()
        assert completed.wait(0.5)
        assert backend.probe_calls == 0
    finally:
        finish.set()
        guard.release()


def test_both_monitor_contracts_integrate_with_fake_live_trial_only():
    import live_trial_adapter as adapter
    from relay import _monotonic
    from test_live_trial_adapter import Transport, source
    class RealClock(object):
        def now(self):
            return _monotonic()
        def sleep(self, seconds):
            time.sleep(seconds)
    clock = RealClock()
    class DynamicRos(Backend):
        def __init__(self):
            super(DynamicRos, self).__init__({})
            self.bound = False
        def sample(self):
            now = clock.now()
            publishers = ["/relay"] if self.bound else []
            caller = "/relay" if self.bound else "none"
            return {"cmd_vel_publishers": publishers, "watchdog_publishers": ["/limo_base"],
                    "node_pids": {"/relay": 222, "/limo_base": 11139},
                    "watchdog": {"data": "v2 boot_11139 4 0 %d 0 %s %.6f %.6f" %
                                           (1 if self.bound else 0, caller, now - 0.01, now - 0.002),
                                 "publisher": "/limo_base", "received_at": now}}
    ros_backend = DynamicRos()
    class FakeRelayTransport(Transport):
        def start(self, limit):
            super(FakeRelayTransport, self).start(limit)
            self.process = Obj(pid=222, poll=lambda: None)
            ros_backend.bound = True
    sample = serial_sample()
    sample['fuser_at'] = clock.now()
    serial_backend = Backend(sample)
    ros = module().RosCallerMonitor(ros_backend, clock, 11139, poll_interval=0.01,
                                     acquire_timeout=0.4, bind_timeout=0.4)
    serial = module().SerialOwnershipMonitor(serial_backend, clock, 11139,
        "/tmp/candidate/limo_base_node", "a" * 64, poll_interval=0.01, acquire_timeout=0.4,
        ros_monitor=ros)
    model = adapter.DecisionsAdapter(lambda payload, timeout: {"ok": True, "choice": "forward",
                                                               "captured_at": payload["captured_at"]})
    relay = adapter.RelayAdapter(FakeRelayTransport(), clock)
    result = adapter.LiveTrial(source(clock)[0], model, relay, ros, clock, enable_motion=True,
                               serial_guard=serial).run()
    assert result["reason"] == "motion_limit"
    assert result["serial_guard_evidence"]["initial_open_errno"] == "EBUSY"
    assert result["ros_guard_evidence"]["relay_pid"] == 222
    assert result["physical_stop_verified"] is False


def diagnostic_wait(guard, reason, timeout=0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        report = guard.diagnostic()
        if report["reason"] == reason:
            return report
        time.sleep(0.002)
    assert guard.diagnostic()["reason"] == reason


def test_diagnostic_is_read_only_sanitized_and_reports_healthy_binding():
    clock = Clock()
    backend = Backend(ros_sample(clock))
    guard = ros_guard(clock, backend)
    initial = guard.diagnostic()
    assert initial["reason"] == "not_started"
    assert not initial["fault_latched"]
    assert guard._thread is None
    initial["reason"] = "corrupted"
    assert guard.diagnostic()["reason"] == "not_started"
    try:
        assert guard.acquire()
        backend.set(ros_sample(clock, ["/relay"], caller="/relay", primed=1))
        assert guard.bind_relay(Relay(222))
        assert guard.verify()
        report = guard.diagnostic()
        assert report["reason"] == "ok"
        assert report["driver_pid_expected"] == 11139
        assert report["driver_pid_observed"] == 11139
        assert report["relay_pid_expected"] == 222
        assert report["relay_pid_observed"] == 222
        assert report["cmd_publisher_count"] == 1
        assert report["watchdog_publisher_count"] == 1
        assert report["watchdog_source"] == "matches_cmd_publisher"
        assert "/relay" not in repr(report)
        assert "/limo_base" not in repr(report)
    finally:
        guard.release()


@pytest.mark.parametrize("problem,reason", [
    ("other_publisher", "cmd_publisher_count"),
    ("driver_pid", "driver_pid_mismatch"),
    ("relay_pid", "relay_pid_mismatch"),
    ("wrong_source", "watchdog_source_mismatch"),
    ("latched", "watchdog_latched"),
    ("status_stale", "watchdog_expired"),
    ("unprimed", "watchdog_unprimed"),
    ("missing", "watchdog_missing_or_malformed"),
])
def test_diagnostic_preserves_sanitized_first_fault(problem, reason):
    clock = Clock()
    backend = Backend(ros_sample(clock))
    guard = ros_guard(clock, backend)
    try:
        assert guard.acquire()
        healthy = ros_sample(clock, ["/relay"], caller="/relay", primed=1)
        backend.set(healthy)
        assert guard.bind_relay(Relay(222))
        bad = copy.deepcopy(healthy)
        if problem == "other_publisher":
            bad["cmd_vel_publishers"].append("/sensitive_caller")
        elif problem == "driver_pid":
            bad["node_pids"]["/limo_base"] = 11140
        elif problem == "relay_pid":
            bad["node_pids"]["/relay"] = 333
        elif problem == "wrong_source":
            bad["watchdog"]["data"] = bad["watchdog"]["data"].replace(" /relay ", " /sensitive_caller ")
        elif problem == "latched":
            bad = ros_sample(clock, ["/relay"], caller="/relay", primed=1, latched=1)
        elif problem == "status_stale":
            bad["watchdog"]["received_at"] = 99.0
        elif problem == "unprimed":
            bad = ros_sample(clock, ["/relay"], caller="/relay", primed=0)
        else:
            bad["watchdog"] = None
        backend.set(bad)
        report = diagnostic_wait(guard, reason)
        assert report["fault_latched"] is True
        assert not guard.verify()
        assert "/sensitive_caller" not in repr(report)
        if problem == "driver_pid":
            assert report["driver_pid_observed"] == 11140
        if problem == "relay_pid":
            assert report["relay_pid_observed"] == 333
        if problem == "wrong_source":
            assert report["watchdog_source"] == "other"
        backend.set(healthy)
        assert not guard.verify()
        assert guard.diagnostic()["reason"] == reason
    finally:
        guard.release()


def test_diagnostic_distinguishes_inflight_sample_timeout_and_cached_age():
    clock = Clock()
    entered, blocked = threading.Event(), threading.Event()
    class StallBackend(Backend):
        def sample(self):
            if self.block:
                entered.set()
                self.block.wait(1.0)
            return super(StallBackend, self).sample()
    backend = StallBackend(serial_sample())
    guard = module().SerialOwnershipMonitor(backend, clock, 11139,
        "/tmp/fake", "a" * 64, poll_interval=0.01, max_age=0.05, acquire_timeout=0.2)
    # Only the fake observation changes; no process inspection or device access.
    backend.data["before"]["exe"] = "/tmp/fake"
    backend.data["after"]["exe"] = "/tmp/fake"
    try:
        assert guard.acquire() and guard.verify()
        backend.block = blocked
        assert entered.wait(0.2)
        clock.t += 0.06
        report = guard.diagnostic()
        assert report["reason"] == "cache_expired"
        assert report["sampling"] is True
        time.sleep(0.06)
        report = guard.diagnostic()
        assert report["reason"] == "background_sample_timeout"
        assert not guard.verify()
        assert guard.diagnostic()["reason"] == "background_sample_timeout"
    finally:
        blocked.set()
        guard.release()


@pytest.mark.parametrize("error,reason", [(RuntimeError("secret rpc detail"), "rpc_exception"),
                                            (socket.timeout("secret timeout"), "rpc_timeout")])  # noqa: UP041
def test_diagnostic_reports_rpc_exception_without_details(error, reason):
    mod = module()
    clock = Clock()
    class Proxy(object):
        def __init__(self, uri):
            pass
        def getSystemState(self, caller):  # noqa: N802 - ROS XML-RPC name
            raise error
    class Subscriber(object):
        def unregister(self):
            pass
    backend = mod.Ros1ReadOnlyBackend("http://private-master", clock,
        proxy_factory=Proxy, subscriber_factory=lambda callback: Subscriber())
    guard = ros_guard(clock, backend)
    try:
        assert not guard.acquire()
        report = guard.diagnostic()
        assert report["reason"] == reason
        assert report["fault_latched"] is True
        assert "secret" not in repr(report)
        assert "private-master" not in repr(report)
    finally:
        guard.release()


def test_diagnostic_preserves_acquire_failure_after_release():
    clock = Clock()
    bad = ros_sample(clock)
    bad["node_pids"]["/limo_base"] = 11140
    guard = module().RosCallerMonitor(Backend(bad), clock, 11139, poll_interval=0.005,
                                      max_age=0.25, acquire_timeout=0.04)
    assert not guard.acquire()  # acquire releases internally
    report = guard.diagnostic()
    assert report["reason"] == "driver_pid_mismatch"
    assert report["driver_pid_observed"] == 11140
    assert report["fault_latched"] is False
    guard.release()
    assert guard.diagnostic()["reason"] == "driver_pid_mismatch"


def test_diagnostic_after_release_retains_fault_without_clock_and_is_json_safe():
    import json
    clock = Clock()
    backend = Backend(ros_sample(clock))
    guard = ros_guard(clock, backend)
    try:
        assert guard.acquire()
        healthy = ros_sample(clock, ["/relay"], caller="/relay", primed=1)
        backend.set(healthy)
        assert guard.bind_relay(Relay(222))
        bad = copy.deepcopy(healthy)
        bad["node_pids"]["/limo_base"] = 11140
        backend.set(bad)
        assert diagnostic_wait(guard, "driver_pid_mismatch")["fault_latched"]
    finally:
        guard.release()
    def broken_clock():
        raise RuntimeError("private clock exception")
    clock.now = broken_clock
    report = guard.diagnostic()
    assert report["reason"] == "driver_pid_mismatch"
    assert report["driver_pid_observed"] == 11140
    assert report["fault_latched"] is True
    encoded = json.dumps(report, allow_nan=False)
    assert "private" not in encoded


def test_diagnostic_preserves_initial_background_timeout_after_acquire_releases():
    clock = Clock()
    blocked = threading.Event()
    backend = Backend(ros_sample(clock))
    backend.block = blocked
    guard = module().RosCallerMonitor(backend, clock, 11139, poll_interval=0.005,
                                      max_age=0.05, acquire_timeout=0.08)
    try:
        assert not guard.acquire()
        report = guard.diagnostic()
        assert report["reason"] == "background_sample_timeout"
        assert report["fault_latched"] is False
    finally:
        blocked.set()
        guard.release()
