"""Offline relay registry cleanup tests: all ROS/process/network objects are fake."""
import copy
import errno

import pytest
import relay_registry as registry

NODE = "/standalone_limo_relay_456_1700000000000"
OTHER = "/unrelated"
NODE_API = "http://127.0.0.1:4000/"
SERVICE_API = "rosrpc://127.0.0.1:5000/"


class FakeMaster(object):
    def __init__(self):
        self.state = [
            [["/cmd_vel", [NODE, OTHER]], ["/other_topic", [OTHER]]],
            [["/watchdog", [NODE]], ["/unrelated_sub", [OTHER]]],
            [["/relay/get_loggers", [NODE]], ["/other_service", [OTHER]]],
        ]
        self.nodes = {NODE: NODE_API, OTHER: "http://127.0.0.1:4001/"}
        self.services = {"/relay/get_loggers": SERVICE_API}
        self.calls = []
        self.hook = None

    def call(self, method, args, timeout):
        self.calls.append((method, args, timeout))
        if self.hook:
            replacement = self.hook(self, method, args)
            if replacement is not None:
                return replacement
        if method == "getSystemState":
            return [1, "ok", copy.deepcopy(self.state)]
        if method == "lookupNode":
            return [1, "ok", self.nodes[args[1]]]
        if method == "lookupService":
            return [1, "ok", self.services[args[1]]]
        kinds = {"unregisterPublisher": 0, "unregisterSubscriber": 1, "unregisterService": 2}
        index = kinds[method]
        node, resource, uri = args
        expected_uri = self.services[resource] if index == 2 else self.nodes[node]
        if uri != expected_uri:
            return [1, "wrong uri", 0]
        for entry in self.state[index]:
            if entry[0] == resource and node in entry[1]:
                entry[1].remove(node)
                return [1, "removed", 1]
        return [1, "missing", 0]


def cleanup(master, **options):
    defaults = {"node": NODE, "pid_exists": lambda pid: False,
                "probe_node": lambda uri, timeout: "unreachable",
                "is_local_host": lambda host: host == "127.0.0.1", "timeout": 0.1}
    defaults.update(options)
    return registry.cleanup_relay(master=master, **defaults)


def mutations(master):
    return [(method, args) for method, args, _ in master.calls if method.startswith("unregister")]


def test_default_is_dry_run_and_only_exact_target_resources_are_planned():
    master = FakeMaster()
    result = cleanup(master)
    assert result["status"] == "planned"
    assert result["reason"] == "validated"
    assert result["pid"] == 456
    assert result["planned"] == {"publishers": 1, "subscribers": 1, "services": 1}
    assert result["removed"] == {"publishers": 0, "subscribers": 0, "services": 0}
    assert mutations(master) == []
    assert all(timeout == 0.1 for _, _, timeout in master.calls)
    assert all(method != "lookupNode" or args[1] == NODE for method, args, _ in master.calls)


def test_apply_revalidates_then_removes_only_target_using_real_ros_arguments():
    master = FakeMaster()
    pids, probes = [], []
    result = cleanup(master, apply=True, pid_exists=lambda pid: pids.append(pid) or False,
                     probe_node=lambda uri, timeout: probes.append((uri, timeout)) or "unreachable")
    assert result["status"] == "complete"
    assert result["removed"] == {"publishers": 1, "subscribers": 1, "services": 1}
    assert set(mutations(master)) == {
        ("unregisterPublisher", (NODE, "/cmd_vel", NODE_API)),
        ("unregisterSubscriber", (NODE, "/watchdog", NODE_API)),
        ("unregisterService", (NODE, "/relay/get_loggers", SERVICE_API))}
    assert master.state[0][0][1] == [OTHER]
    assert OTHER in master.state[1][1][1] and OTHER in master.state[2][1][1]
    assert len(pids) >= 4 and set(pids) == {456}
    assert len(probes) >= 4 and all(uri == NODE_API and timeout == 0.1 for uri, timeout in probes)


@pytest.mark.parametrize("node", ["/standalone_limo_relay_1_1700", "/standalone_limo_relay_0_1700",
    "/standalone_limo_relay_0456_1700", "/standalone_limo_relay_456_0",
    "/standalone_limo_relay_456_1700/child", "/ns/standalone_limo_relay_456_1700",
    "/standalone_limo_relay_456_1700\n", "/other_456_1700", "standalone_limo_relay_456_1700",
    "/standalone_limo_relay_4194305_1700", None, True])
def test_invalid_or_other_node_names_refused_before_any_access(node):
    master = FakeMaster()
    result = cleanup(master, node=node, apply=True)
    assert result["status"] == "refused" and result["reason"] == "invalid_node"
    assert master.calls == []


@pytest.mark.parametrize("value,reason", [(True, "pid_exists"), (None, "pid_unknown"),
                                            (0, "pid_unknown"), ("absent", "pid_unknown")])
def test_pid_must_be_definitively_absent(value, reason):
    master = FakeMaster()
    result = cleanup(master, apply=True, pid_exists=lambda pid: value)
    assert result["reason"] == reason
    assert mutations(master) == []


def test_local_host_identity_required_and_uncertainty_refused():
    for value in (False, None, 1):
        master = FakeMaster()
        result = cleanup(master, apply=True, is_local_host=lambda host, current=value: current)
        assert result["reason"] == "node_not_proven_local"
        assert mutations(master) == []
    master = FakeMaster()
    master.nodes[NODE] = "http://user:password@127.0.0.1:4000/"
    assert cleanup(master, apply=True)["reason"] == "invalid_node_uri"
    assert mutations(master) == []


@pytest.mark.parametrize("answer,reason", [("reachable", "node_reachable"),
    ("timeout", "probe_uncertain"), ("unknown", "probe_uncertain"), (False, "probe_uncertain")])
def test_probe_must_definitively_report_unreachable(answer, reason):
    master = FakeMaster()
    result = cleanup(master, apply=True, probe_node=lambda uri, timeout: answer)
    assert result["reason"] == reason
    assert mutations(master) == []


def test_exceptions_are_uncertain_and_never_leak_details():
    def broken(*args):
        raise RuntimeError("sensitive information")
    for key in ("pid_exists", "probe_node", "is_local_host"):
        master = FakeMaster()
        result = cleanup(master, apply=True, **{key: broken})
        assert result["status"] == "refused"
        assert "sensitive" not in repr(result)
        assert mutations(master) == []


def test_replaced_node_uri_before_cleanup_refused_without_mutation():
    master = FakeMaster()
    counter = [0]
    def hook(master, method, args):
        if method == "lookupNode":
            counter[0] += 1
            if counter[0] >= 2:
                master.nodes[NODE] = "http://127.0.0.1:4002/"
    master.hook = hook
    result = cleanup(master, apply=True)
    assert result["reason"] == "registration_changed"
    assert mutations(master) == []


def test_target_registration_change_before_cleanup_refused():
    master = FakeMaster()
    counter = [0]
    def hook(master, method, args):
        if method == "getSystemState":
            counter[0] += 1
            if counter[0] == 2:
                master.state[0].append(["/new", [NODE]])
    master.hook = hook
    assert cleanup(master, apply=True)["reason"] == "registration_changed"
    assert mutations(master) == []


def test_service_uri_change_or_multiple_providers_is_refused():
    master = FakeMaster()
    master.state[2][0][1].append(OTHER)
    assert cleanup(master, apply=True)["reason"] == "ambiguous_service"
    assert mutations(master) == []
    master = FakeMaster()
    counter = [0]
    def hook(master, method, args):
        if method == "lookupService":
            counter[0] += 1
            if counter[0] >= 2:
                master.services[args[1]] = "rosrpc://127.0.0.1:5001/"
    master.hook = hook
    assert cleanup(master, apply=True)["reason"] == "registration_changed"
    assert mutations(master) == []


def test_if_registration_changes_after_first_removal_stop_with_partial_result():
    master = FakeMaster()
    def hook(master, method, args):
        if method == "unregisterPublisher":
            master.nodes[NODE] = "http://127.0.0.1:4999/"
            return [1, "removed", 1]
    master.hook = hook
    result = cleanup(master, apply=True)
    assert result["status"] == "partial" and result["reason"] == "registration_changed"
    assert result["removed"]["publishers"] == 1
    assert len(mutations(master)) == 1


def test_failed_or_uncertain_unregister_stops_without_retry():
    for response in ([1, "not removed", 0], [0, "error", 0], [1, "malformed", True]):
        master = FakeMaster()
        master.hook = lambda m, method, args, current=response: current if method.startswith("unregister") else None
        result = cleanup(master, apply=True)
        assert result["status"] == "indeterminate"
        assert len(mutations(master)) == 1
    master = FakeMaster()
    def broken(master, method, args):
        if method.startswith("unregister"):
            raise RuntimeError("secret failure")
    master.hook = broken
    result = cleanup(master, apply=True)
    assert result["status"] == "indeterminate"
    assert len(mutations(master)) == 1
    assert "secret" not in repr(result)


def test_malformed_master_response_or_invalid_limits_never_delete():
    master = FakeMaster()
    master.hook = lambda m, method, args: [1, "ok", "invalid"] if method == "getSystemState" else None
    assert cleanup(master, apply=True)["status"] == "refused"
    assert mutations(master) == []
    for timeout in (0, -1, 1.1, float("nan"), True):
        master = FakeMaster()
        assert cleanup(master, apply=True, timeout=timeout)["reason"] == "invalid_options"
        assert master.calls == []
    master = FakeMaster()
    assert cleanup(master, apply=1)["reason"] == "invalid_options"
    assert master.calls == []


def test_process_check_refuses_permission_and_unknown_os_errors(monkeypatch):
    def kill(pid, signal):
        raise OSError(errno.ESRCH, "gone")
    monkeypatch.setattr(registry.os, "kill", kill)
    assert registry.pid_exists_local(456) is False
    def kill(pid, signal):
        raise OSError(errno.EPERM, "denied")
    monkeypatch.setattr(registry.os, "kill", kill)
    assert registry.pid_exists_local(456) is None


def test_replacement_during_unreachable_probe_is_detected_before_any_unregistration():
    master = FakeMaster()
    count = [0]
    def probe(uri, timeout):
        count[0] += 1
        if count[0] == 2:
            master.nodes[NODE] = "http://127.0.0.1:4002/"
        return "unreachable"
    result = cleanup(master, apply=True, probe_node=probe)
    assert result["reason"] == "registration_changed"
    assert mutations(master) == []


def test_node_registration_change_during_snapshot_is_detected():
    master = FakeMaster()
    def hook(master, method, args):
        if method == "getSystemState":
            master.nodes[NODE] = "http://127.0.0.1:4002/"
    master.hook = hook
    result = cleanup(master, apply=True)
    assert result["reason"] == "registration_changed"
    assert mutations(master) == []


def test_pid_or_probe_changes_before_mutation_refuse_cleanup():
    for kind in ("pid", "probe"):
        master = FakeMaster()
        calls = []
        def pid_exists(pid, events=calls, current_kind=kind):
            events.append("pid")
            return len(events) > 2 if current_kind == "pid" else False
        def probe(uri, timeout, events=calls, current_kind=kind):
            events.append("probe")
            return "reachable" if current_kind == "probe" and events.count("probe") > 1 else "unreachable"
        result = cleanup(master, apply=True, pid_exists=pid_exists, probe_node=probe)
        assert result["reason"] in ("pid_exists", "node_reachable")
        assert mutations(master) == []


def test_master_client_uses_numeric_ipv4_and_closes_transport(monkeypatch):
    class Proxy(object):
        def __init__(self, uri, transport):
            self.uri, self.transport = uri, transport
            seen.append((uri, transport))
        def getSystemState(self, caller):  # noqa: N802 - ROS XML-RPC name
            return [1, "ok", caller]
    seen = []
    monkeypatch.setattr(registry.xmlrpc, "ServerProxy", Proxy)
    client = registry.RosMaster("http://127.0.0.1:11311/")
    assert seen == []  # no construction-time connection
    assert client.call("getSystemState", ("/caller",), 0.12) == [1, "ok", "/caller"]
    assert seen[0][0] == "http://127.0.0.1:11311/"
    assert seen[0][1].timeout == 0.12
    assert seen[0][1].connections == []
    for uri in ("http://localhost:11311/", "https://127.0.0.1:11311/",
                "http://127.0.0.1:11311/?token=secret", "http://user:pass@127.0.0.1:11311/"):
        with pytest.raises(ValueError):
            registry.RosMaster(uri)


def test_tcp_probe_refusal_timeout_and_reachable_use_bounded_fake_socket(monkeypatch):
    calls = []
    class FakeSocket(object):
        def settimeout(self, timeout):
            calls.append(("timeout", timeout))
        def connect(self, address):
            calls.append(("connect", address))
            if mode[0] == "unreachable":
                raise registry.socket.error(errno.ECONNREFUSED, "refused")
            if mode[0] == "timeout":
                raise registry.socket.timeout("late")  # noqa: UP041
        def close(self):
            calls.append(("close",))
    monkeypatch.setattr(registry.socket, "socket", lambda *args: FakeSocket())
    mode = ["unreachable"]
    assert registry.probe_node_tcp(NODE_API, 0.1) == "unreachable"
    mode[0] = "timeout"
    assert registry.probe_node_tcp(NODE_API, 0.1) == "timeout"
    mode[0] = "reachable"
    assert registry.probe_node_tcp(NODE_API, 0.1) == "reachable"
    assert calls.count(("timeout", 0.1)) == 3
    assert calls.count(("connect", ("127.0.0.1", 4000))) == 3
    assert calls.count(("close",)) == 3
    assert registry.probe_node_tcp("http://named-host:4000/", 0.1) == "unknown"
    assert calls.count(("connect", ("127.0.0.1", 4000))) == 3


def test_node_uri_change_during_last_snapshot_is_detected_before_unregistration():
    master = FakeMaster()
    count = [0]
    def hook(master, method, args):
        if method == "getSystemState":
            count[0] += 1
            if count[0] == 2:
                master.nodes[NODE] = "http://127.0.0.1:4002/"
    master.hook = hook
    result = cleanup(master, apply=True)
    assert result["reason"] == "registration_changed"
    assert mutations(master) == []


def test_against_ros1_registration_manager_in_memory_only():
    import sys
    if sys.version_info[0] != 2:  # noqa: UP036 - ROS Melodic runs Python 2
        pytest.skip("ROS Melodic's installed registration manager is Python 2")
    registrations = pytest.importorskip("rosmaster.registrations")
    class Pool(object):
        def queue_task(self, *args):
            raise AssertionError("no task or network access expected")
    manager = registrations.RegistrationManager(Pool())
    manager.register_publisher("/cmd_vel", NODE, NODE_API)
    manager.register_publisher("/cmd_vel", OTHER, "http://127.0.0.1:4001/")
    manager.register_subscriber("/watchdog", NODE, NODE_API)
    manager.register_service("/relay/get_loggers", NODE, NODE_API, SERVICE_API)
    class InMemoryRosMaster(object):
        def call(self, method, args, timeout):
            if method == "getSystemState":
                return [1, "", [manager.publishers.get_state(), manager.subscribers.get_state(),
                                 manager.services.get_state()]]
            if method == "lookupNode":
                node = manager.get_node(args[1])
                return [1, "", node.api] if node else [-1, "", ""]
            if method == "lookupService":
                return [1, "", manager.services.get_service_api(args[1])]
            methods = {"unregisterPublisher": manager.unregister_publisher,
                       "unregisterSubscriber": manager.unregister_subscriber,
                       "unregisterService": manager.unregister_service}
            node, resource, uri = args
            return methods[method](resource, node, uri)
    result = cleanup(InMemoryRosMaster(), apply=True)
    assert result["status"] == "complete"
    assert manager.get_node(NODE) is None
    assert manager.get_node(OTHER) is not None
    assert manager.publishers.get_state() == [["/cmd_vel", [OTHER]]]
    assert manager.subscribers.get_state() == []
    assert manager.services.get_state() == []


def test_non_integer_ros_success_code_is_not_accepted():
    master = FakeMaster()
    master.hook = lambda m, method, args: [1.0, "ambiguous", copy.deepcopy(m.state)] if method == "getSystemState" else None
    result = cleanup(master, apply=True)
    assert result["status"] == "refused" and result["reason"] == "master_invalid_response"
    assert mutations(master) == []


def test_process_checker_rejects_invalid_pid_without_system_call(monkeypatch):
    calls = []
    monkeypatch.setattr(registry.os, "kill", lambda *args: calls.append(args))
    for pid in (0, 1, -1, True, None, 4194305):
        assert registry.pid_exists_local(pid) is None
    assert calls == []


def test_master_without_accessible_call_is_a_refusal():
    class BrokenMaster(object):
        @property
        def call(self):
            raise RuntimeError("secret")
    result = cleanup(BrokenMaster(), apply=True)
    assert result["status"] == "refused" and result["reason"] == "invalid_options"
    assert "secret" not in repr(result)
