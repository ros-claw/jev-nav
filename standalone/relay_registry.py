#!/usr/bin/env python2
"""Opt-in cleanup of one precisely named, demonstrably dead relay registration.

Import and construction do not contact ROS. cleanup_relay() is dry-run by
 default; apply=True is required to unregister anything. The caller must provide
an authoritative is_local_host(host) check for its own host and PID namespace.
Only a PID definitely absent *and* a node endpoint definitely unreachable
allow cleanup; timeouts, permissions errors, malformed data and any ambiguity
refuse it. The master and probe interfaces must honour the supplied timeout.

ROS master queries and unregister calls are not an atomic transaction. Every
operation is revalidated, and the master's unregister APIs also compare caller
and API handles, but a replacement can still race the final check. An RPC can
succeed on the master while its reply is lost: such an outcome is reported
indeterminate and never retried. This tool provides no motion authorization.
"""
from __future__ import division

import errno
import math
import os
import re
import socket

try:
    import httplib as httpclient
    import urlparse
    import xmlrpclib as xmlrpc
except ImportError:
    import http.client as httpclient
    import urllib.parse as urlparse
    import xmlrpc.client as xmlrpc

try:
    INTS = (int, long)  # noqa: F821
    STRINGS = (str, unicode)  # noqa: F821
except NameError:
    INTS = (int,)
    STRINGS = (str,)

_NODE = re.compile(r"\A/standalone_limo_relay_([1-9][0-9]{0,6})_([1-9][0-9]{0,19})\Z")
_NAME = re.compile(r"\A/[A-Za-z0-9_/]+\Z")
_CALLER = "/standalone_relay_registry"
_KINDS = ("publishers", "subscribers", "services")
_METHODS = ("unregisterPublisher", "unregisterSubscriber", "unregisterService")


class _RefusalError(Exception):
    pass


def _number(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, INTS + (float,))
                and not math.isnan(value) and not math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _node_pid(node):
    if not isinstance(node, STRINGS) or len(node) > 128:
        return None
    match = _NODE.match(node)
    if match is None:
        return None
    pid = int(match.group(1))
    return pid if 1 < pid <= 4194304 else None


def _valid_name(name):
    return (isinstance(name, STRINGS) and len(name) <= 1024 and bool(_NAME.match(name))
            and not name.endswith("/") and "//" not in name)


def _uri_host(uri, scheme):
    if not isinstance(uri, STRINGS) or len(uri) > 2048:
        return None
    try:
        parts = urlparse.urlsplit(uri)
        if (parts.scheme != scheme or not parts.hostname or parts.username is not None
                or parts.password is not None or parts.path not in ("", "/")
                or parts.query or parts.fragment or not 1 <= parts.port <= 65535):
            return None
        return parts.hostname
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def pid_exists_local(pid):
    """Return False only for ESRCH; EPERM and all other errors are unknown."""
    if isinstance(pid, bool) or not isinstance(pid, INTS) or not 1 < pid <= 4194304:
        return None
    try:
        os.kill(pid, 0)
        return True
    except OSError as exc:
        return False if exc.errno == errno.ESRCH else None


def probe_node_tcp(uri, timeout):
    """Probe literal IPv4 node URI; only connection refusal proves unreachable.

    A successful TCP handshake means reachable. Timeouts and all other errors
    are unknown. No DNS is performed; names and IPv6 need an injected probe.
    """
    host = _uri_host(uri, "http")
    if host is None or not _number(timeout) or not 0 < timeout <= 1:
        return "unknown"
    sock = None
    try:
        socket.inet_pton(socket.AF_INET, host)
        port = urlparse.urlsplit(uri).port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        return "reachable"
    except socket.timeout:  # noqa: UP041 - Python 2
        return "timeout"
    except (socket.error, ValueError, TypeError) as exc:
        return "unreachable" if getattr(exc, "errno", None) == errno.ECONNREFUSED else "unknown"
    finally:
        if sock is not None:
            sock.close()


class _TimeoutTransport(xmlrpc.Transport):
    def __init__(self, timeout):
        xmlrpc.Transport.__init__(self)
        self.timeout = timeout
        self.connections = []

    def make_connection(self, host):
        connection = httpclient.HTTPConnection(host, timeout=self.timeout)
        self.connections.append(connection)
        return connection

    def close(self):
        for connection in self.connections:
            connection.close()
        self.connections = []


class RosMaster(object):
    """Master adapter with per-RPC socket timeout and a literal IPv4 master URI."""
    def __init__(self, master_uri):
        host = _uri_host(master_uri, "http")
        if host is None:
            raise ValueError("invalid master URI")
        try:
            socket.inet_pton(socket.AF_INET, host)
        except (socket.error, ValueError, TypeError):
            raise ValueError("master URI must use a literal IPv4 address")
        self.master_uri = master_uri

    def call(self, method, args, timeout):
        if (method not in ("getSystemState", "lookupNode", "lookupService") + _METHODS
                or not _number(timeout) or not 0 < timeout <= 1):
            raise ValueError("invalid master call")
        transport = _TimeoutTransport(timeout)
        try:
            proxy = xmlrpc.ServerProxy(self.master_uri, transport=transport)
            return getattr(proxy, method)(*args)
        finally:
            transport.close()


def _call(master, method, args, timeout):
    try:
        response = master.call(method, args, timeout)
    except BaseException:
        raise _RefusalError("master_exception")
    if (not isinstance(response, (tuple, list)) or len(response) != 3
            or isinstance(response[0], bool) or not isinstance(response[0], INTS)
            or response[0] != 1):
        raise _RefusalError("master_invalid_response")
    return response[2]


def _target_state(master, node, timeout):
    state = _call(master, "getSystemState", (_CALLER,), timeout)
    if not isinstance(state, (tuple, list)) or len(state) != 3:
        raise _RefusalError("master_invalid_response")
    selected = []
    for index, entries in enumerate(state):
        if not isinstance(entries, (tuple, list)):
            raise _RefusalError("master_invalid_response")
        found, seen = set(), set()
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise _RefusalError("master_invalid_response")
            name, owners = entry
            if (not _valid_name(name) or name in seen or not isinstance(owners, (tuple, list))
                    or not all(_valid_name(owner) for owner in owners)
                    or len(set(owners)) != len(owners)):
                raise _RefusalError("master_invalid_response")
            seen.add(name)
            if node in owners:
                if index == 2 and len(owners) != 1:
                    raise _RefusalError("ambiguous_service")
                found.add(name)
        selected.append(found)
    return selected


def _local(is_local_host, host, reason):
    try:
        answer = is_local_host(host)
    except BaseException:
        answer = None
    if answer is not True:
        raise _RefusalError(reason)


def _snapshot(master, node, timeout, is_local_host):
    uri = _call(master, "lookupNode", (_CALLER, node), timeout)
    host = _uri_host(uri, "http")
    if host is None:
        raise _RefusalError("invalid_node_uri")
    _local(is_local_host, host, "node_not_proven_local")
    resources = _target_state(master, node, timeout)
    service_uris = {}
    for service in sorted(resources[2]):
        service_uri = _call(master, "lookupService", (_CALLER, service), timeout)
        service_host = _uri_host(service_uri, "rosrpc")
        if service_host is None:
            raise _RefusalError("invalid_service_uri")
        _local(is_local_host, service_host, "service_not_proven_local")
        service_uris[service] = service_uri
    if _call(master, "lookupNode", (_CALLER, node), timeout) != uri:
        raise _RefusalError("registration_changed")
    return uri, resources, service_uris


def _dead(pid, uri, timeout, pid_exists, probe_node):
    try:
        exists = pid_exists(pid)
    except BaseException:
        exists = None
    if exists is True:
        raise _RefusalError("pid_exists")
    if exists is not False:
        raise _RefusalError("pid_unknown")
    try:
        probe = probe_node(uri, timeout)
    except BaseException:
        probe = "unknown"
    if probe == "reachable":
        raise _RefusalError("node_reachable")
    if probe != "unreachable":
        raise _RefusalError("probe_uncertain")


def cleanup_relay(master, node, is_local_host, pid_exists=pid_exists_local,
                  probe_node=probe_node_tcp, timeout=0.1, apply=False):
    """Validate one relay and optionally unregister only its own registrations.

    master.call(method, args, timeout) must implement bounded ROS XML-RPC and
    return the unmodified (code, message, value) tuple. is_local_host(host)
    must return True only for the local host in the PID checker's namespace;
    pid_exists(pid) must return exactly False when absent. probe_node(uri,
    timeout) must return exactly 'unreachable' on a definite refusal. Each
    callback must honour its timeout where supplied. Results contain counts and
    fixed reasons, never raw ROS replies or exception strings. apply=False
    performs all checks without issuing unregister calls.
    """
    result = {"status": "refused", "reason": "invalid_node", "pid": None,
              "planned": dict.fromkeys(_KINDS, 0),
              "removed": dict.fromkeys(_KINDS, 0)}
    pid = _node_pid(node)
    if pid is None:
        return result
    result["pid"] = pid
    try:
        master_callable = callable(getattr(master, "call", None))
    except BaseException:
        master_callable = False
    if (apply is not True and apply is not False or not _number(timeout)
            or not 0 < timeout <= 1 or not callable(is_local_host)
            or not callable(pid_exists) or not callable(probe_node)
            or not master_callable):
        result["reason"] = "invalid_options"
        return result
    try:
        # Check the PID before even reading the graph. No unknown PID qualifies.
        try:
            exists = pid_exists(pid)
        except BaseException:
            exists = None
        if exists is True:
            raise _RefusalError("pid_exists")
        if exists is not False:
            raise _RefusalError("pid_unknown")
        uri, expected, services = _snapshot(master, node, timeout, is_local_host)
        _dead(pid, uri, timeout, pid_exists, probe_node)
        result["planned"] = {kind: len(expected[index]) for index, kind in enumerate(_KINDS)}
        operations = [(index, name) for index, names in enumerate(expected) for name in sorted(names)]
        if not operations:
            raise _RefusalError("no_target_registrations")

        def revalidate():
            _dead(pid, uri, timeout, pid_exists, probe_node)
            current_uri, current, current_services = _snapshot(master, node, timeout, is_local_host)
            if current_uri != uri or current != expected or current_services != services:
                raise _RefusalError("registration_changed")

        if not apply:
            revalidate()
            result["status"], result["reason"] = "planned", "validated"
            return result
        for index, name in operations:
            revalidate()
            api = services[name] if index == 2 else uri
            try:
                removed = _call(master, _METHODS[index], (node, name, api), timeout)
            except _RefusalError:
                result["status"], result["reason"] = "indeterminate", "unregister_unconfirmed"
                return result
            if isinstance(removed, bool) or not isinstance(removed, INTS) or removed != 1:
                result["status"], result["reason"] = "indeterminate", "unregister_unconfirmed"
                return result
            result["removed"][_KINDS[index]] += 1
            expected[index].remove(name)
            if index == 2:
                del services[name]
        # The final unregister may remove lookupNode's record. Verify graph only.
        if any(_target_state(master, node, timeout)):
            raise _RefusalError("registration_changed")
        result["status"], result["reason"] = "complete", "unregistered"
        return result
    except _RefusalError as exc:
        result["reason"] = exc.args[0]
        if any(result["removed"].values()):
            result["status"] = "partial"
        return result
    except BaseException:
        result["reason"] = "unexpected_error"
        if any(result["removed"].values()):
            result["status"] = "partial"
        return result
