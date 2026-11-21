"""Run the existing depth corridor projection in a dedicated process."""

from __future__ import division

import collections
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time

from relay import _monotonic

HEADER = struct.Struct('!II')
MAX_METADATA = 8192
MAX_IMAGE = 4 * 1024 * 1024
PROJECT_BUDGET_SEC = .065
TRANSPORT_BUDGET_SEC = .15


def _json(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':')).encode('ascii')


def _remaining(deadline):
    remaining = deadline - _monotonic()
    if remaining <= 0:
        raise socket.timeout('depth projection deadline')
    return remaining


def _read(sock, size, deadline=None):
    chunks = []
    while size:
        if deadline is not None:
            sock.settimeout(_remaining(deadline))
        chunk = sock.recv(size)
        if not chunk:
            raise EOFError('depth projection disconnected')
        chunks.append(chunk)
        size -= len(chunk)
    return b''.join(chunks)


def _project(params, data):
    import numpy as np
    from projected_depth_gate import project_depth_frame

    try:
        encoding = params['encoding']
        kind = {'16UC1': 'u2', '32FC1': 'f4'}.get(encoding)
        if kind is None or params['is_bigendian'] not in (0, 1):
            return None
        dtype = np.dtype(('>' if params['is_bigendian'] else '<') + kind)
        width, height, step = (params[name] for name in ('width', 'height', 'step'))
        if (any(type(item) is not int for item in (width, height, step))
                or not 1 <= width <= 8192 or not 1 <= height <= 8192
                or step < width * dtype.itemsize or len(data) != step * height):
            return None
        array = np.ndarray((height, width), dtype=dtype, buffer=data, strides=(step, dtype.itemsize))
        return project_depth_frame(array, encoding, params['intrinsics'], params['matrix'],
                                   params['stamp'], params['config'])
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None


class DepthProjectionProcess(object):
    """Exchange one frame at a time over a private socket with a fixed deadline."""

    WORKER_SCRIPT = 'depth_projection_process.py'

    def __init__(self):
        self._lock = threading.Lock()
        self._durations = collections.deque(maxlen=512)
        self._counts = collections.Counter()
        self._closed = False
        self._recovery = None
        self._sock, self.process = self._spawn()

    def _spawn(self):
        sock, peer = socket.socketpair()
        process = None
        try:
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.WORKER_SCRIPT)
            # The worker imports only local projection code and NumPy. It
            # needs neither the ROS environment nor API credentials.
            process = subprocess.Popen([sys.executable, '-u', script], stdin=peer,
                                       close_fds=True, env={'PYTHONNOUSERSITE': '1'})
            if _read(sock, 1, _monotonic() + 4.) != b'R':
                raise ValueError('depth projection failed to initialize')
            return sock, process
        except BaseException:
            sock.close()
            self._stop_process(process)
            raise
        finally:
            peer.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def project(self, params, data):
        if not self._lock.acquire(False):
            return None
        try:
            if self._sock is None:
                self._start_recovery()
                return None
            if type(data) is not bytes or not 1 <= len(data) <= MAX_IMAGE:
                return None
            try:
                payload = _json(params)
            except (TypeError, ValueError, OverflowError):
                return None
            if len(payload) > MAX_METADATA:
                return None
            started = _monotonic()
            deadline = started + TRANSPORT_BUDGET_SEC
            self._sock.settimeout(_remaining(deadline))
            self._sock.sendall(HEADER.pack(len(payload), len(data)) + payload)
            self._sock.settimeout(_remaining(deadline))
            self._sock.sendall(data)
            result_size, body_size = HEADER.unpack(_read(self._sock, HEADER.size, deadline))
            if not 1 <= result_size <= MAX_METADATA or body_size != 0:
                raise ValueError('invalid depth projection response')
            result = json.loads(_read(self._sock, result_size, deadline).decode('ascii'))
            _remaining(deadline)
            if result is not None and (not isinstance(result, dict) or result.get('stamp') != params['stamp']):
                raise ValueError('depth projection frame mismatch')
            duration = _monotonic() - started
            self._durations.append(duration)
            if duration > PROJECT_BUDGET_SEC:
                # Drain a late but valid response so it cannot be mistaken for
                # the next frame. The caller still receives no projection.
                self._counts['over_budget'] += 1
                return None
            self._counts['unavailable' if result is None else 'received'] += 1
            return result
        except socket.timeout:
            self._counts['transport_timeout'] += 1
            self._disconnect()
            return None
        except (IOError, OSError, EOFError, ValueError, TypeError, OverflowError):
            self._counts['transport_fault'] += 1
            self._disconnect()
            return None
        finally:
            self._lock.release()

    def _disconnect(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _start_recovery(self):
        if self._closed or self._recovery is not None or self._counts['restarts'] >= 2:
            return
        self._counts['restarts'] += 1
        self._recovery = threading.Thread(target=self._restart, args=(self.process,), name='projection-recovery')
        self._recovery.daemon = True
        self._recovery.start()

    def _restart(self, previous):
        replacement = None
        try:
            self._stop_process(previous)
            time.sleep(.05)
            with self._lock:
                if self._closed:
                    return
            replacement = self._spawn()
            with self._lock:
                if not self._closed:
                    self._sock, self.process = replacement
                    replacement = None
                    self._counts['restarted'] += 1
        except Exception:
            with self._lock:
                self._counts['restart_failed'] += 1
        finally:
            if replacement is not None:
                sock, process = replacement
                sock.close()
                self._stop_process(process)
            with self._lock:
                self._recovery = None

    def summary(self):
        with self._lock:
            return {'counts': dict(self._counts), 'durations_sec': list(self._durations),
                    'connected': self._sock is not None}

    def close(self):
        with self._lock:
            self._closed = True
            self._disconnect()
            recovery = self._recovery
        if recovery is not None:
            recovery.join(4.5)
            if recovery.is_alive():
                return
        self._stop_process(self.process)

    @staticmethod
    def _stop_process(process):
        if process is None:
            return
        deadline = _monotonic() + .1
        while process.poll() is None and _monotonic() < deadline:
            time.sleep(.005)
        if process.poll() is None:
            process.kill()
        process.wait()


def _run(process=None):
    # The parent installs a private AF_UNIX socket on fd 0. No controller or
    # ROS objects are imported here.
    if process is None:
        import numpy  # noqa: F401
        import projected_depth_gate  # noqa: F401
        process = _project

    peer = socket.fromfd(0, socket.AF_UNIX, socket.SOCK_STREAM)
    peer.sendall(b'R')
    try:
        while True:
            metadata_size, size = HEADER.unpack(_read(peer, HEADER.size))
            if not 1 <= metadata_size <= MAX_METADATA or not 1 <= size <= MAX_IMAGE:
                return 2
            metadata = json.loads(_read(peer, metadata_size).decode('ascii'))
            result = _json(process(metadata, _read(peer, size)))
            peer.sendall(HEADER.pack(len(result), 0) + result)
    except (IOError, OSError, EOFError, TypeError, ValueError):
        return 0
    finally:
        peer.close()


if __name__ == '__main__':
    sys.exit(_run())
