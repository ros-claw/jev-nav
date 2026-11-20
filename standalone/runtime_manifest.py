"""Read-only executable/source digest evidence; never reads environments."""
from __future__ import division

import hashlib
import os


def _digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        before = os.fstat(stream.fileno())
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime):
        raise ValueError('file changed while hashing')
    return digest.hexdigest()


def capture_files(paths):
    result = {}
    for name, path in paths.items():
        try:
            result[name] = {'available': True, 'path': os.path.realpath(path), 'sha256': _digest(path)}
        except (OSError, IOError, TypeError, ValueError) as exc:
            result[name] = {'available': False, 'error_class': type(exc).__name__}
    return result


def same_files(previous, paths):
    current = capture_files(paths)
    return (set(previous) == set(current)
            and all(current[name].get('available') is True and previous[name] == current[name]
                    for name in current))


def capture_process(pid):
    """Read process identity and executable only; never include argv/env."""
    try:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
            raise ValueError('invalid pid')
        root = '/proc/%d' % pid
        def start_ticks():
            with open(root + '/stat') as stream:
                data = stream.read()
            return int(data[data.rfind(')') + 2:].split()[19])
        before = start_ticks()
        executable = os.readlink(root + '/exe')
        digest = _digest(root + '/exe')
        after = start_ticks()
        if before != after:
            raise ValueError('process identity changed')
        return {'available': True, 'pid': pid, 'start_ticks': before,
                'executable': executable, 'sha256': digest}
    except (OSError, IOError, IndexError, TypeError, ValueError) as exc:
        return {'available': False, 'pid': pid, 'error_class': type(exc).__name__}


def same_processes(before, after):
    if not before and not after:
        return None
    if len(before) != len(after):
        return False
    return all(a.get('available') is True and b.get('available') is True and a == b
               for a, b in zip(before, after))
