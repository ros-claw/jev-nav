"""The multiprocess probe is read-only and has bounded runtime."""
import pytest
import errno
import os
import multiprocess_probe as module
from multiprocess_probe import parse_args


def test_probe_requires_output_and_bounded_duration():
    assert parse_args(['--output-prefix', '/tmp/probe']).seconds == 20.
    assert parse_args(['--output-prefix', '/tmp/probe', '--observe-pid', '123']).observe_pid == [123]
    with pytest.raises(SystemExit):
        parse_args(['--output-prefix', '/tmp/probe', '--seconds', '301'])


def test_failed_spawn_does_not_leak_pipe_descriptors(tmpdir):
    prefix = str(tmpdir.join('probe'))
    open(prefix + '.scan.log', 'w').close()
    before = len(os.listdir('/proc/self/fd'))
    with pytest.raises(OSError):
        module._spawn('scan', prefix, 1.)
    assert len(os.listdir('/proc/self/fd')) == before


def test_reader_initialization_failure_reaps_started_child(tmpdir, monkeypatch):
    class Child(object):
        pass
    child, reaped = Child(), []
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *_a, **_k: child)
    monkeypatch.setattr(module, 'FrameReader', lambda *_: (_ for _ in ()).throw(ValueError('bad reader')))
    monkeypatch.setattr(module, '_stop_child', lambda value: reaped.append(value))
    before = len(os.listdir('/proc/self/fd'))
    with pytest.raises(ValueError):
        module._spawn('scan', str(tmpdir.join('probe')), 1.)
    assert reaped == [child] and len(os.listdir('/proc/self/fd')) == before


def test_child_exiting_during_kill_is_still_reaped(monkeypatch):
    class Child(object):
        def poll(self):
            return None
        def send_signal(self, _signal):
            pass
        def kill(self):
            raise OSError(errno.ESRCH, 'already exited')
        def wait(self):
            return 0
    times = iter([0., 4.])
    monkeypatch.setattr(module, '_monotonic', lambda: next(times))
    assert module._stop_child(Child()) == 0
