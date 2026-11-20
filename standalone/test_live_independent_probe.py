"""Argument and opt-in bounds for the read-only independent sensing probe."""
import pytest
from live_independent_probe import parse_args, poll_for, aligned_odometry


def test_stream_selection_defaults_to_both_and_can_isolate_one():
    assert parse_args(['--timeline', '/tmp/a']).stream == 'both'
    assert parse_args(['--timeline', '/tmp/a', '--stream', 'scan']).stream == 'scan'
    assert parse_args(['--timeline', '/tmp/a', '--stream', 'depth']).stream == 'depth'
    with pytest.raises(SystemExit):
        parse_args(['--timeline', '/tmp/a', '--stream', 'motor'])


def test_private_pipe_requires_one_sensor_stream():
    with pytest.raises(SystemExit):
        parse_args(['--timeline', '/tmp/a', '--events-fd', '5'])
    args = parse_args(['--timeline', '/tmp/a', '--stream', 'scan', '--events-fd', '5'])
    assert args.events_fd == 5


def test_controlled_interrupt_returns_a_status_for_cleanup():
    class Clock(object):
        def now(self):
            return 10.
    def interrupt(_seconds):
        raise KeyboardInterrupt()
    assert poll_for(1., Clock(), lambda: False, lambda: True, interrupt) == 'interrupted'


def test_sensor_frame_uses_one_recent_odom_snapshot_and_original_stamp():
    import threading
    class Obj(object):
        pass
    class Guard(object):
        faulted = False
    class Clock(object):
        def now(self):
            return 10.08
    obs = Obj()
    obs.lock, obs.epoch = threading.Lock(), 1
    obs.clock_guard = obs.odom_guard = Guard()
    earlier = {'stamp': 10., 'pose': (0., 0., 0.), 'epoch': 1}
    recent = {'stamp': 10.07, 'pose': (.001, 0., 0.), 'epoch': 1}
    obs.odom_history, obs.latest = [earlier, recent], {'odom': recent}
    aligned, current, now = aligned_odometry(obs, 10., 1, Clock())
    assert aligned == earlier and current == recent and now == 10.08
    assert aligned_odometry(obs, 10., 2, Clock())[0] is None
