import math

from navigation import Pose
from progress_monitor import ProgressMonitor


def run(wheel, independent=None, command=(.02, 0.), interrupt=False):
    monitor = ProgressMonitor(require_independent=independent is not None)
    for step in range(1, 121):
        at = step * .05
        applied = (0., 0.) if interrupt and step % 17 == 0 else command
        camera = independent(at) if independent is not None else None
        fault = monitor.update(at, [(at,) + applied], wheel(at), 0, camera)
        if fault:
            return fault, at
    return None, at


def test_stationary_wheel_odometry_is_flagged_under_a_sustained_relay_output():
    fault, at = run(lambda _: Pose(0., 0., 0.))
    assert fault == 'no_progress' and 2.9 < at < 3.3


def test_independent_translation_can_distinguish_slip_and_missing_evidence():
    wheel = lambda t: Pose(t * .02, 0., 0.)
    assert run(wheel, lambda t: ('camera', Pose(0., 0., 0.)))[0] == 'slip'
    assert run(wheel, lambda t: None)[0] == 'pose_uncertain'
    assert run(wheel, lambda t: ('camera', wheel(t)))[0] is None


def test_interruptions_and_successful_rotation_do_not_look_like_a_stall():
    assert run(lambda _: Pose(0., 0., 0.), interrupt=True)[0] is None
    assert run(lambda t: Pose(0., 0., .05 * t), command=(0., .05))[0] is None
    assert run(lambda _: Pose(0., 0., 0.), command=(0., .05))[0] == 'no_progress'
    assert run(lambda _: Pose(0., 0., 0.), command=(.003, 0.))[0] is None


def test_stale_command_and_epoch_change_reset_the_progress_window():
    monitor = ProgressMonitor()
    pose = Pose(0., 0., 0.)
    for i in range(1, 55):
        at = i * .05
        assert monitor.update(at, [(at, .02, 0.)], pose, 0) is None
    assert monitor.update(3.10, [], pose, 0) is None
    for i in range(63, 117):
        at = i * .05
        assert monitor.update(at, [(at, .02, 0.)], pose, 1) is None


def test_sustained_creep_is_checked_when_it_accumulates_enough_expected_distance():
    monitor = ProgressMonitor()
    pose = Pose(0., 0., 0.)
    for i in range(1, 201):
        at = i * .05
        assert monitor.update(at, [(at, .006, 0.)], pose, 0) is None
    assert monitor.update(10.05, [(10.05, .006, 0.)], pose, 0) == 'no_progress'
