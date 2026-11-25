from collections import deque
import threading

import pytest

from navigation import Command, Grid, Pose, View


def test_room_recovery_waits_for_new_motion_evidence_and_a_fresh_relay():
    from navigation_actuator import ActuationError, RoomRoamingOutput

    class Clock(object):
        at = 1.

        def now(self):
            return self.at

    class Transport(object):
        process = object()
        exited = False

    class Actuator(object):
        def __init__(self):
            self.transport = Transport()
            self.stop_line = None
            self.caller = '/owned_relay_%d' % len(opened)

        def open(self):
            opened.append(self)

        def close(self):
            self.transport.exited = True

    class Heartbeat(object):
        fault = None

        def __init__(self, actuator, clock):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def submit(self, command, view, now):
            return command

    def current_view(at):
        return View(at, at, at, at, at, 0, Pose(0., 0., 0.),
                    Grid(0., 0., .05, 1, 1, (0,)), True)

    clock, opened = Clock(), []
    stationary = [False]
    output = RoomRoamingOutput(Actuator, clock, lambda after, now: stationary[0], Heartbeat)
    motion = Command(.014, 0., 'forward', 'tracking')
    assert output.update(current_view(1.), 1.) is None
    assert output.submit(motion, current_view(1.), 1.).linear == 0.
    stationary[0] = True
    assert output.update(current_view(1.), 1.) == 'bound'
    assert output.submit(motion, current_view(1.), 1.) == motion

    old = output.actuator
    old.stop_line = b'FAULT heartbeat\n'
    output.heartbeat.fault = ActuationError('navigation_relay_fault')
    stationary[0] = False
    clock.at = 2.
    assert output.update(current_view(2.), 2.) == 'retired'
    assert old.transport.exited
    assert output.submit(motion, current_view(2.), 2.).linear == 0.
    assert output.update(current_view(2.5), 2.5) is None
    stationary[0] = True
    assert output.update(current_view(2.), 2.6) is None
    assert output.update(current_view(2.6), 2.6) == 'bound'
    assert output.actuator is not old
    assert output.callers == {'/owned_relay_0', '/owned_relay_1'}

    output.actuator.stop_line = b'FAULT driver_status\n'
    output.heartbeat.fault = ActuationError('navigation_relay_fault')
    with pytest.raises(ActuationError):
        output.update(current_view(2.7), 2.7)
    assert len(opened) == 2
    output.close()


def test_rebinding_needs_distinct_recent_stationary_wheel_samples():
    from supervised_trial import ObservationRecorder

    recorder = ObservationRecorder.__new__(ObservationRecorder)
    recorder.lock = threading.Lock()
    recorder.odometry = deque([
        (1.01, 1., 2., 0., .4, 0., 50.01, 101),
        (1.07, 1., 2., 0., .4, 0., 50.07, 102),
        (1.13, 1., 2., 0., .4, 0., 50.13, 103),
        (1.19, 1., 2., 0., .4, 0., 50.19, 104),
    ])
    assert recorder.stationary_since(1., 1.21)
    assert not recorder.stationary_since(1., 1.4)
    assert not recorder.stationary_since(1.10, 1.21)
    recorder.odometry[-1] = (1.19, 1.01, 2., .006, .4, 0., 50.19, 104)
    assert not recorder.stationary_since(1., 1.21)


def test_room_roam_remains_opt_in_and_allows_explicit_trial_bounds():
    from run_navigation import parse_args

    args = ['--timeline', '/tmp/trace', '--drive-arcs', '--static-room', '--decisions',
            '--driver-pid', '100', '--driver-exe', '/verified/driver', '--driver-sha256', 'a'*64]
    with pytest.raises(SystemExit):
        parse_args(args)
    live = parse_args(args + ['--roam'])
    assert live.seconds is None and live.max_travel_m is None
    rehearsal = parse_args(args + ['--roam', '--seconds', '60', '--max-travel-m', '.30'])
    assert rehearsal.seconds == 60. and rehearsal.max_travel_m == .30
    with pytest.raises(SystemExit):
        parse_args(['--timeline', '/tmp/trace', '--roam'])


@pytest.mark.parametrize('stamp,arrival,reason', [
    (10.01, 10.24, 'stale_frame'),
    (10.41, 10.42, 'heartbeat'),
    (10.3, 10.24, 'invalid_frame'),
])
def test_relay_distinguishes_timing_holds_from_invalid_room_setpoints(stamp, arrival, reason):
    from relay import RelayGate

    gate = RelayGate(10., arc_navigation=True, room_pivot=True)
    assert gate.accept(dict(sequence=0, stamp=10., linear=.014, angular=0.), 10.18)
    assert not gate.accept(dict(sequence=1, stamp=stamp, linear=.014, angular=0.), arrival)
    assert gate.reason == reason and gate.command(arrival) == (0., 0.)
    assert not gate.accept(dict(sequence=2, stamp=arrival+.01, linear=.014, angular=0.), arrival+.01)
