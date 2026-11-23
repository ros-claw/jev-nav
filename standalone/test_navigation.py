from __future__ import division

import collections
import json
import math
import threading
import time

import pytest

from navigation import Command, Grid, Navigation, Pose, View, arc_pose, sweep_clear
from navigation_runtime import NavigationRuntime


def grid(markings=None):
    markings = markings or {}
    size, resolution = 100, .04
    cells = [0] * (size * size)
    for (x, y), value in markings.items():
        index = int(math.floor((y + 2.) / resolution)) * size + int(math.floor((x + 2.) / resolution))
        cells[index] = value
    return Grid(-2., -2., resolution, size, size, tuple(cells))


def view(now, pose=None, cells=None, scan=None, depth=None, epoch=0):
    return View(now, now if scan is None else scan, now if depth is None else depth,
                now, now, epoch, pose or Pose(0., 0., 0.), cells or grid(), True)


def select(navigation, key, at, observation):
    request = navigation.request(observation, at)
    assert key in request.choices
    assert navigation.accept(request, key, observation, at)
    return request


def test_goal_keeps_running_across_model_failure_and_uses_odometry_to_complete():
    navigation = Navigation()
    observation = view(1.)
    select(navigation, 'forward', 1., observation)
    pose = observation.pose
    first, last = None, None
    failed_choice = False
    for index in range(1, 281):
        now = 1. + index * .05
        observation = view(now, pose)
        if pose.x > .17 and not failed_choice:
            follow_up = navigation.request(observation, now)
            assert not navigation.accept(follow_up, None, observation, now)
            failed_choice = True
        output = navigation.step(observation, now)
        if output.linear > 0:
            first = now if first is None else first
            last = now
        pose = arc_pose(pose, output.linear * .05, 0.)
    assert last - first > 8.0
    assert pose.x > .21
    assert failed_choice
    assert output.reason == 'waiting_for_goal'


@pytest.mark.parametrize('replacement', ['occupied', 'stale_scan', 'stale_depth'])
def test_lost_geometry_stops_immediately_and_a_real_obstacle_discards_the_active_goal(replacement):
    navigation = Navigation()
    select(navigation, 'forward', 1., view(1.))
    assert navigation.step(view(1.05), 1.05).linear > 0
    if replacement == 'occupied':
        observation = view(1.10, cells=grid({(.25, 0.): 100}))
    elif replacement == 'stale_scan':
        observation = view(1.10, scan=.70)
    else:
        observation = view(1.10, depth=.70)
    stopped = navigation.step(observation, 1.10)
    assert stopped.linear == stopped.angular == 0.
    recovered = navigation.step(view(1.15), 1.15).linear
    if replacement == 'occupied':
        assert recovered == 0.
    else:
        assert recovered > 0.


def test_old_choice_cannot_restart_a_cleared_goal():
    navigation = Navigation()
    request = select(navigation, 'forward', 1., view(1.))
    assert navigation.step(view(1.05, cells=grid({(.25, 0.): 100})), 1.05).linear == 0.
    assert not navigation.accept(request, 'forward', view(1.10), 1.10)


def test_left_path_turns_and_model_only_sees_geometrically_available_choices():
    navigation = Navigation()
    observation = view(1., cells=grid({(.377, -.097): -1}))
    request = navigation.request(observation, 1.)
    assert 'stop' in request.choices and 'right' not in request.choices
    assert 'left' in request.choices
    assert not navigation.accept(request, 'right', observation, 1.)
    select(navigation, 'left', 1., observation)
    output = navigation.step(view(1.05), 1.05)
    assert output.linear > 0. and output.angular > 0.


def test_unknown_swept_space_is_rejected_but_occupied_body_is_never_ignored():
    pose = Pose(0., 0., 0.)
    assert sweep_clear(grid(), pose, .20, .8)
    assert not sweep_clear(grid({(.25, .08): -1}), pose, .20, .8)
    assert not sweep_clear(grid({(0., 0.): 100}), pose, .20, 0.)


def test_arc_does_not_claim_the_area_directly_behind_a_forward_moving_body():
    pose = Pose(0., 0., 0.)
    assert sweep_clear(grid({(-.18, .02): -1}), pose, .24, .8)
    assert not sweep_clear(grid({(.05, -.117): -1}), pose, .24, .8)


def test_straight_goal_uses_forward_only_if_an_unobserved_correction_is_blocked():
    navigation = Navigation()
    select(navigation, 'forward', 1., view(1.))
    cells = grid({(-.138, .114): -1})
    observation = view(1.05, Pose(.01, .002, .01), cells)
    command = navigation.step(observation, 1.05)
    assert command.action == 'forward' and command.linear > 0. and command.angular == 0.


def test_successor_is_decided_while_the_robot_follows_current_segment():
    navigation = Navigation()
    select(navigation, 'forward', 1., view(1.))
    near_end = view(9., Pose(.17, 0., 0.))
    request = navigation.request(near_end, 9.)
    assert 'left' in request.choices
    assert navigation.step(near_end, 9.).action == 'forward'
    accepted_at = view(11., Pose(.21, 0., 0.))
    assert navigation.accept(request, 'left', accepted_at, 11.)
    transition = navigation.step(view(11.5, Pose(.232, 0., 0.)), 11.5)
    assert transition.action == 'left' and transition.angular > 0.


def test_hardware_profile_offers_forward_and_does_not_issue_steering():
    navigation = Navigation(forward_only=True, speed=.014)
    observation = view(1.)
    request = navigation.request(observation, 1.)
    assert request.choices == ('forward', 'stop')
    assert navigation.accept(request, 'forward', observation, 1.)
    command = navigation.step(view(1.05, Pose(.01, .005, .02)), 1.05)
    assert 0. < command.linear <= .014
    assert command.angular == 0.
    command = navigation.step(view(1.10, Pose(.01, .025, .02)), 1.10)
    assert command.reason == 'path_tracking_lost' and command.linear == 0.


@pytest.mark.parametrize('timed_out', [False, True])
def test_forward_successor_has_time_to_retry_before_the_first_segment_ends(timed_out):
    navigation = Navigation(forward_only=True, speed=.014)
    first = select(navigation, 'forward', 1., view(1.))
    early = view(9., Pose(.095, 0., 0.))
    request = navigation.request(early, 9.)
    assert request is not None and request.predecessor == first.sequence
    if timed_out:
        assert not navigation.accept(request, None, view(15., Pose(.164, 0., 0.)), 15.)
        request = navigation.request(view(16., Pose(.175, 0., 0.)), 16.)
        assert navigation.accept(request, 'forward', view(20., Pose(.219, 0., 0.)), 20.)
    else:
        assert navigation.accept(request, 'forward', early, 9.1)
    assert navigation.step(view(21., Pose(.231, 0., 0.)), 21.).linear > 0.
    assert navigation.goal.sequence == request.sequence


def test_navigation_trajectory_counts_motion_after_a_pause():
    from run_navigation import summarize_navigation_trajectory
    commands = [(1., .014, 0.), (1.05, .014, 0.), (1.1, 0., 0.),
                (1.2, .014, 0.), (1.25, .014, 0.), (1.30, .014, 0.), (1.35, 0., 0.)]
    odometry = [(1., 0., 0., 0.), (1.15, .01, 0., 0.), (1.35, .02, 0., 0.)]
    result = summarize_navigation_trajectory({'commands': commands, 'odom': odometry})
    assert result['nonzero_cmd_vel_messages'] == 5
    assert result['longest_nonzero_sec'] == pytest.approx(.15)
    assert result['odom_displacement_m'] == .02 and result['zero_after_last_motion'] == 1.35


def test_turn_measurement_unwraps_heading_while_the_robot_moves():
    from run_navigation import summarize_navigation_trajectory
    commands = [(1., .014, .011), (1.05, .014, .011), (1.10, 0., 0.)]
    odometry = [(1., 0., 0., .014, 3.12), (1.05, .001, 0., .014, -3.13),
                (1.10, .002, 0., 0., -3.09)]
    result = summarize_navigation_trajectory({'commands': commands, 'odom': odometry})
    assert result['nonzero_angular_cmd_vel_messages'] == 2
    assert result['odom_heading_span_rad'] == pytest.approx(2 * math.pi - 6.21)


def test_room_loop_requires_turning_and_returning_to_the_arc_start():
    from run_navigation import summarize_room_loop
    odom = []
    for i in range(128):
        angle = i * 2. * math.pi / 127.
        yaw = math.atan2(math.sin(angle), math.cos(angle))
        odom.append((1. + i * .1, math.sin(angle), 1. - math.cos(angle), .018, yaw))
    trace = {1: {'started_at': 1., 'action': 'left', 'heading_change_rad': .18}}
    trajectory = {'odom': odom, 'commands': [(row[0], .018, .014) for row in odom]}
    assert summarize_room_loop(trajectory, trace)['closed']
    assert summarize_room_loop(trajectory, trace)['continuous_commands']
    trajectory['commands'][20] = (odom[20][0], 0., 0.)
    assert not summarize_room_loop(trajectory, trace)['continuous_commands']
    trajectory['odom'] = odom[:90]
    assert not summarize_room_loop(trajectory, trace)['closed']


def test_low_light_uplift_uses_a_pixelwise_bounded_transform():
    import numpy as np
    from run_navigation import brighten_low_light

    dark = np.dstack([np.arange(6, 26, dtype=np.uint8).reshape(4, 5)] * 3)
    brightened = brighten_low_light(dark)
    assert brightened.shape == dark.shape and brightened.dtype == dark.dtype
    assert (brightened >= dark).all() and brightened.max() > dark.max()
    assert brighten_low_light(np.zeros_like(dark)).max() == 0
    well_exposed = dark + 100
    assert brighten_low_light(well_exposed) is well_exposed


@pytest.mark.parametrize('arc_navigation', [False, True])
def test_hardware_authority_loss_withholds_velocity_and_cannot_hide_relay_fault(arc_navigation):
    from navigation_actuator import ActuationError, NavigationActuator
    from relay import RelayGate, _monotonic

    class Clock(object):
        now = staticmethod(_monotonic)
        sleep = staticmethod(time.sleep)

    class Transport(object):
        def start_forward_navigation(self):
            self.gate = RelayGate(_monotonic(), forward_navigation=True)
            self.ready = b'READY forward_navigation\n'

        def start_arc_navigation(self):
            self.gate = RelayGate(_monotonic(), arc_navigation=True)
            self.ready = b'READY arc_navigation\n'

        def read_line(self, timeout):
            return self.ready

        def check_status(self):
            status, self.status = self.status, None
            return status

        def write_line(self, data, timeout):
            return self.gate.accept(json.loads(data), _monotonic())

        def close(self):
            self.gate.stop('closed')

    class Authority(object):
        def acquire(self):
            self.allowed, self.released = True, False
            return True

        def bind_relay(self, actuator):
            return True

        def verify(self):
            return self.allowed

        def evidence(self):
            return {'relay_node': '/independent_relay'}

        def cached_command_deadline(self):
            return _monotonic() + .05 if self.allowed else None

        def release(self):
            self.released = True

    transport = Transport()
    transport.status = None
    ros, serial = Authority(), Authority()
    actuator = NavigationActuator(transport, ros, serial, Clock(), arc_navigation=arc_navigation)
    try:
        actuator.open()
        command = (Command(.014, -.0112, 'right', 'tracking') if arc_navigation
                   else Command(.014, 0., 'forward', 'tracking'))
        assert actuator.publish(command) == command
        assert transport.gate.command(_monotonic()) == (command.linear, command.angular)
        serial.allowed = False
        actual = actuator.publish(command)
        assert actual.linear == 0. and actual.reason == 'command_authority_unavailable'
        assert actuator.unavailable_guards == ('serial_authority',)
        assert transport.gate.command(_monotonic()) == (0., 0.)
        serial.allowed = True
        assert actuator.publish(command) == command
        assert not actuator.unavailable_guards
        transport.status = b'FAULT heartbeat\n'
        with pytest.raises(ActuationError, match='navigation_relay_fault'):
            actuator.publish(command)
        with pytest.raises(ActuationError, match='navigation_relay_fault'):
            actuator.publish(Command(0., 0., 'stop', 'shutdown'))
        assert actuator.stop_line == b'FAULT heartbeat\n'
    finally:
        actuator.close()
    assert ros.released and serial.released and transport.gate.command(_monotonic()) == (0., 0.)


def test_output_heartbeats_continue_at_zero_when_the_controller_stops_updating():
    from navigation_actuator import NavigationHeartbeat
    from relay import _monotonic
    from supervised_trial import RealClock

    class Actuator(object):
        def __init__(self):
            self.commands = []

        def publish(self, command):
            self.commands.append((_monotonic(), command))
            return command

    actuator = Actuator()
    heartbeat = NavigationHeartbeat(actuator, RealClock())
    heartbeat.start()
    try:
        start = _monotonic()
        heartbeat.submit(Command(.018, .014, 'left', 'tracking'), view(start), start)
        time.sleep(.30)
    finally:
        heartbeat.close()
    observed = actuator.commands
    assert any(command.linear > 0. for _, command in observed)
    expired = [command for at, command in observed if at - start >= .16]
    assert len(expired) >= 2 and all(command.linear == command.angular == 0. for command in expired)


def test_answer_received_during_a_sensor_gap_waits_for_revalidation():
    runtime = NavigationRuntime(lambda request, cancelled: 'forward', lambda command: None)
    try:
        runtime.cycle(view(1.), 1.)
        timeout = time.time() + 1.
        while runtime._result is None and time.time() < timeout:
            time.sleep(.001)
        assert runtime._result is not None
        command, _, decision = runtime.cycle(None, 1.05)
        assert command.linear == 0. and decision is None
        command, request, decision = runtime.cycle(view(1.10), 1.10)
        assert command.linear > 0. and request is None and decision[2]
    finally:
        assert runtime.close()


def test_room_intent_reuses_one_model_choice_for_locally_checked_successors():
    calls = []

    def choose(request, cancelled):
        calls.append(request.sequence)
        return 'circulate_left'

    navigation = Navigation(successor_lead=.15)
    runtime = NavigationRuntime(choose, lambda command: None, navigation, static_room=True)
    try:
        runtime.cycle(view(1.), 1.)
        timeout = time.time() + 1.
        while runtime._result is None and time.time() < timeout:
            time.sleep(.001)
        command, _, decision = runtime.cycle(view(1.05), 1.05)
        assert command.action == 'left' and decision == (1, 'circulate_left', True)
        on_arc = arc_pose(Pose(0., 0., 0.), .10, .8)
        _, request, decision = runtime.cycle(view(6.5, on_arc), 6.5)
        assert request is None and decision is None
        assert runtime.room_continuation == (2, 'left', 1)
        assert navigation.successor.action == 'left' and calls == [1]
        lost, _, _ = runtime.cycle(None, 6.55)
        assert lost.linear == 0. and lost.angular == 0.
    finally:
        assert runtime.close()


@pytest.mark.parametrize('markings,stop_reason', [({(-.13, -.119): -1}, 'path_unknown'),
                                                   ({(.22, .13): 100, (.22, -.13): 100}, 'path_occupied')])
def test_room_reobserves_straight_when_a_turn_is_blocked(markings, stop_reason):
    navigation = Navigation(successor_lead=.15)
    runtime = NavigationRuntime(lambda request, cancelled: 'circulate_left', lambda command: None,
                                navigation, static_room=True)
    try:
        runtime.cycle(view(1.), 1.)
        timeout = time.time() + 1.
        while runtime._result is None and time.time() < timeout:
            time.sleep(.001)
        assert runtime.cycle(view(1.05), 1.05)[0].action == 'left'
        cells = grid(markings)
        assert runtime.cycle(view(1.10, cells=cells), 1.10)[0].reason == stop_reason
        assert runtime.room_continuation == (2, 'forward', 1)
        command, _, _ = runtime.cycle(view(1.15, cells=cells), 1.15)
        assert command.linear > 0. and command.angular == 0.
    finally:
        assert runtime.close()


def test_room_can_use_an_observed_away_turn_before_driving_toward_an_obstacle():
    navigation = Navigation(successor_lead=.15)
    runtime = NavigationRuntime(lambda request, cancelled: 'circulate_left', lambda command: None,
                                navigation, static_room=True)
    try:
        runtime.cycle(view(1.), 1.)
        timeout = time.time() + 1.
        while runtime._result is None and time.time() < timeout:
            time.sleep(.001)
        assert runtime.cycle(view(1.05), 1.05)[0].action == 'left'
        cells = grid({(.22, .13): 100})
        assert runtime.cycle(view(1.10, cells=cells), 1.10)[0].reason == 'path_occupied'
        assert runtime.room_continuation == (2, 'right', 1)
    finally:
        assert runtime.close()


def test_room_turns_around_an_obstacle_beyond_the_next_valid_segment():
    navigation = Navigation(successor_lead=.15)
    runtime = NavigationRuntime(lambda request, cancelled: 'circulate_left', lambda command: None,
                                navigation, static_room=True)
    try:
        cells = grid({(.52, .20): 100, (.62, -.35): -1})
        runtime.cycle(view(1., cells=cells), 1.)
        timeout = time.time() + 1.
        while runtime._result is None and time.time() < timeout:
            time.sleep(.001)
        command, _, decision = runtime.cycle(view(1.05, cells=cells), 1.05)
        assert decision == (1, 'circulate_left', True)
        assert command.action == 'right' and command.angular < 0.
    finally:
        assert runtime.close()


def test_room_can_prepare_a_checked_pivot_at_a_close_obstacle():
    navigation = Navigation(static_room=True)
    seen = view(1., cells=grid({(.47, 0.): 100}))
    assert navigation.prepare_pivot(seen, 1., 'left') == 'pivot_left'
    command = navigation.step(seen, 1.05)
    assert command.linear == 0. and command.angular > 0.
    stopped = navigation.step(None, 1.10)
    assert stopped.linear == stopped.angular == 0.

    blind = Navigation(static_room=True)
    assert blind.prepare_pivot(view(1., cells=grid({(.47, 0.): 100, (-.18, -.06): -1})), 1., 'left') is None


def test_runtime_tracks_for_thirty_seconds_through_multiple_delayed_model_choices():
    release = threading.Event()
    choices = iter(('forward', 'left', 'right', 'forward'))
    commands = []

    def decide(request, cancelled):
        while not release.wait(.005):
            if cancelled():
                return None
        return next(choices)

    def finish_model():
        release.set()
        timeout = time.time() + 1.
        while runtime._result is None and time.time() < timeout:
            time.sleep(.001)
        assert runtime._result is not None
        release.clear()

    runtime = NavigationRuntime(decide, commands.append)
    cells = grid()
    pose = Pose(0., 0., 0.)
    pending_until = None
    accepted, actions = [], set()
    try:
        _, request, _ = runtime.cycle(view(1., pose, cells), 1.)
        assert request is not None
        finish_model()
        for index in range(1, 621):
            now = 1. + index * .05
            if pending_until is not None and now >= pending_until:
                finish_model()
                pending_until = None
            command, request, decision = runtime.cycle(view(now, pose, cells), now)
            assert command.linear > 0., (now, command)
            actions.add(command.action)
            if request:
                pending_until = now + 2.8
            if decision:
                assert decision[2]
                accepted.append(decision[1])
            pose = arc_pose(pose, command.linear * .05, command.angular / command.linear)

        assert accepted[:3] == ['forward', 'left', 'right']
        assert {'forward', 'left', 'right'} <= actions
        assert pose.x > .56
        stopped, _, _ = runtime.cycle(view(now + .05, pose, cells, scan=now - .5), now + .05)
        assert stopped.reason == 'perception_unavailable' and stopped.linear == stopped.angular == 0.
    finally:
        assert runtime.close()


def test_ros_grid_keeps_sensor_source_times_and_accepts_independent_transport_sequences(monkeypatch):
    import explore
    from run_navigation import RosNavigationInput

    def message(**fields):
        return type('Message', (object,), fields)()

    def stamp(value):
        return message(to_sec=lambda self: value)

    source = RosNavigationInput.__new__(RosNavigationInput)
    source.latest_grid = source.last_grid_times = None
    source.grid_updates = collections.Counter()
    source.rospy = message(Time=message(now=lambda self: stamp(101.)))
    monkeypatch.setattr(explore, 'sample_clocks', lambda *_: (101., 11.))
    orientation = message(x=0., y=0., z=0., w=1.)
    origin = message(orientation=orientation, position=message(x=0., y=0.))
    header_time = stamp(100.98)
    header = message(frame_id='odom', stamp=header_time, seq=50)
    grid_header = message(frame_id='odom', stamp=header_time, seq=0)
    info = message(origin=origin, width=2, height=2, resolution=.05)
    local = message(header=grid_header, info=info, data=(-1, 0, 100, 0))
    packet = message(grid=local, header=header, scan_fresh=True, depth_fresh=True,
                     scan_stamp=stamp(100.85), depth_stamp=stamp(100.92))
    source._grid(packet)
    times, accepted = source.latest_grid
    assert times == pytest.approx((10.98, 10.85, 10.92))
    assert accepted.cells == (-1, 0, 100, 0)
    assert source.grid_updates['ready'] == 1
    packet.depth_fresh = False
    source._grid(packet)
    assert source.latest_grid is None
    packet.depth_fresh, packet.depth_stamp = True, stamp(100.90)
    source._grid(packet)
    assert source.latest_grid is None and source.grid_updates['grid_regression'] == 1
