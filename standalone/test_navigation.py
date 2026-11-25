from __future__ import division

import collections
import json
import math
import threading
import time

import pytest

from navigation import Command, Grid, Navigation, Pose, View, arc_pose, sweep_clear
from navigation_runtime import MappedChoice, NavigationRuntime


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


def test_room_starts_at_effective_crawl_only_after_a_current_clear_sweep():
    navigation = Navigation(static_room=True)
    select(navigation, 'forward', 1., view(1.))
    assert navigation.step(view(1.05), 1.05).linear == pytest.approx(.014)
    assert navigation.step(view(1.10, scan=.65), 1.10).linear == 0.
    assert navigation.step(view(1.15, cells=grid({(.22, 0.): 100})), 1.15).linear == 0.


def test_free_roam_checks_a_shorter_segment_with_braking_space_when_full_starts_are_blocked():
    wall = grid({(.255, i * .04): 100 for i in range(-7, 8)})
    observation = view(1., cells=wall)
    navigation = Navigation(static_room=True, short_room_segments=True)
    assert Navigation(static_room=True).request(observation, 1.) is None
    request = select(navigation, 'forward', 1., observation)
    distance = dict(request.short_segments)['forward']
    assert .03 < distance < navigation.length
    assert navigation.step(view(1.05, cells=wall), 1.05).linear > 0.
    assert navigation.step(view(1.10, Pose(distance - .005, 0., 0.), wall), 1.10).reason == 'goal_reached'

    tight_wall = grid({(.215, i * .04): 100 for i in range(-7, 8)})
    assert Navigation(static_room=True, short_room_segments=True).request(view(2., cells=tight_wall), 2.) is None


def test_decision_evidence_reports_the_checked_short_start_distance():
    from navigation import Request
    from run_navigation import navigation_laser_evidence

    request = Request(1, 1., 0, Pose(0., 0., 0.), ('forward', 'stop'), None, (),
                      short_segments=(('forward', .05),))
    evidence = navigation_laser_evidence(request, static_room=True)
    assert evidence['checked_start_distances_m'] == {'forward': .05}
    assert 'locally_checked_segment_length_m' not in evidence


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


def test_a_late_control_sample_does_not_create_a_second_gap():
    from run_navigation import control_sample_delay

    assert control_sample_delay(0., .061) == 0.
    assert control_sample_delay(.061, .081) == pytest.approx(.030)


def test_navigation_trajectory_counts_motion_after_a_pause():
    from run_navigation import summarize_navigation_trajectory
    commands = [(1., .014, 0.), (1.05, .014, 0.), (1.1, 0., 0.),
                (1.2, .014, 0.), (1.25, .014, 0.), (1.30, .014, 0.), (1.35, 0., 0.)]
    odometry = [(1., 0., 0., 0.), (1.15, .01, 0., 0.), (1.35, .02, 0., 0.)]
    result = summarize_navigation_trajectory({'commands': commands, 'odom': odometry})
    assert result['nonzero_cmd_vel_messages'] == 5
    assert result['longest_nonzero_sec'] == pytest.approx(.15)
    assert result['odom_displacement_m'] == .02 and result['zero_after_last_motion'] == 1.35


def test_navigation_distinguishes_wheel_pauses_and_missing_odometry_from_relay_output():
    from run_navigation import summarize_navigation_trajectory
    commands = [(i * .05, .02, 0.) for i in range(40)] + [(2., 0., 0.)]
    odometry = [(i * .05, i * .001, 0., .02 if i < 11 or i >= 18 else 0., 0., 0.)
                for i in range(40) if not 26 <= i <= 31]
    observed = summarize_navigation_trajectory({'commands': commands, 'odom': odometry})
    assert observed['longest_nonzero_sec'] == 2.
    assert observed['wheel_motion']['pause_count'] == 1
    assert observed['wheel_motion']['stationary_sec'] == pytest.approx(.35)
    assert observed['wheel_motion']['unobserved_gaps'] == 1
    assert observed['wheel_motion']['longest_sec'] == pytest.approx(.55)


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
        if stop_reason == 'path_unknown':
            assert runtime.room_continuation is None
            assert runtime.cycle(view(1.15, cells=cells), 1.15)[0].action == 'forward'
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


@pytest.mark.parametrize('front_cell', [0, -1, 100])
def test_room_reanchors_a_paused_forward_segment_only_if_an_alternative_is_observed(front_cell):
    side, resolution = 60, .05
    cells = [0] * (side * side)
    cells[30 * side + 30] = -1
    cells[33 * side + (30 if front_cell == -1 else 29)] = front_cell
    observed = Grid(-1.305, -1.659, resolution, side, side, tuple(cells))
    navigation = Navigation(static_room=True)
    select(navigation, 'forward', 1., view(1.))
    runtime = NavigationRuntime(lambda request, cancelled: None, lambda command: None,
                                navigation, static_room=True)
    runtime.room_plan = (1, 'left')
    try:
        command, _, _ = runtime.cycle(view(1.05, cells=observed), 1.05)
        if front_cell:
            assert command.linear == command.angular == 0.
        else:
            assert command.action == 'left' and command.linear > 0. and command.angular > 0.
            assert sweep_clear(observed, Pose(0., 0., 0.), .10, command.angular / command.linear)
            assert runtime.room_replacement
            assert runtime.room_plan == (1, 'left')
    finally:
        assert runtime.close()


def test_room_rechecks_failed_alternatives_after_geometry_changes(monkeypatch):
    navigation = Navigation(static_room=True)
    select(navigation, 'forward', 1., view(1.))
    runtime = NavigationRuntime(lambda request, cancelled: None, lambda command: None,
                                navigation, static_room=True)
    runtime.room_plan = (1, 'left')
    checked = []
    replace = navigation.replace_unknown_path

    def count_checks(observation, now, direction, *args):
        checked.append(now)
        return replace(observation, now, direction, *args)

    monkeypatch.setattr(navigation, 'replace_unknown_path', count_checks)
    try:
        for now in (1.05, 1.10, 1.15):
            command, _, _ = runtime.cycle(view(now, cells=grid({(.21, 0.): -1})), now)
            assert command.reason == 'path_unknown'
        runtime.cycle(view(1.20, cells=grid({(.21, 0.): -1, (.51, 0.): -1})), 1.20)
        assert checked == [1.05, 1.20]
    finally:
        assert runtime.close()


def test_unknown_detour_keeps_speed_with_a_verified_slewed_transition():
    navigation = Navigation(static_room=True)
    select(navigation, 'left', 1., view(1.))
    navigation.speed, navigation.steer = .02, .40
    sent = []
    runtime = NavigationRuntime(None, sent.append, navigation, static_room=True)
    runtime.room_plan = (1, 'left')
    try:
        cells = grid({(.23, .12): -1})
        command, _, _ = runtime.cycle(view(1.05, cells=cells), 1.05)
        assert sent == [command]
        assert command.action == 'forward' and command.reason == 'unknown_detour'
        assert command.linear == .02
        assert command.angular / command.linear == pytest.approx(.15)
        assert sweep_clear(cells, Pose(0., 0., 0.), .10, command.angular / command.linear)
    finally:
        assert runtime.close()


def test_room_prefers_the_fully_observed_exit_when_the_requested_arc_loses_coverage_early():
    navigation = Navigation(static_room=True)
    runtime = NavigationRuntime(None, lambda command: None, navigation, static_room=True)
    try:
        observation = view(1., cells=grid({(.35, .16): -1}))
        request = navigation.request(observation, 1.)
        assert {'forward', 'left'}.issubset(request.choices)
        assert runtime._room_action(request, 'left', observation) == 'forward'
    finally:
        assert runtime.close()


def test_room_waits_for_sustained_fresh_geometry_before_resuming_the_same_path():
    navigation = Navigation(static_room=True)
    select(navigation, 'forward', 1., view(1.))
    obscured = grid({(.21, 0.): -1})
    assert navigation.step(view(1.05, cells=obscured), 1.05).reason == 'path_unknown'
    clear_frame = view(1.10)
    for now in (1.10, 1.25):
        command = navigation.step(clear_frame, now)
        assert command.reason == 'path_reacquiring' and command.linear == command.angular == 0.
    assert navigation.step(view(1.30, cells=obscured), 1.30).reason == 'path_unknown'
    for now in (1.35, 1.48, 1.60, 1.70):
        command = navigation.step(view(now), now)
        assert command.reason == 'path_reacquiring' and command.linear == command.angular == 0.
    assert navigation.step(view(1.78), 1.78).linear > 0.
    blocked = navigation.step(view(1.82, cells=grid({(.21, 0.): 100})), 1.82)
    assert blocked.reason == 'path_occupied' and blocked.linear == blocked.angular == 0.


def test_room_offers_safe_pivots_to_the_model_when_forward_paths_end_at_a_box():
    navigation = Navigation(static_room=True)
    cells = grid({(.25, .015): 100})
    request = navigation.request(view(1., cells=cells), 1.)
    assert request.choices == ('pivot_left', 'pivot_right', 'stop')
    assert {action: reason for action, reason, _, _ in request.blocks} == {
        'forward': 'occupied', 'left': 'occupied', 'right': 'occupied'}
    new_box = view(1.05, cells=grid({(.25, .015): 100, (.13, .13): 100}))
    assert not navigation.accept(request, 'pivot_left', new_box, 1.05)

    runtime = NavigationRuntime(lambda request, cancelled: 'circulate_left', lambda command: None,
                                navigation, static_room=True)
    try:
        _, request, _ = runtime.cycle(view(2., cells=cells), 2.)
        assert request.choices[0] == 'pivot_left'
        until = time.time() + 1.
        while runtime._result is None and time.time() < until:
            time.sleep(.001)
        command, _, decision = runtime.cycle(view(2.05, cells=cells), 2.05)
        assert decision == (request.sequence, 'circulate_left', True)
        assert command.linear == 0. and command.angular > 0.
    finally:
        assert runtime.close()


def test_room_keeps_the_model_route_when_tracking_needs_a_new_anchor():
    calls = []

    def choose(request, cancelled):
        calls.append(request.sequence)
        return 'circulate_left'

    nav = Navigation(static_room=True)
    runtime = NavigationRuntime(choose, lambda command: None, nav, static_room=True)
    try:
        runtime.cycle(view(1.), 1.)
        timeout = time.time() + 1.
        while runtime._result is None and time.time() < timeout:
            time.sleep(.001)
        assert runtime.cycle(view(1.05), 1.05)[0].action == 'left'
        observation = view(1.10, Pose(.03, .06, .08))
        command, request, _ = runtime.cycle(observation, 1.10)
        assert command.reason == 'path_tracking_lost' and command.linear == 0.
        assert request is None and runtime.room_continuation == (2, 'left', 1)
        assert runtime.cycle(view(1.15, observation.pose), 1.15)[0].linear > 0.
        assert calls == [1]
    finally:
        assert runtime.close()


def test_room_looks_into_a_checked_successor_before_the_waypoint():
    nav = Navigation(static_room=True)
    select(nav, 'forward', 1., view(1.))
    assert nav.step(view(1.05), 1.05).linear > 0.
    before_join = view(7., Pose(.105, .003, 0.))
    select(nav, 'forward', 7., before_join)
    nav.speed = .02
    cmd = nav.step(before_join, 7.)
    assert cmd.linear == pytest.approx(.02)
    assert -.03 < cmd.angular < 0.


def test_room_slew_limiting_never_holds_an_occupied_sweep():
    nav = Navigation(static_room=True)
    select(nav, 'left', 1., view(1.))
    nav.steer, nav.speed = -2., .02
    # Gradual steering still swings the rear toward this cell. The new left
    # curve has a clear swept area and can be used immediately.
    cmd = nav.step(view(1.05, cells=grid({(-.10, .14): 100})), 1.05)
    assert cmd.linear > 0. and cmd.angular > 0.
    stopped = nav.step(view(1.10, cells=grid({(.22, 0.): 100})), 1.10)
    assert stopped.reason == 'path_occupied' and stopped.linear == stopped.angular == 0.


def test_room_can_replace_an_occupied_arc_without_publishing_an_intermediate_stop():
    navigation = Navigation(static_room=True)
    select(navigation, 'left', 1., view(1.))
    navigation.speed, navigation.steer = .02, navigation.curvature['left']
    sent = []
    runtime = NavigationRuntime(lambda request, cancelled: None, sent.append, navigation, static_room=True)
    runtime.room_plan = (1, 'left')
    pose = arc_pose(Pose(0., 0., 0.), .015, navigation.curvature['left'])
    try:
        command, offered, decision = runtime.cycle(view(1.05, pose, grid({(.23, .12): 100})), 1.05)
        assert sent == [command]
        assert command == Command(.02, 0., 'forward', 'occupied_detour')
        assert navigation.goal.origin == pose and runtime.room_plan == (1, 'left')
        assert runtime.occupied_detour[:2] == ('left', 'forward')
        assert offered is decision is None

        stopped, _, _ = runtime.cycle(view(1.10, pose, grid({(.23, 0.): 100})), 1.10)
        assert stopped == Command(0., 0., 'stop', 'path_occupied')
    finally:
        assert runtime.close()


def test_room_pivot_hands_off_to_checked_forward_without_a_zero_setpoint():
    nav = Navigation(static_room=True)
    cells = grid({(.47, 0.): 100})
    assert nav.prepare_pivot(view(1., cells=cells), 1., 'left') == 'pivot_left'
    near_end = view(10., Pose(0., 0., .62), cells)
    assert nav.step(near_end, 10.).angular > 0.
    exit_request = nav.request(near_end, 10.)
    assert exit_request.choices == ('forward', 'stop')
    assert nav.accept(exit_request, 'forward', near_end, 10.)
    after_pivot = view(10.05, Pose(0., 0., .695), cells)
    cmd = nav.step(after_pivot, 10.05)
    assert cmd.action == 'forward' and cmd.linear > 0.


def test_room_stationary_pivot_can_find_an_exit_beyond_the_fixed_turn():
    side = 120
    coordinates = [-3. + (index + .5) * .05 for index in range(side)]
    cells = tuple(100 if (abs(x) >= 1.92 or abs(y) >= 1.60 or -1.60 <= x < -1.45 and .70 <= y < .85)
                  else 0 for y in coordinates for x in coordinates)
    corner = Grid(-3., -3., .05, side, side, cells)
    pose = Pose(-1.6477, 1.3001, 2.8506)
    observation = view(1., pose, corner)
    nav = Navigation(static_room=True)
    assert nav.request(observation, 1.) is None
    runtime = NavigationRuntime(lambda request, cancelled: None, lambda command: None, nav, static_room=True)
    runtime.room_plan = (1, 'left')
    try:
        for at in (1., 1.05, 1.10, 1.15, 1.20):
            assert runtime.cycle(view(at, pose, corner), at)[0].action == 'stop'
            assert nav.goal is None
        runtime.cycle(view(1.25, pose, corner), 1.25)
        assert runtime.room_continuation[1] == 'pivot_right'
    finally:
        assert runtime.close()
    assert nav.goal.turn == pytest.approx(-2.45)
    assert nav.step(view(1.30, pose, corner), 1.30).angular < 0.
    turned = pose._replace(yaw=pose.yaw + nav.goal.turn)
    exit_view = view(50., turned._replace(yaw=turned.yaw + .06), corner)
    assert nav.step(exit_view, 50.).angular < 0.
    select(nav, 'forward', 50., exit_view)
    assert nav.step(view(50.05, turned, corner), 50.05).linear > 0.


def test_room_prefers_a_checked_detour_arc_to_a_pivot():
    nav = Navigation(static_room=True)
    observation = view(1., cells=grid({(.35, .08): 100}))
    offered = nav.request(observation, 1.)
    assert 'right' in offered.choices
    assert nav.prepare_pivot(observation, 1., 'left', offered) is None
    assert nav.pending is offered


@pytest.mark.parametrize('scenario', ['clear_detour', 'blocked_detour', 'unknown_detour', 'known_successor',
                                     'near_obstacle', 'stale', 'short_deadline'])
def test_room_preempts_a_new_obstacle_before_the_immediate_sweep_is_blocked(scenario):
    navigation = Navigation(static_room=True)
    select(navigation, 'forward', 1., view(1.))
    pose = Pose(.06, 0., 0.)
    navigation.step(view(1.05, pose), 1.05)
    successor = 'left' if scenario == 'known_successor' else 'forward'
    select(navigation, successor, 1.05, view(1.05, pose))
    obstacle = (.23, 0.) if scenario == 'near_obstacle' else (
        (.41, 0.) if scenario == 'blocked_detour' else (.45, -.095))
    cells = {obstacle: 100}
    if scenario == 'unknown_detour':
        cells[.17, .10] = -1
    measured = view(1.10, pose, cells=grid(cells),
                    scan={'stale': .75, 'short_deadline': .83}.get(scenario))
    runtime = NavigationRuntime(lambda request, cancelled: None, lambda command: None,
                                navigation, static_room=True)
    runtime.room_plan = (1, 'left')
    try:
        command, _, _ = runtime.cycle(measured, 1.10)
        if scenario == 'clear_detour':
            assert command.action == 'left' and command.linear > 0.
            assert runtime.early_detour[0] == 'forward'
            assert runtime.room_plan == (1, 'left')
            following, _, _ = runtime.cycle(view(1.15, pose, cells=measured.grid), 1.15)
            assert following.linear > 0. and following.angular > 0.
        else:
            assert runtime.early_detour is None
            if scenario in ('near_obstacle', 'stale'):
                assert command.linear == command.angular == 0.
            else:
                assert command.action == 'forward' and command.linear > 0.
                assert navigation.successor.action == successor
    finally:
        assert runtime.close()


@pytest.mark.parametrize('elapsed, marks, expected', [(2.7, {}, 'right'), (21., {}, 'left'),
                                                     (2.7, {(.51, -.22): 100}, 'left')])
def test_room_keeps_a_checked_detour_through_a_brief_loss_of_the_obstacle(elapsed, marks, expected):
    navigation = Navigation(static_room=True)
    select(navigation, 'forward', 1., view(1.))
    pose = Pose(.06, 0., 0.)
    navigation.step(view(1.05, pose), 1.05)
    select(navigation, 'forward', 1.05, view(1.05, pose))
    runtime = NavigationRuntime(lambda request, cancelled: None, lambda command: None,
                                navigation, static_room=True)
    runtime.room_plan = (1, 'left')
    try:
        command, _, _ = runtime.cycle(view(1.10, pose, cells=grid({(.45, .095): 100})), 1.10)
        assert command.action == 'right' and command.linear > 0.
        later = 1.10 + elapsed
        on_detour = arc_pose(pose, .06, -1.35)
        command, _, _ = runtime.cycle(view(later, on_detour, grid(marks)), later)
        assert command.linear > 0.
        assert (navigation.successor or navigation.goal).action == expected
        assert runtime.detour_continuation == (expected == 'right')
        assert runtime.room_plan == (1, 'left')
    finally:
        assert runtime.close()


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


def test_room_waits_for_a_new_decision_after_a_stop_received_with_stale_perception():
    navigation = Navigation(static_room=True)
    sensors = view(1.)
    request = select(navigation, 'forward', 1., sensors)
    runtime = NavigationRuntime(lambda request, cancelled: None, lambda command: None,
                                navigation, static_room=True)
    try:
        runtime.room_plan = request.sequence, 'left'
        runtime.review_origin = request.sequence
        runtime._result = request, 'stop'

        command, _, decision = runtime.cycle(view(1.10, scan=.60), 1.10)
        assert command.linear == command.angular == 0.
        assert decision == (request.sequence, 'stop', True)
        assert runtime.room_plan is None and navigation.goal is None

        command, new_request, _ = runtime.cycle(view(1.15), 1.15)
        assert command.linear == command.angular == 0. and new_request is None

        command, new_request, _ = runtime.cycle(view(2.15), 2.15)
        assert command.linear == command.angular == 0. and new_request is not None
    finally:
        assert runtime.close()


def test_room_rechecks_route_with_the_model_without_waiting_at_local_segments(monkeypatch):
    import navigation_runtime
    monkeypatch.setattr(navigation_runtime, 'ROOM_ROUTE_REVIEW_SEC', 5.)
    release = threading.Event()
    requests = []

    def decide(request, cancelled):
        requests.append(request)
        if len(requests) == 1:
            return 'circulate_left'
        while not release.wait(.005):
            if cancelled():
                return None
        return 'circulate_right'

    def await_decision():
        until = time.time() + 1.
        while runtime._result is None and time.time() < until:
            time.sleep(.001)
        assert runtime._result is not None

    runtime = NavigationRuntime(decide, lambda command: None, Navigation(static_room=True), static_room=True)
    pose, cells = Pose(0., 0., 0.), grid()
    answer_at = None
    actions, accepted = set(), []
    try:
        runtime.cycle(view(1., pose, cells), 1.)
        await_decision()
        for index in range(1, 520):
            now = 1. + index * .05
            if answer_at is not None and now >= answer_at and not release.is_set():
                release.set()
                await_decision()
            command, request, decision = runtime.cycle(view(now, pose, cells), now)
            assert command.linear > 0., (now, command)
            actions.add(command.action)
            if request and answer_at is None:
                answer_at = now + 5.5
            if decision:
                assert decision[2]
                accepted.append(decision[1])
            pose = arc_pose(pose, command.linear * .05, command.angular / command.linear)
        assert accepted[:2] == ['circulate_left', 'circulate_right']
        assert actions == {'left', 'right'}
    finally:
        assert runtime.close()


def test_room_reuses_a_fresh_direction_when_the_offered_arc_becomes_blocked():
    navigation = Navigation(static_room=True)
    runtime = NavigationRuntime(lambda request, cancelled: 'circulate_left', lambda command: None,
                                navigation, static_room=True)
    try:
        _, first, _ = runtime.cycle(view(1.), 1.)
        assert 'left' in first.choices
        until = time.time() + 1.
        while runtime._result is None and time.time() < until:
            time.sleep(.001)
        blocked = grid({(.25, .015): 100})
        command, request, decision = runtime.cycle(view(1.05, cells=blocked), 1.05)
        assert command.linear == command.angular == 0.
        assert decision == (first.sequence, 'circulate_left', True)
        assert request is None and runtime.room_continuation[1:] == ('pivot_left', first.sequence)
        command, _, _ = runtime.cycle(view(1.10, cells=blocked), 1.10)
        assert command.action == 'pivot_left' and command.angular > 0.
    finally:
        assert runtime.close()


def test_mapped_choice_keeps_target_and_cannot_substitute_a_blocked_model_action():
    for cells in (grid(), grid({(.25, .015): 100})):
        sent = []
        runtime = NavigationRuntime(lambda request, cancelled: MappedChoice('left', '%s-left' % request.sequence),
                                    sent.append, Navigation(static_room=True), static_room=True)
        try:
            _, request, _ = runtime.cycle(view(1.), 1.)
            assert 'left' in request.choices
            until = time.time() + 2.
            while runtime._result is None and time.time() < until:
                time.sleep(.001)
            assert runtime._result is not None
            command, _, decision = runtime.cycle(view(1.05, cells=cells), 1.05)
            assert runtime.selected_target_id == '%s-left' % request.sequence
            accepted = not any(cell > 0 for cell in cells.cells)
            assert decision == (request.sequence, 'left', accepted)
            assert len(sent) == 2 and sent[-1] == command
            if accepted:
                assert command.action == 'left' and command.linear > 0.
            else:
                assert command.linear == command.angular == 0.
                assert runtime.room_plan is None
        finally:
            assert runtime.close()


def test_unknown_during_a_pivot_allows_a_replacement_only_after_fresh_settled_odometry(monkeypatch):
    import navigation as tracker

    observe = view(1.)
    navigation = Navigation(static_room=True)
    navigation.epoch = 0
    navigation.goal = tracker.Segment(1, 'pivot_left', observe.pose, .70)
    runtime = NavigationRuntime(None, lambda command: None, navigation, static_room=True)
    runtime.room_plan = (1, 'left')
    try:
        turning, _, _ = runtime.cycle(observe, 1.)
        assert turning.linear == 0. and turning.angular > 0.
        monkeypatch.setattr(tracker, 'rotation_block', lambda *args: ('unknown', .15, .10))
        for at in (1.05, 1.10, 1.20, 1.29):
            stopped, _, _ = runtime.cycle(view(at), at)
            assert stopped.linear == stopped.angular == 0. and stopped.reason == 'path_unknown'
        command, _, _ = runtime.cycle(view(1.31), 1.31)
        assert command.linear > 0. and command.reason == 'unknown_detour'
    finally:
        assert runtime.close()


def test_free_roam_replans_a_settled_blocked_pivot_with_the_shorter_safe_options(monkeypatch):
    import navigation as tracker

    wall = grid({(.255, i * .04): 100 for i in range(-7, 8)})
    observation = view(1., cells=wall)
    navigation = Navigation(static_room=True, short_room_segments=True)
    navigation.epoch = 0
    navigation.goal = tracker.Segment(1, 'pivot_left', observation.pose, .25)
    runtime = NavigationRuntime(lambda request, cancelled: None, lambda command: None,
                                navigation, static_room=True)
    runtime.room_plan = (1, 'left')
    monkeypatch.setattr(tracker, 'rotation_block', lambda *args: ('unknown', .15, .10))
    try:
        for at in (1., 1.1, 1.2):
            command, request, _ = runtime.cycle(view(at, cells=wall), at)
            assert command.reason == 'path_unknown' and request is None
        command, request, _ = runtime.cycle(view(1.26, cells=wall), 1.26)
        assert command.reason == 'pivot_replan' and command.linear == command.angular == 0.
        assert request is not None and dict(request.short_segments)['forward'] > 0.
        assert navigation.goal is None
    finally:
        assert runtime.close()


def test_initial_mapped_forward_can_continue_with_locally_checked_successors():
    runtime = NavigationRuntime(lambda request, cancelled: MappedChoice('forward', '%d-forward' % request.sequence),
                                lambda command: None, Navigation(static_room=True), static_room=True)
    try:
        _, request, _ = runtime.cycle(view(1.), 1.)
        until = time.time() + 2.
        while runtime._result is None and time.time() < until:
            time.sleep(.001)
        assert runtime._result is not None
        command, _, decision = runtime.cycle(view(1.05), 1.05)
        assert (command.action, decision[2], runtime.selected_target_stage) == ('forward', True, 'mapped_target')
        assert runtime.room_plan[0] == request.sequence
    finally:
        assert runtime.close()


def test_a_room_decision_reads_a_single_bounded_color_frame():
    from run_navigation import RosNavigationInput

    frame, message_type = object(), object()
    calls = []

    class ROS(object):
        def wait_for_message(self, topic, kind, timeout):
            calls.append((topic, kind, timeout))
            return frame

    source = RosNavigationInput.__new__(RosNavigationInput)
    source.rospy, source.color_type = ROS(), message_type
    assert source.decision_image(lambda: False) is frame
    assert source.decision_image(lambda: True) is None
    assert calls == [('/camera/color/image_raw', message_type, .22)]


def test_a_mismatched_old_room_view_leaves_current_local_perception_ready():
    from run_navigation import RosNavigationInput

    class Unregistered(object):
        attempts = 0

        def registered_cells(self, view, scan, now, ros_now):
            self.attempts += 1
            return []

    source = RosNavigationInput.__new__(RosNavigationInput)
    source.revisit, source.revisit_scan = Unregistered(), None
    source.latest_scan = object()
    source.rospy = type('Ros', (), {'Time': type('Time', (), {
        'now': staticmethod(lambda: type('Stamp', (), {'to_sec': lambda self: 100.})()),
    })})()
    local = view(1.)
    assert source.register_revisit(None, local, 1.) is local
    assert source.register_revisit(None, local, 1.01) is local
    assert source.revisit.attempts == 1
    source.latest_scan = object()
    assert source.register_revisit(None, local, 1.02) is local
    assert source.revisit.attempts == 2


def test_ros_grid_keeps_sensor_source_times_and_accepts_independent_transport_sequences(monkeypatch):
    import explore
    from run_navigation import RosNavigationInput

    def message(**fields):
        return type('Message', (object,), fields)()

    def stamp(value):
        return message(to_sec=lambda self: value, to_nsec=lambda self: int(round(value * 1e9)))

    source = RosNavigationInput.__new__(RosNavigationInput)
    source.latest_grid = source.last_grid_times = source.grid_changed = None
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
                     scan_stamp=stamp(100.85), depth_stamp=stamp(100.92),
                     depth_hits=(message(index=2, min_z=.01, max_z=.04),))
    source._grid(packet)
    times, accepted, depth = source.latest_grid
    assert times == pytest.approx((10.98, 10.85, 10.92))
    assert accepted.cells == (-1, 0, 100, 0)
    assert depth.sequence == 100920000000 and depth.frame == 'odom' and depth.hits == {2: (.01, .04)}
    assert source.grid_updates['ready'] == 1
    packet.depth_fresh = False
    source._grid(packet)
    assert source.latest_grid is None
    packet.depth_fresh, packet.depth_stamp = True, stamp(100.90)
    source._grid(packet)
    assert source.latest_grid is None and source.grid_updates['grid_regression'] == 1


def test_room_model_can_select_the_exact_checked_shallow_pivot(monkeypatch):
    tracker = Navigation(static_room=True)
    monkeypatch.setattr(tracker, '_pivot_clear', lambda local, anchor, action, turn=None:
                        action == 'pivot_left' and turn == .25)
    local = view(1., cells=grid({(.25, .015): 100}))
    request = tracker.request(local, 1.)
    assert request.choices == ('pivot_left', 'stop')
    assert dict(request.pivot_turns) == {'pivot_left': .25}
    assert tracker.accept(request, 'pivot_left', local, 1.)
    assert tracker.goal.turn == .25


def test_room_offers_a_checked_pivot_over_arcs_that_lead_to_a_measured_obstacle(monkeypatch):
    tracker = Navigation(static_room=True)
    monkeypatch.setattr(tracker, '_pivot_clear', lambda local, anchor, action, turn=None:
                        action == 'pivot_left' and turn == .25)
    local = view(1., cells=grid({(.40, .015): 100}))
    request = tracker.request(local, 1.)
    assert request.choices == ('pivot_left', 'stop')
    assert request.pivot_turns == (('pivot_left', .25),)


def test_room_observation_pivot_rechecks_only_the_rotation_and_does_not_repeat_in_place(monkeypatch):
    import navigation as tracker

    nav = Navigation(static_room=True)
    monkeypatch.setattr(nav, '_pivot_clear', lambda *args: False)
    monkeypatch.setattr(tracker, 'rotation_block',
                        lambda cells, origin, angle, body=None: None if angle == .25 else ('unknown', 0., 0.))
    local = view(1., cells=grid({(.35, -.05): 100, (.30, .30): -1}))
    request = nav.request(local, 1.)
    assert request.observation_turns == ('pivot_left',)
    assert request.choices == ('pivot_left', 'stop')
    assert nav.accept(request, 'pivot_left', local, 1.)
    assert nav.goal.turn == .25
    nav.stop('ended')
    again = nav.request(view(1.1, cells=local.grid), 1.1)
    assert again is None or not again.observation_turns


def test_pivot_keeps_the_measured_body_at_its_start_when_one_cell_leaves_view():
    import navigation as tracker

    cells = [0] * 3600
    cells[29 * 60 + 35] = -1
    grid_view = Grid(-1.5, -4., .05, 60, 60, tuple(cells))
    initial = Pose(.40858, -2.55591, .28641)
    angled = initial._replace(yaw=.37490)
    nav = Navigation(static_room=True)
    nav.goal = tracker.Segment(1, 'pivot_left', initial, .25)
    nav.epoch = 0
    assert nav.step(view(1., initial, grid_view), 1.).angular > 0.
    assert tracker.rotation_block(grid_view, angled, .13)[0] == 'unknown'
    assert nav.step(view(1.05, angled, grid_view), 1.05).angular > 0.
    cells[29 * 60 + 35] = 100
    occupied = grid_view._replace(cells=tuple(cells))
    assert nav.step(view(1.1, angled, occupied), 1.1).reason == 'path_occupied'
