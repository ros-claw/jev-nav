import math
from collections import namedtuple

import pytest

from navigation import Pose
from trail_recovery import (RetraceClearance, TrailFollower, front_obstruction, front_registration, measured_trail,
                            recorded_scan_matches, scan_matches, straight_trail)


def test_retrace_requires_bounded_measured_travel_ending_at_the_reference():
    odom = [(index * .05, index * .001, 0., .02, 0., 0.) for index in range(411)]
    trace = {'mode': 'supervised_static_room', 'relay_exited': True,
             'trajectory': {'clock': 'host_monotonic_receipt', 'odom': odom}}
    reference = {'x': .41, 'y': 0., 'yaw': 0.}
    assert straight_trail(trace, reference, .32)['measured_trail_m'] >= .36
    with pytest.raises(ValueError, match='reference'):
        straight_trail(trace, dict(reference, y=.1), .32)
    odom[250] = (12.5, .25, .03, .02, 0., 0.)
    with pytest.raises(ValueError, match='straight'):
        straight_trail(trace, reference, .32)


def test_retrace_joins_recordings_at_an_identical_stopped_pose_without_filling_time():
    def recording(first, last, paused_at):
        rows = [(index * .05, (index - paused_at) * .001, 0.,
                 0. if index in (first, last) else .02, 0., 0., index * .05 + 1000., index)
                for index in range(first, last + 1)]
        return {'mode': 'supervised_static_room', 'relay_exited': True, 'relay_fault': None,
                'authority_diagnostics': {'ros': {'driver_pid_expected': 99}},
                'trajectory': {'clock': 'host_monotonic_receipt', 'odom': rows}}

    prior = recording(100, 400, 400)
    current = recording(500, 560, 500)
    reference = {'x': .04, 'y': 0., 'yaw': 0.}
    _, report = measured_trail(current, reference, .20, start_sequence=540, prior=prior)
    assert report['recorded_anchor'] == Pose(.04, 0., 0.)
    assert report['recording_join_gap_sec'] == 5.

    joined = current['trajectory']['odom']
    joined[0] = joined[0][:2] + (.002,) + joined[0][3:]
    with pytest.raises(ValueError, match='same stopped pose'):
        measured_trail(current, reference, .20, start_sequence=540, prior=prior)


def test_retrace_can_finish_at_the_end_of_a_short_recorded_strip():
    rows = [(index * .05, index * .001, 0., .02, 0., 0.) for index in range(89)]
    trace = {'mode': 'supervised_static_room', 'relay_exited': True,
             'trajectory': {'clock': 'host_monotonic_receipt', 'odom': rows}}
    points, report = measured_trail(trace, {'x': .088, 'y': 0., 'yaw': 0.}, .087)
    assert report['measured_trail_m'] == pytest.approx(.087)
    assert points[-1] == Pose(-.087, 0., 0.)


def test_front_registration_rejects_unrelated_or_insufficient_returns():
    scan = namedtuple('Scan', 'angle_min angle_increment range_min range_max ranges')
    reference = {'closest_rays': [(angle, .30) for angle in range(-4, 5)]}
    sample = scan(math.radians(-4), math.radians(1), .1, 8., [.325] * 9)
    assert front_registration(sample, reference) == pytest.approx(.025, abs=.001)
    assert front_registration(sample._replace(ranges=[.6] * 9), reference) is None
    assert front_registration(sample._replace(ranges=[.325] * 3 + [float('inf')] * 6), reference) is None


def test_retrace_tracks_measured_translation_and_pivot_in_sequence():
    rows = [(index * .05, index * .001, 0., .02, 0., 0.) for index in range(91)]
    rows.extend((4.5 + index * .05, .09, 0., 0., index * .0025, .05) for index in range(1, 65))
    rows.extend((7.7 + index * .05, .09 + index * .001 * math.cos(.16),
                 index * .001 * math.sin(.16), .02, .16, 0.) for index in range(1, 101))
    trace = {'mode': 'supervised_static_room', 'relay_exited': True,
             'trajectory': {'clock': 'host_monotonic_receipt', 'odom': rows}}
    end = rows[-1]
    reference = {'x': end[1], 'y': end[2], 'yaw': end[4]}
    points, _ = measured_trail(trace, reference, .17)
    follower = TrailFollower(points)
    pose = Pose(0., 0., 0.)
    assert follower.step(pose._replace(y=.03)).reason == 'retrace_path_departure'
    rotated = False
    for _ in range(600):
        command = follower.step(pose)
        if not (command.linear or command.angular):
            break
        rotated |= not command.linear and bool(command.angular)
        pose = Pose(pose.x + command.linear * .04 * math.cos(pose.yaw),
                    pose.y + command.linear * .04 * math.sin(pose.yaw), pose.yaw + command.angular * .04)
    assert rotated and command.reason == 'retrace_distance_reached'
    assert math.hypot(pose.x - points[-1].x, pose.y - points[-1].y) < .008


def test_retrace_stops_for_live_front_obstacle_or_no_valid_scan():
    scan = namedtuple('Scan', 'angle_min angle_increment range_min range_max ranges')
    sample = scan(-.1, .025, .04, 8., [.3] * 9)
    assert front_obstruction(sample, .05) is None
    obstruction = front_obstruction(sample._replace(ranges=[.055] * 9), -.05)
    assert obstruction['current_footprint'] and obstruction['range_m'] == .055
    assert front_obstruction(sample._replace(ranges=[float('inf')] * 9), .05) == {'valid_returns': 0}


def test_retrace_checks_reverse_translation_along_with_the_turn():
    scan = namedtuple('Scan', 'angle_min angle_increment range_min range_max ranges')
    # This front-left return stopped the prior trial when the check treated
    # the reversing vehicle as rotating in place.
    sample = scan(math.radians(55.122497), .01, .04, 8., [.1185] + [.6] * 8)
    angular = -.02756939
    assert front_obstruction(sample, angular)['current_footprint'] is False
    assert front_obstruction(sample, angular, linear=-.018) is None
    close = sample._replace(ranges=[.113] + [.6] * 8)
    assert front_obstruction(close, angular, linear=-.018)['current_footprint'] is True


def test_retrace_resumes_only_after_sustained_fresh_clearance():
    gate = RetraceClearance()
    assert gate.observe(1., 1., False) == 'clear'
    assert gate.observe(1.05, 1.04, True) == 'hold'
    assert gate.observe(1.20, 1.18, False) == 'hold'
    assert gate.observe(1.41, 1.18, False) == 'hold'
    assert gate.observe(1.42, None, False) == 'hold'
    assert gate.observe(1.65, 1.64, False) == 'hold'
    assert gate.observe(1.84, 1.82, False) == 'hold'
    assert gate.observe(1.99, 1.98, False) == 'hold'
    assert gate.observe(2.04, 2.03, False) == 'clear'
    assert gate.observe(2.10, 2.10, True) == 'hold'
    assert gate.observe(4.12, 4.11, True) == 'timed_out'


def test_recorded_reference_handles_alternating_invalid_laser_bins():
    scan = namedtuple('Scan', 'angle_min angle_increment range_min range_max ranges')
    expected = {'closest_rays': [(index * .8, .2 + index * .002) for index in range(10)]}
    ranges = []
    for _, distance in expected['closest_rays']:
        ranges.extend([float('inf'), distance])
    sample = scan(0., math.radians(.4), .04, 8., ranges)
    assert not scan_matches(sample, expected)
    assert recorded_scan_matches(sample, expected)
    assert not recorded_scan_matches(sample._replace(ranges=[.22] * len(ranges)), expected)
