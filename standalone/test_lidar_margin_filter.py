import math
from collections import deque

import pytest

from lidar_margin_filter import isolated_margin_bins
from explore import RosObservation


class Scan(object):
    angle_min = -1.57
    angle_increment = .007
    range_min = .1
    range_max = 8.
    time_increment = .00024
    ranges = ()

    def __init__(self, readings, sequence=0):
        self.ranges = [float('nan')] * 449
        self.header = type('Header', (), {'frame_id': 'laser_link', 'seq': sequence})()
        for index, value in readings.items():
            self.ranges[index] = value


EXTRINSICS = (.105, 0., .08, 0., 0., 0.)


def test_isolated_margin_returns_require_two_real_background_observations():
    scan = Scan({30: .1695, 33: 4.51, 220: 2.2, 221: 0.})
    previous = Scan({34: 4.48, 220: 2.2})
    assert isolated_margin_bins(scan, previous, EXTRINSICS) == (30,)
    assert scan.ranges[30] == .1695 and scan.ranges[221] == 0.
    assert math.isnan(scan.ranges[31])


@pytest.mark.parametrize('present,previous', [
    ({33: 4.51}, {}),
    ({33: 4.51}, {34: 2.17}),
    ({33: 4.51, 31: .17}, {34: 4.48}),
    ({33: 4.51}, {34: 4.48, 29: .17}),
])
def test_unobserved_background_and_correlated_close_returns_are_preserved(present, previous):
    scan = Scan(present)
    scan.ranges[30] = .1695
    assert isolated_margin_bins(scan, Scan(previous), EXTRINSICS) == ()


def test_keeps_single_core_hit_and_disables_filter_on_metadata_change():
    scan = Scan({30: .1695, 33: 4.51, 223: .1695, 225: 3.})
    previous = Scan({34: 4.48, 226: 3.})
    assert isolated_margin_bins(scan, previous, EXTRINSICS) == (30,)
    assert isolated_margin_bins(scan, previous, (.105, .03, .08, 0., 0., 0.)) == ()
    previous.angle_increment = .009
    assert isolated_margin_bins(scan, previous, EXTRINSICS) == ()


def observation_with_scan_history():
    source = RosObservation.__new__(RosObservation)
    source.epoch = source.fusion_epoch = 2
    source.extrinsics = EXTRINSICS
    source._lidar_filtered_frames = source._lidar_filtered_bins = 0
    pose = (0., 0., 0.)
    source.odom_history = deque({'stamp': t, 'pose': pose, 'epoch': 2} for t in (5., 5.11, 5.22))
    first = {'stamp': 5., 'message': Scan({34: 4.48}, sequence=18)}
    second = {'stamp': 5.11, 'message': Scan({30: .1695, 33: 4.51}, sequence=19)}
    source._filter_scan_entry(first, None)
    return source, first, second, pose


def test_owned_frame_keeps_the_unfiltered_sample_and_never_filters_persistence():
    source, first, second, _ = observation_with_scan_history()
    source._filter_scan_entry(second, first)
    assert second['raw_message'].ranges[30] == .1695
    assert math.isnan(second['message'].ranges[30])
    assert second['message'].header is second['raw_message'].header
    third = {'stamp': 5.22, 'message': Scan({30: .1695, 33: 4.51}, sequence=20)}
    source._filter_scan_entry(third, second)
    assert third['message'].ranges[30] == .1695
    assert source._lidar_filtered_frames == 1 and source._lidar_filtered_bins == 1


def test_preserves_margin_return_if_pose_or_sequence_continuity_is_lost():
    source, first, second, _ = observation_with_scan_history()
    second['message'].header.seq += 2
    source._filter_scan_entry(second, first)
    assert second['message'].ranges[30] == .1695
    second['message'].header.seq = 19
    first['filter_pose'] = (0., .01, 0.)
    source._filter_scan_entry(second, first)
    assert second['message'].ranges[30] == .1695


def test_original_range_remains_a_veto_if_latest_pose_places_it_inside_the_core():
    source, first, second, pose = observation_with_scan_history()
    source._filter_scan_entry(second, first)
    assert not source._cached_newest_scan_safe_locked({}, second, {'pose': pose},
                                                      {'pose': (0., -.02, 0.)}, 5.2)
    assert source._scan_veto_issue == 'front_blocked'
    assert not source._recent_margin_hit_in_core(pose, 5.3)
    assert source._recent_margin_hit_in_core((0., -.02, 0.), 5.3)
    assert not source._recent_margin_hit_in_core((0., -.02, 0.), 5.8)
