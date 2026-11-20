"""Explicit free/occupied/unknown merging; no ROS, device, or motion I/O."""
import pytest

from local_evidence_grid import LocalEvidenceGrid


def test_depth_can_support_lidar_unknown_without_erasing_obstacle():
    grid = LocalEvidenceGrid((4, 4, 2), {'scan': .3, 'depth': .2}, 'odom', epoch=1)
    a, b, c = (1, 1, 0), (2, 1, 0), (3, 1, 0)
    grid.update('scan', 10., 10.25, 10.01, 'odom', 1, free=[a], occupied=[b])
    grid.update('depth', 10., 10.18, 10.01, 'odom', 1, free=[b, c], occupied=[])
    view = grid.query([a, b, c, (0, 0, 1)], 10.02, 'odom', 1)
    assert view['cells'] == ['free', 'occupied', 'free', 'unknown']
    assert view['sampled_candidate'] is False and view['executable'] is False


def test_complementary_free_evidence_expiry_is_original_sensor_deadline():
    grid = LocalEvidenceGrid((3, 3, 1), {'scan': .3, 'depth': .2}, 'odom', epoch=1)
    a, b = (0, 0, 0), (1, 0, 0)
    grid.update('scan', 10., 10.25, 10.01, 'odom', 1, free=[a], occupied=[])
    grid.update('depth', 10., 10.18, 10.01, 'odom', 1, free=[a, b], occupied=[])
    view = grid.query([a, b], 10.02, 'odom', 1)
    assert view['sampled_candidate'] is True and view['valid_until'] == pytest.approx(10.18)
    assert grid.query([a, b], 10.18, 'odom', 1)['cells'] == ['free', 'unknown']


def test_new_free_frame_cannot_erase_unexpired_prior_obstacle():
    grid = LocalEvidenceGrid((2, 2, 1), {'scan': .3, 'depth': .2}, 'odom', epoch=1)
    cell = (0, 0, 0)
    grid.update('scan', 10., 10.25, 10.01, 'odom', 1, free=[], occupied=[cell])
    grid.update('scan', 10.1, 10.35, 10.11, 'odom', 1, free=[cell], occupied=[])
    assert grid.query([cell], 10.2, 'odom', 1)['cells'] == ['occupied']
    assert grid.query([cell], 10.26, 'odom', 1)['cells'] == ['free']


def test_expired_obstacle_does_not_revive_older_free_evidence():
    grid = LocalEvidenceGrid((2, 2, 1), {'scan': .3, 'depth': .2}, 'odom', epoch=1)
    cell = (0, 0, 0)
    grid.update('scan', 10., 10.3, 10.01, 'odom', 1, free=[cell], occupied=[])
    grid.update('depth', 10.1, 10.2, 10.11, 'odom', 1, free=[], occupied=[cell])
    assert grid.query([cell], 10.15, 'odom', 1)['cells'] == ['occupied']
    assert grid.query([cell], 10.21, 'odom', 1)['cells'] == ['unknown']


def test_missing_mismatched_or_out_of_order_evidence_never_becomes_free():
    grid = LocalEvidenceGrid((2, 2, 1), {'scan': .3}, 'odom', epoch=1)
    assert grid.query([(0, 0, 0)], 10., 'odom', 1)['cells'] == ['unknown']
    with pytest.raises(ValueError):
        grid.update('scan', 10., 10.31, 10.01, 'odom', 1, free=[(0, 0, 0)], occupied=[])
    with pytest.raises(ValueError):
        grid.update('scan', 10., 10.2, 10.01, 'map', 1, free=[(0, 0, 0)], occupied=[])
    with pytest.raises(ValueError):
        grid.update('scan', 10., 10.2, 10.01, 'odom', 1, free=[(2, 0, 0)], occupied=[])
