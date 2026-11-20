"""Project only explicitly classified sample centers into grid cell IDs."""
import pytest

from sensor_cell_projection import depth_cells


def visibility(statuses):
    centers = [(0.6, -.1, .1), (0.6, 0., .1), (0.6, .1, .1), (0.7, 0., .1)]
    result = {'status': 'ok', 'executable': False, 'geometric_clearance_verified': False,
              'unobserved_space': True,
              'voxels': [{'center_base_m': c, 'status': s} for c, s in zip(centers, statuses)]}
    cells = [((i, 0, 0), c) for i, c in enumerate(centers)]
    return result, cells


def test_only_explicitly_observed_centers_become_free_or_occupied():
    result, cells = visibility(['sampled_free', 'measured', 'occluded', 'unknown'])
    mapped = depth_cells(result, cells)
    assert mapped['free'] == [(0, 0, 0)]
    assert mapped['occupied'] == [(1, 0, 0)]
    assert mapped['unknown'] == [(2, 0, 0), (3, 0, 0)]
    assert mapped['executable'] is False and mapped['unobserved_space'] is True


def test_wrong_center_or_duplicate_cell_cannot_be_relabelled():
    result, cells = visibility(['sampled_free'] * 4)
    cells[0] = (cells[0][0], (0.6, -.2, .1))
    with pytest.raises(ValueError):
        depth_cells(result, cells)
    result, cells = visibility(['sampled_free'] * 4)
    cells[1] = (cells[0][0], cells[1][1])
    with pytest.raises(ValueError):
        depth_cells(result, cells)


def test_real_depth_visibility_can_complement_a_separate_observed_lidar_cell():
    import numpy as np
    from local_evidence_grid import LocalEvidenceGrid
    from test_depth_visibility import run
    centers = [(1.1, 0., .2), (.2, 0., .2)]
    classified = run(np.full((3, 3), 1.5, dtype=np.float32), centers)
    mapped = depth_cells(classified, [((0, 0, 0), centers[0]), ((1, 0, 0), centers[1])])
    assert mapped['free'] == [(0, 0, 0)] and mapped['unknown'] == [(1, 0, 0)]
    grid = LocalEvidenceGrid((2, 1, 1), {'scan': .3, 'depth': .2}, 'odom', epoch=1)
    grid.update('depth', 10., 10.18, 10.01, 'odom', 1, mapped['free'], mapped['occupied'])
    grid.update('scan', 10., 10.25, 10.01, 'odom', 1, [(1, 0, 0)], [])
    view = grid.query([(0, 0, 0), (1, 0, 0)], 10.02, 'odom', 1)
    assert view['sampled_candidate'] is True and view['executable'] is False
