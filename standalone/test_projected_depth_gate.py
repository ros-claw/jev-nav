from __future__ import division

import copy

import pytest
from projected_depth_gate import ProjectedDepthGate, project_depth_frame


def sample():
    return {'depth_at': 100.0, 'depth': {'stamp': 100.0, 'corridor': {
        'stamp': 100.0, 'status': 'observed', 'sample_count': 4000,
        'valid_count': 300, 'invalid_count': 3700, 'sample_stride': 4,
        'ground_count': 100, 'obstacle_count': 0, 'unclassified_corridor_count': 0,
        'outside_corridor_count': 200, 'calibration_verified': False,
        'unobserved_space': True, 'geometric_clearance_verified': False,
        'executable': False,
    }}}


def test_valid_ground_and_no_observed_obstacles_is_candidate_but_not_clearance():
    result = ProjectedDepthGate().assess(sample(), 100.2)
    assert result['candidate'] is True
    assert result['reason'] == 'sampled_depth_corridor'
    assert result['depth_at'] == 100.0
    assert result['geometric_clearance_verified'] is False
    assert result['unobserved_space'] is True
    assert result['calibration_verified'] is False


@pytest.mark.parametrize('mutator,reason', [
    (lambda s: s['depth']['corridor'].update(obstacle_count=1, outside_corridor_count=199), 'depth_obstacle'),
    (lambda s: s['depth']['corridor'].update(unclassified_corridor_count=1, outside_corridor_count=199), 'depth_unclassified'),
    (lambda s: s['depth']['corridor'].update(ground_count=0, outside_corridor_count=300), 'depth_unavailable'),
    (lambda s: s['depth']['corridor'].update(status='no_valid_depth'), 'depth_unavailable'),
    (lambda s: s['depth']['corridor'].update(stamp=99.9), 'depth_unavailable'),
    (lambda s: s['depth'].update(stamp=99.9), 'depth_unavailable'),
    (lambda s: s['depth']['corridor'].update(calibration_verified=True), 'depth_unavailable'),
    (lambda s: s['depth']['corridor'].update(unobserved_space=False), 'depth_unavailable'),
    (lambda s: s['depth']['corridor'].update(obstacle_count=False), 'depth_unavailable'),
    (lambda s: s['depth']['corridor'].update(valid_count=301), 'depth_unavailable'),
])
def test_rejects_obstacles_or_invalid_evidence(mutator, reason):
    item = copy.deepcopy(sample())
    mutator(item)
    result = ProjectedDepthGate().assess(item, 100.2)
    assert result['candidate'] is False
    assert result['reason'] == reason
    assert result['geometric_clearance_verified'] is False


def test_rejects_stale_or_malformed_sample():
    gate = ProjectedDepthGate()
    for value in (None, {}, {'depth_at': float('nan')}, sample()):
        assert gate.assess(value, 101.0)['candidate'] is False


def test_projection_is_tagged_with_exact_frame_time_and_assumptions(monkeypatch):
    captured = []

    def fake(array, encoding, intrinsics, transform, config):
        captured.append((array, encoding, intrinsics, transform, config))
        return {'status': 'observed', 'executable': False, 'unobserved_space': True,
                'geometric_clearance_verified': False}

    monkeypatch.setattr('projected_depth_gate.diagnose_depth_corridor', fake)
    result = project_depth_frame('array', '16UC1', {'rectified': True}, 'transform', 12.3, {'x': 1})
    assert result['stamp'] == 12.3
    assert result['calibration_verified'] is False
    assert captured == [('array', '16UC1', {'rectified': True}, 'transform', {'x': 1})]


def test_projection_rejects_nonfinite_timestamp():
    assert project_depth_frame(None, None, None, None, float('nan'), None) is None
