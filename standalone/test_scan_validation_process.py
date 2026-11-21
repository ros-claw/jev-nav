import math

import pytest

from explore import RosObservation
from scan_validation_process import ScanFrame, ScanHeader, ScanValidationProcess


@pytest.mark.parametrize('kind', ('clear', 'obstacle', 'unknown'))
def test_scan_process_matches_local_geometry_and_expiry(kind):
    now, stamp = 10.1, 10.0
    angle, step = -.35, .02
    values = [2.] * 36
    if kind == 'obstacle':
        values[18] = .3
    elif kind == 'unknown':
        values = [float('nan')] * 36
    message = ScanFrame(ScanHeader('laser_link'), angle, step, .1, 8., .0001, values)
    scan = {'message': message, 'stamp': stamp}
    rays, hits = [], []
    for index in range(len(values)):
        theta = angle + index * step
        point = (.1 + 2. * math.cos(theta), 2. * math.sin(theta))
        ray_stamp = 9.68
        rays.append({'start_base': (.1, 0.), 'hit_base': point, 'stamp': ray_stamp})
        hits.append({'point_base': point, 'stamp': ray_stamp})
    pose = {'pose': (0., 0., 0.)}
    sample = {'scene': {'status': 'ok', 'coverage': {'sensor_fresh': True},
                        'rays': rays, 'occupied_hits': hits}, 'pose': pose['pose']}
    observer = RosObservation.__new__(RosObservation)
    observer._split_validation = True
    observer.epoch, observer.extrinsics = 1, (.1, 0., .1, 0., 0., 0.)
    accepted = observer._cached_newest_scan_safe_locked(sample, scan, pose, pose, now)
    assert accepted is (kind == 'clear')
    with ScanValidationProcess() as worker:
        verdict = worker.check(observer.epoch, observer.extrinsics, sample, scan, pose, pose, now)
        assert verdict == {'stamp': stamp, 'accepted': accepted, 'reason': observer._scan_veto_issue,
                           'valid_until': observer._scan_veto_valid_until if accepted else None}


@pytest.mark.parametrize('blocked', (False, True))
def test_veto_distinguishes_sparse_history_from_a_measured_hit(blocked):
    from explore import newest_scan_veto
    scan = ScanFrame(ScanHeader('laser_link'), -.3, .1, .1, 8., 0., [2.] * 7)
    ray_end = (.4, .01) if blocked else (2., .19)
    ray = {'start_base': (.1, 0.), 'hit_base': ray_end, 'stamp': 9.68}
    hit = {'point_base': ray_end, 'stamp': 9.68}
    scene = {'status': 'ok', 'coverage': {'sensor_fresh': True}, 'rays': [ray] * 33,
             'occupied_hits': [hit] * 33}
    pose = (0., 0., 0.)
    diagnostic = {}
    assert newest_scan_veto(scan, (.1, 0., .1, 0., 0., 0.), scene, 10.1, 10., pose, pose, pose,
                            diagnostic=diagnostic, future_at=10.29) is False
    assert diagnostic['reason'] == ('front_blocked' if blocked else 'scan_coverage_pending')
