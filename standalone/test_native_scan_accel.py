from __future__ import division

import os
import random
import subprocess
import sys

import explore
import pytest


def test_accelerator_loads_when_policy_is_imported_before_observer():
    if explore._native_scan is None:
        pytest.skip('optional native scan accelerator is not built')
    env = {'PYTHONPATH': os.path.dirname(os.path.abspath(__file__)), 'PATH': '/usr/bin:/bin'}
    code = ('import policy, forward_trial; '
            'assert policy._native_scan is not None and forward_trial._native_scan is not None')
    result = subprocess.call([sys.executable, '-c', code], env=env)
    assert result == 0


def test_optional_native_scan_path_matches_existing_checker(monkeypatch):
    assert hasattr(explore, '_native_scan')
    native = explore._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    random.seed(813)
    for case in range(160):
        rays, hits = [], []
        scan_at, now = 10.5, 10.55
        for _index in range(100):
            start = (float(.1 + random.uniform(-.01, .01)), float(random.uniform(-.01, .01)))
            end = (float(random.uniform(.8, 3.)), float(random.uniform(-1., 1.)))
            stamp = float(scan_at - random.uniform(0., .6))
            rays.append({'start_base': start, 'hit_base': end, 'stamp': stamp})
            hits.append({'point_base': end, 'stamp': stamp})
        if case % 5 == 0:
            hit = (0.4, 0.0)
            rays[10]['hit_base'] = hit
            hits[10]['point_base'] = hit
        if case % 11 == 0:
            rays[-1]['start_base'] = (float('nan'), 0.)
        if case % 13 == 0:
            hits[-1]['stamp'] = 10.4
        crossings = [float(-.145 + .01 * index) for index in range(30)]
        pose = (float(random.uniform(-.01, .01)), float(random.uniform(-.01, .01)),
                float(random.uniform(-.01, .01)))
        args = (rays, hits, crossings, now, scan_at, (0., 0., 0.), pose)
        monkeypatch.setattr(explore, '_native_scan', None)
        expected = explore._fast_newest_history(*args)
        monkeypatch.setattr(explore, '_native_scan', native)
        assert explore._fast_newest_history(*args) is expected


def test_native_threshold_ties_fall_back_to_checked_python():
    native = explore._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    stamp = 10.0
    rays, hits = [], []
    for index in range(32):
        end = (.65, 0.) if index == 0 else (2., .5)
        rays.append({'start_base': (.1, 0.), 'hit_base': end, 'stamp': stamp})
        hits.append({'point_base': end, 'stamp': stamp})
    args = (rays, hits, [0.], 10.1, 10.0, (0., 0., 0.), (0., 0., 0.))
    assert native.fast_newest_history(*args) is None
    assert explore._fast_newest_history(*args) is False


def test_optional_native_coverage_matches_existing_checker(monkeypatch):
    import policy
    assert hasattr(policy, '_native_scan')
    native = policy._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    assert hasattr(native, 'fast_assess_rays')
    random.seed(919)
    for case in range(160):
        rays, hits = [], []
        scan_at, now = 10.5, 10.55
        for index in range(100):
            start = (.1, 0.)
            end = (float(random.uniform(.8, 3.)), float(random.uniform(-1., 1.)))
            stamp = float(scan_at if index % 3 == 0 else scan_at - random.uniform(.01, .6))
            rays.append({'start_base': start, 'hit_base': end, 'stamp': stamp})
            hits.append({'point_base': end, 'stamp': stamp})
        latest_valid = sum(ray['stamp'] == scan_at for ray in rays)
        if case % 5 == 0:
            hit = (.4, 0.)
            rays[10]['hit_base'] = hit
            hits[10]['point_base'] = hit
        if case % 11 == 0:
            rays[-1]['start_base'] = (float('nan'), 0.)
        if case % 13 == 0:
            hits[-1]['stamp'] = 10.4
        args = (rays, hits, scan_at, latest_valid, now, .6, .15, .65, .035, .65, .4)
        monkeypatch.setattr(policy, '_native_scan', None)
        expected = policy._fast_assess_rays(*args)
        monkeypatch.setattr(policy, '_native_scan', native)
        assert policy._fast_assess_rays(*args) == expected


@pytest.mark.parametrize('mode', ('invalid', 'throws', 'unsafe_dict'))
def test_bad_native_result_falls_back_to_existing_checks(monkeypatch, mode):
    import policy
    class Broken(object):
        def fast_newest_history(self, *_args):
            if mode == 'throws':
                raise RuntimeError('failed')
            return 'invalid'
        def fast_assess_rays(self, *_args):
            if mode == 'throws':
                raise RuntimeError('failed')
            if mode == 'unsafe_dict':
                return {'candidate': True, 'reason': 'sampled_forward_corridor',
                        'geometric_clearance_verified': False}
            return 'invalid'
    stamp = 10.0
    rays, hits = [], []
    for _index in range(40):
        end = (2., .5)
        rays.append({'start_base': (.1, 0.), 'hit_base': end, 'stamp': stamp})
        hits.append({'point_base': end, 'stamp': stamp})
    newest_args = (rays, hits, [0.], 10.1, stamp, (0., 0., 0.), (0., 0., 0.))
    coverage_args = (rays, hits, stamp, 40, 10.1, .6, .15, .65, .035, .65, .4)
    monkeypatch.setattr(explore, '_native_scan', None)
    monkeypatch.setattr(policy, '_native_scan', None)
    expected_newest = explore._fast_newest_history(*newest_args)
    expected_coverage = policy._fast_assess_rays(*coverage_args)
    monkeypatch.setattr(explore, '_native_scan', Broken())
    monkeypatch.setattr(policy, '_native_scan', Broken())
    assert explore._fast_newest_history(*newest_args) is expected_newest
    assert policy._fast_assess_rays(*coverage_args) == expected_coverage


def test_raw_scan_veto_without_canonical_copy_matches_full_validation():
    import forward_trial
    from test_forward_trial import Clock, Source
    sample = Source(Clock()).snapshot()
    cases = ([2.] * 33, [0.] * 33, [float('nan')] * 33,
             [float('inf')] * 33, [None] * 33, [2.] * 16 + [.2] + [2.] * 16,
             [2.] * 16 + [True] + [2.] * 16)
    for values in cases:
        sample['raw_scan']['ranges'] = values
        sample['scene']['coverage']['front_samples'] = len(values)
        sample['scene']['coverage']['latest_valid_samples'] = sum(
            not isinstance(value, bool) and value is not None and explore._finite(value)
            and .1 <= value < 10. for value in values)
        expected, _ = forward_trial._raw_scan(sample)
        actual, canonical = forward_trial._raw_scan(sample, need_canonical=False)
        assert actual == expected and canonical is None


def test_native_raw_veto_matches_full_validation_on_varied_scans(monkeypatch):
    import forward_trial
    from test_forward_trial import Clock, Source
    native = forward_trial._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    calls = []
    class Meter(object):
        def fast_raw_veto(self, *args):
            result = native.fast_raw_veto(*args)
            calls.append(result is not None)
            return result
    monkeypatch.setattr(forward_trial, '_native_scan', Meter())
    random.seed(1031)
    sample = Source(Clock()).snapshot()
    sample['raw_scan']['angle_min'] = -.7
    sample['raw_scan']['angle_increment'] = 1.4 / 120.
    sample['scene']['coverage']['front_samples'] = 121
    for case in range(250):
        values = [random.uniform(.1, 3.) if random.random() < .65 else 0. for _index in range(121)]
        if case % 7 == 0:
            values[13] = float('nan')
        if case % 11 == 0:
            values[30] = float('inf')
        if case % 13 == 0:
            values[40] = None
        if case % 17 == 0:
            values[50] = True
        sample['raw_scan']['ranges'] = values
        sample['scene']['coverage']['latest_valid_samples'] = sum(
            not isinstance(value, bool) and value is not None and explore._finite(value)
            and .1 <= value < 10. for value in values)
        sample['scene']['coverage']['scan_pose'] = (random.uniform(-.01, .01),
                                                     random.uniform(-.01, .01),
                                                     random.uniform(-.01, .01))
        sample['pose'] = (random.uniform(-.01, .01), random.uniform(-.01, .01),
                          random.uniform(-.01, .01))
        expected, _ = forward_trial._raw_scan(sample)
        actual, canonical = forward_trial._raw_scan(sample, need_canonical=False)
        assert actual == expected and canonical is None
    assert sum(calls) > 100


def test_optional_native_scan_ingest_matches_scalar_fusion(monkeypatch):
    import limo_scan_fusion as fusion
    assert hasattr(fusion, '_native_scan')
    native = fusion._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    assert hasattr(native, 'fast_scan_points')
    random.seed(1447)
    for case in range(100):
        stamp = 10.0
        values = [random.uniform(.1, 5.) if random.random() < .65 else 0. for _index in range(61)]
        if case % 7 == 0:
            values[11] = float('nan')
        if case % 11 == 0:
            values[21] = float('inf')
        if case % 13 == 0:
            values[31] = None
        if case % 17 == 0:
            values[41] = True
        packet = {'stamp': stamp, 'ranges': values, 'angle_min': -1.4,
                  'angle_increment': 2.8 / 60., 'range_min': .1, 'range_max': 8.}
        pose = (random.uniform(-.01, .01), random.uniform(-.01, .01), random.uniform(-.02, .02))
        sensor = (.105, 0., .01)
        monkeypatch.setattr(fusion, '_native_scan', None)
        scalar = fusion.RollingScanFusion(sensor)
        expected_ok = scalar.add_scan(packet, pose, stamp, stamp)
        expected = scalar.snapshot(pose, stamp, stamp)
        monkeypatch.setattr(fusion, '_native_scan', native)
        accelerated = fusion.RollingScanFusion(sensor)
        actual_ok = accelerated.add_scan(packet, pose, stamp, stamp)
        actual = accelerated.snapshot(pose, stamp, stamp)
        assert actual_ok == expected_ok
        assert actual == expected
