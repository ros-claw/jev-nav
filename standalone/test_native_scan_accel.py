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


def test_accelerator_loads_when_fusion_is_imported_before_observer():
    if explore._native_scan is None:
        pytest.skip('optional native scan accelerator is not built')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {'PYTHONPATH': os.pathsep.join((os.path.join(root, 'worker'),
                                          os.path.join(root, 'standalone'))),
           'PATH': '/usr/bin:/bin'}
    code = ('import limo_scan_fusion as fusion; import explore; '
            'assert fusion._native_scan is not None')
    assert subprocess.call([sys.executable, '-c', code], env=env) == 0


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


def test_native_newest_raw_matches_scalar_scan_veto(monkeypatch):
    from test_policy import fusion_scene
    native = explore._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    assert hasattr(native, 'fast_newest_raw')
    calls = []
    class Meter(object):
        def fast_newest_raw(self, *args):
            value = native.fast_newest_raw(*args)
            calls.append(value is not None)
            return value
        def fast_newest_history(self, *args):
            return native.fast_newest_history(*args)
    class Obj(object):
        pass
    scan = Obj()
    scan.header = Obj()
    scan.header.frame_id = 'laser_link'
    scan.angle_min, scan.angle_increment = -.8, 1.6 / 180.
    scan.time_increment, scan.range_min, scan.range_max = 0., .1, 8.
    pose = (0., 0., 0.)
    scene = fusion_scene(10.0, 10.1, [2.] * 33, [2.] * 33)
    extrinsics = (.105, 0., .1, 0., 0., 0.)
    random.seed(6724)
    for case in range(250):
        scan.ranges = [random.uniform(.1, 4.) if random.random() < .7 else 0. for _ in range(181)]
        if case % 4 == 0:
            scan.ranges[90] = .3
        if case % 7 == 0:
            scan.ranges[15] = float('nan')
        if case % 11 == 0:
            scan.ranges[45] = float('inf')
        if case % 13 == 0:
            scan.ranges[55] = None
        if case % 17 == 0:
            scan.ranges[65] = True
        shifted = (random.uniform(-.005, .005), random.uniform(-.005, .005),
                   random.uniform(-.005, .005))
        args = (scan, extrinsics, scene, 10.2, 10.1, pose, shifted, pose)
        monkeypatch.setattr(explore, '_native_scan', None)
        expected = explore.newest_scan_veto(*args)
        monkeypatch.setattr(explore, '_native_scan', Meter())
        assert explore.newest_scan_veto(*args) is expected
    assert sum(calls) > 100
    accepted = rejected = 0
    verified_future = 0
    for case in range(100):
        scan.ranges = [2. if random.random() > .1 else 0. for _ in range(181)]
        shifted = (random.uniform(-.002, .002), random.uniform(-.002, .002),
                   random.uniform(-.002, .002))
        if case % 3 == 0:
            scan.ranges[90] = .3
        args = (scan, extrinsics, scene, 10.2, 10.1, pose, shifted, pose)
        monkeypatch.setattr(explore, '_native_scan', None)
        expected = explore.newest_scan_veto(*args)
        future = 10.399999
        expected_at_future = explore.newest_scan_veto(*(args[:3] + (future,) + args[4:]))
        accepted += int(expected)
        rejected += int(not expected)
        monkeypatch.setattr(explore, '_native_scan', Meter())
        diagnostic = {}
        assert explore.newest_scan_veto(*args, **{'diagnostic': diagnostic, 'future_at': future}) is expected
        if 'future_covered' in diagnostic:
            assert diagnostic['future_covered'] is expected_at_future
            verified_future += 1
    assert accepted > 0 and rejected > 0
    assert verified_future > 0


def test_native_newest_raw_falls_back_for_geometry_boundaries_and_malformed_ranges():
    native = explore._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    pose = (0., 0., 0.)
    extrinsics = (.1, 0., .1, 0., 0., 0.)
    # A hit on the obstacle boundary must use the scalar comparison.
    assert native.fast_newest_raw([.55], 0., .1, .1, 8., extrinsics,
                                  pose, pose, explore.math.pi / 2) is None
    assert native.fast_newest_raw([True], 0., .1, .1, 8., extrinsics,
                                  pose, pose, explore.math.pi / 2) is None
    assert native.fast_newest_raw([None], 0., .1, .1, 8., extrinsics,
                                  pose, pose, explore.math.pi / 2) is None


def test_native_newest_raw_never_bypasses_custom_range_iteration(monkeypatch):
    native = explore._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    class MisleadingRanges(list):
        def __iter__(self):
            for index, value in enumerate(list.__iter__(self)):
                yield .3 if index == 90 else value
    class Obj(object):
        pass
    scan = Obj()
    scan.header = Obj()
    scan.header.frame_id = 'laser_link'
    scan.angle_min, scan.angle_increment = -.9, .01
    scan.time_increment, scan.range_min, scan.range_max = 0., .1, 8.
    scan.ranges = MisleadingRanges([2.] * 181)
    extrinsics, pose = (.1, 0., .1, 0., 0., 0.), (0., 0., 0.)
    assert native.fast_newest_raw(scan.ranges, scan.angle_min, scan.angle_increment,
                                  scan.range_min, scan.range_max, extrinsics, pose, pose,
                                  explore.math.pi / 2) is None
    monkeypatch.setattr(explore, '_native_scan', native)
    assert explore.newest_scan_veto(scan, extrinsics) is False


def test_native_newest_raw_never_bypasses_numeric_subclass_semantics(monkeypatch):
    native = explore._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    class Lower(float):
        def __le__(self, _other):
            return False
    class Obj(object):
        pass
    scan = Obj()
    scan.header = Obj()
    scan.header.frame_id = 'laser_link'
    scan.angle_min, scan.angle_increment = -.9, .01
    scan.time_increment, scan.range_min, scan.range_max = 0., Lower(.1), 8.
    scan.ranges = [2.] * 181
    extrinsics = (.1, 0., .1, 0., 0., 0.)
    monkeypatch.setattr(explore, '_native_scan', None)
    expected = explore.newest_scan_veto(scan, extrinsics)
    monkeypatch.setattr(explore, '_native_scan', native)
    assert expected is False
    assert explore.newest_scan_veto(scan, extrinsics) is expected


def test_native_newest_raw_agrees_near_coverage_gap_threshold(monkeypatch):
    native = explore._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    class Obj(object):
        pass
    scan = Obj()
    scan.header = Obj()
    scan.header.frame_id = 'laser_link'
    scan.time_increment, scan.range_min, scan.range_max = 0., .1, 8.
    scan.ranges = [2.] * 10
    extrinsics = (.1, 0., .1, 0., 0., 0.)
    lower, upper = .067, .068
    monkeypatch.setattr(explore, '_native_scan', None)
    for _ in range(35):
        step = (lower + upper) / 2.
        scan.angle_min, scan.angle_increment = -4.5 * step, step
        if explore.newest_scan_veto(scan, extrinsics):
            lower = step
        else:
            upper = step
    values = []
    for adjustment in (-1e-9, -1e-12, 0., 1e-12, 1e-9):
        step = (lower + upper) / 2. + adjustment
        scan.angle_min, scan.angle_increment = -4.5 * step, step
        monkeypatch.setattr(explore, '_native_scan', None)
        expected = explore.newest_scan_veto(scan, extrinsics)
        monkeypatch.setattr(explore, '_native_scan', native)
        assert explore.newest_scan_veto(scan, extrinsics) is expected
        values.append(expected)
    assert any(values) and not all(values)


def test_native_front_expiry_matches_checked_scalar_across_ray_histories(monkeypatch):
    import forward_trial
    import live_trial_adapter
    from test_policy import fusion_scene
    native = forward_trial._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    assert hasattr(native, 'fast_front_expiry')
    calls = []
    class Meter(object):
        def fast_front_expiry(self, *args):
            value = native.fast_front_expiry(*args)
            calls.append(value is not None)
            return value
    random.seed(1843)
    hits = [2.] * 33
    for case in range(180):
        scene = fusion_scene(10., 10.1, hits, hits)
        for ray in scene['rays']:
            if random.random() < .15:
                ray['stamp'] = 9.7
            if random.random() < .15:
                ray['hit_base'] = (2., random.uniform(-.5, .5))
        sample = {'scan_at': 10.1, 'snapshot_at': 10.2, 'scene': scene}
        monkeypatch.setattr(forward_trial, '_native_scan', None)
        expected = live_trial_adapter._validated_front_expiry(sample)
        monkeypatch.setattr(forward_trial, '_native_scan', Meter())
        assert live_trial_adapter._validated_front_expiry(sample) == expected
    assert sum(calls) > 20


def test_native_front_expiry_bounds_work_for_pathological_histories():
    import forward_trial
    native = forward_trial._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    ray = {'stamp': 10., 'start_base': (.1, 0.), 'hit_base': (2., 0.)}
    assert native.fast_front_expiry([ray] * 2049, 10., 10.1) is None
    rays = [dict(ray, stamp=10. - i * .001) for i in range(32)]
    assert native.fast_front_expiry(rays, 10., 10.1) is None


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


def test_native_fusion_transform_matches_scalar_and_keeps_shared_hit_identity(monkeypatch):
    import limo_scan_fusion as fusion
    native = fusion._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    assert hasattr(native, 'fast_transform_records')
    calls = []
    class Meter(object):
        def fast_scan_points(self, *args):
            return native.fast_scan_points(*args)
        def fast_transform_records(self, *args):
            result = native.fast_transform_records(*args)
            calls.append(result is not None)
            return result
    random.seed(4902)
    for case in range(100):
        obj = fusion.RollingScanFusion((.105, 0., .01))
        for frame in range(3):
            stamp = 10. + .1 * frame
            values = [random.uniform(.1, 5.) if random.random() < .7 else 0. for _ in range(121)]
            packet = {'stamp': stamp, 'ranges': values, 'angle_min': -1.4,
                      'angle_increment': 2.8 / 120., 'range_min': .1, 'range_max': 8.}
            pose = (random.uniform(-.01, .01), random.uniform(-.01, .01), random.uniform(-.02, .02))
            assert obj.add_scan(packet, pose, stamp, stamp)
        current = (random.uniform(-.01, .01), random.uniform(-.01, .01), random.uniform(-.02, .02))
        monkeypatch.setattr(fusion, '_native_scan', None)
        expected = obj.snapshot(current, 10.2, 10.2)
        monkeypatch.setattr(fusion, '_native_scan', Meter())
        actual = obj.snapshot(current, 10.2, 10.2)
        assert actual == expected
        for ray, hit in zip(actual['rays'], actual['occupied_hits']):
            assert ray['hit_base'] is hit['point_base']
            assert ray['stamp'] is hit['stamp']
    assert sum(calls) > 90


def test_native_fusion_batches_preserve_old_hits_and_scalar_fallback(monkeypatch):
    import limo_scan_fusion as fusion
    native = fusion._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    obj = fusion.RollingScanFusion((.105, 0., 0.))
    pose = (0., 0., 0.)
    for frame in range(5):
        stamp = 10. + .1 * frame
        values = [2.] * 401
        if frame == 0:
            values[200] = .25
        packet = {'stamp': stamp, 'ranges': values, 'angle_min': -1.4,
                  'angle_increment': .007, 'range_min': .1, 'range_max': 8.}
        assert obj.add_scan(packet, pose, stamp, stamp)
    monkeypatch.setattr(fusion, '_native_scan', None)
    expected = obj.snapshot(pose, 10.4, 10.4)
    calls = []

    class Meter(object):
        def __init__(self, decline_second=False):
            self.decline_second = decline_second

        def fast_transform_records(self, records, current_pose):
            calls.append(sum(len(record['rays']) for record in records))
            if self.decline_second and len(calls) == 2:
                return None
            return native.fast_transform_records(records, current_pose)

    monkeypatch.setattr(fusion, '_native_scan', Meter())
    actual = obj.snapshot(pose, 10.4, 10.4)
    assert len(calls) == 2 and max(calls) <= 1536 and sum(calls) == 2005
    assert actual == expected
    assert any(hit['stamp'] == 10. and hit['point_base'][0] < .4 for hit in actual['occupied_hits'])
    calls[:] = []
    monkeypatch.setattr(fusion, '_native_scan', Meter(decline_second=True))
    assert obj.snapshot(pose, 10.4, 10.4) == expected
    assert len(calls) == 2


def test_native_fusion_transform_rejects_unbounded_or_custom_values():
    import limo_scan_fusion as fusion
    native = fusion._native_scan
    if native is None:
        pytest.skip('optional native scan accelerator is not built')
    ray = {'start_world': (.1, 0.), 'hit_world': (2., 0.), 'stamp': 10.}
    pose = (0., 0., 0.)
    assert native.fast_transform_records([{'rays': [ray] * 1537}], pose) is None
    class CustomFloat(float):
        pass
    bad = dict(ray, hit_world=(CustomFloat(2.), 0.))
    assert native.fast_transform_records([{'rays': [bad]}], pose) is None
    bad = dict(ray, hit_world=(float('inf'), 0.))
    assert native.fast_transform_records([{'rays': [bad]}], pose) is None
