"""Independent scan and depth assessment tests; no robot I/O."""
from __future__ import division
import numpy as np

from independent_perception import DepthProcessor, ScanProcessor
from test_depth_corridor import config, intrinsics, transform


def scan(stamp, hit=None):
    ranges = [2.] * 81
    if hit is not None:
        ranges[40] = hit
    return {'stamp': stamp, 'frame_id': 'laser_link', 'angle_min': -.3,
            'angle_increment': .6/80., 'range_min': .1, 'range_max': 10.,
            'time_increment': 0., 'ranges': ranges}


def test_scan_processing_reports_coverage_and_new_obstacle_once_per_frame():
    processor = ScanProcessor((.105, 0., 0.))
    pose = (0., 0., 0.)
    ok = processor.process(scan(10.), pose, 10., pose, 10., 10.01)
    assert ok['clear'] is True and ok['valid_until'] <= 10.3
    blocked = processor.process(scan(10.1, .2), pose, 10.1, pose, 10.1, 10.11)
    assert blocked['clear'] is False and blocked['reason'] == 'front_blocked'
    duplicate = processor.process(scan(10.1), pose, 10.1, pose, 10.1, 10.12)
    assert duplicate['clear'] is None


def test_scan_raw_validation_is_not_repeated_after_fusion(monkeypatch):
    import independent_perception as module
    actual = module._raw_scan
    calls = []
    def meter(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)
    monkeypatch.setattr(module, '_raw_scan', meter)
    pose = (0., 0., 0.)
    result = ScanProcessor((.105, 0., 0.)).process(scan(10.), pose, 10., pose, 10., 10.01)
    assert result['clear'] is True and len(calls) == 1


def test_scan_processor_owns_immutable_ranges_during_single_validation(monkeypatch):
    import independent_perception as module
    actual = module._raw_scan
    original = scan(10.)
    seen = []
    def meter(sample, *args, **kwargs):
        seen.append(sample['raw_scan']['ranges'])
        original['ranges'][40] = .2
        return actual(sample, *args, **kwargs)
    monkeypatch.setattr(module, '_raw_scan', meter)
    pose = (0., 0., 0.)
    result = ScanProcessor((.105, 0., 0.)).process(original, pose, 10., pose, 10., 10.01)
    assert type(seen[0]) is tuple and seen[0][40] == 2.
    assert result['clear'] is True


def test_scan_copy_does_not_consume_unbounded_or_nonsequence_ranges():
    value = scan(10.)
    value['ranges'] = (2. for _ in range(81))
    pose = (0., 0., 0.)
    assert ScanProcessor((.105, 0., 0.)).process(value, pose, 10., pose, 10., 10.01)['clear'] is None


def test_scan_missing_returns_and_pose_change_do_not_become_clear():
    processor = ScanProcessor((.105, 0., 0.))
    pose = (0., 0., 0.)
    value = scan(10.)
    value['ranges'] = [float('nan')] * 81
    assert processor.process(value, pose, 10., pose, 10., 10.01)['clear'] is None
    processor = ScanProcessor((.105, 0., 0.))
    assert processor.process(scan(10.), pose, 10., (.02, 0., 0.), 10., 10.01)['clear'] is None


def test_new_scan_obstacle_is_not_hidden_by_full_fusion_history():
    processor = ScanProcessor((.105, 0., 0.))
    pose = (0., 0., 0.)
    for i in range(20):
        stamp = 10. + i * .001
        processor.process(scan(stamp), pose, stamp, pose, stamp, stamp)
    stamp = 10.02
    result = processor.process(scan(stamp, .2), pose, stamp, pose, stamp, stamp)
    assert result['clear'] is False and result['reason'] == 'front_blocked'


def test_scan_gaps_remain_unknown_even_when_sampled_corridor_passes():
    processor = ScanProcessor((.105, 0., 0.))
    value = scan(10.)
    value['ranges'][::2] = [float('nan')] * 41
    pose = (0., 0., 0.)
    result = processor.process(value, pose, 10., pose, 10., 10.01)
    assert result['sampled_candidate'] is True and result['unknown_count'] == 41
    assert result['clear'] is None and result['unobserved_space'] is True


def test_scan_count_mismatch_reports_both_counts_without_authorizing(monkeypatch):
    import independent_perception as module
    monkeypatch.setattr(module, '_raw_scan', lambda *_a, **_k: (None, (80, 79)))
    pose = (0., 0., 0.)
    result = ScanProcessor((.105, 0., 0.)).process(scan(10.), pose, 10., pose, 10., 10.01)
    assert result['clear'] is None and result['reason'] == 'scan_count_mismatch'
    assert (result['raw_front_count'], result['raw_valid_count']) == (80, 79)
    assert (result['fused_front_count'], result['fused_valid_count']) == (81, 81)


def test_scan_reports_pose_stale_instead_of_misleading_count_mismatch(monkeypatch):
    processor = ScanProcessor((.105, 0., 0.))
    monkeypatch.setattr(processor.fusion, 'snapshot', lambda *_: {
        'status': 'pose_stale', 'coverage': {'front_samples': 0, 'latest_valid_samples': 0},
        'rays': [], 'occupied_hits': []})
    pose = (0., 0., 0.)
    result = processor.process(scan(10.), pose, 10., pose, 10., 10.01)
    assert result['clear'] is None and result['reason'] == 'fusion_pose_stale'


def test_depth_processor_uses_frame_stamp_and_does_not_treat_unknown_as_clear():
    processor = DepthProcessor(config())
    image = np.ones((16, 16), dtype=np.float32)
    camera = intrinsics(16, 16, fx=1000., fy=1000., cx=8., cy=8.)
    result = processor.process(image, '32FC1', camera, transform(z=0.), 10., 10.01)
    assert result['clear'] is True and result['valid_until'] == 10.15
    result = processor.process(image, '32FC1', camera, transform(z=.25), 10.1, 10.11)
    assert result['clear'] is False and result['reason'] == 'depth_obstacle'
    missing = np.zeros((16, 16), dtype=np.float32)
    result = processor.process(missing, '32FC1', camera, transform(z=0.), 10.2, 10.21)
    assert result['clear'] is None


def test_depth_with_many_invalid_pixels_keeps_unknown_explicit():
    processor = DepthProcessor(config())
    image = np.ones((16, 16), dtype=np.float32)
    image[8:, :] = 0.
    camera = intrinsics(16, 16, fx=1000., fy=1000., cx=8., cy=8.)
    result = processor.process(image, '32FC1', camera, transform(z=0.), 10., 10.01)
    assert result['clear'] is None and result['unobserved_space'] is True
    assert result['invalid_count'] == 128 and result['sampled_candidate'] is True


def test_unsampled_depth_pixels_do_not_become_known_space():
    settings = config()
    settings['sample_stride'] = 4
    processor = DepthProcessor(settings)
    image = np.zeros((64, 64), dtype=np.float32)
    image[::4, ::4] = 1.
    camera = intrinsics(64, 64, fx=1000., fy=1000., cx=32., cy=32.)
    result = processor.process(image, '32FC1', camera, transform(z=0.), 10., 10.01)
    assert result['sampled_candidate'] is True and result['invalid_count'] == 0
    assert result['clear'] is None and result['unobserved_space'] is True
