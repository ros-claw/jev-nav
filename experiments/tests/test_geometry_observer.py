"""Bounded cross-worktree geometry integration; no real motion."""
import base64
import importlib.util
from pathlib import Path
import sys
import unittest

import geometry_observer
import sweep_simulation


@unittest.skipIf(sys.version_info < (3, 11), 'ROSClaw requires Python 3.11+')
class GeometryObserverTests(unittest.TestCase):
    def fixture(self):
        root = Path(__file__).resolve().parents[1]
        source = root / 'worktrees/limo-mcp/tests/test_local_observation_core.py'
        spec = importlib.util.spec_from_file_location('local_bundle_fixture', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scan, depth, info, scan_tf, depth_tf = module.bundle()
        data = module.core.make_bundle(scan, depth, info, scan_tf, depth_tf, now=100.1)
        data.update(ok=True, command_dispatched=False)
        return data

    def config(self):
        return {'resolution': .5, 'width': 10, 'height': 10, 'origin_x': -2.0, 'origin_y': -2.0,
                'min_z': -.1, 'max_z': 2.0, 'max_age': 1, 'max_skew': .1}

    def test_project_bundle_returns_unknown_and_occupied_without_motion_authority(self):
        result = geometry_observer.project_bundle(self.fixture(), self.config(), now=100.1)
        self.assertEqual(result['frame_id'], 'base_link')
        self.assertGreater(result['stats']['depth_projected'], 0)
        self.assertFalse(result['safety_clearance_verified'])
        self.assertFalse(result['executable'])
        self.assertTrue(any(-1 in row for row in result['cells']))

    def test_mismatched_metadata_and_corrupt_base64_fail_closed(self):
        for field, value in (('calibration_verified', True), ('usable_for_real_execution', True),
                             ('command_dispatched', True)):
            data = self.fixture(); data[field] = value
            with self.assertRaises(ValueError):
                geometry_observer.project_bundle(data, self.config(), now=100.1)
        data = self.fixture(); data['depth']['data_base64'] = 'not_base64!'
        with self.assertRaises(ValueError):
            geometry_observer.project_bundle(data, self.config(), now=100.1)

    def test_live_shaped_bundle_to_grid_cannot_treat_unknown_as_free(self):
        grid = geometry_observer.project_bundle(self.fixture(), self.config(), now=100.1)
        decision = sweep_simulation.check_sweep(grid, (0, 0, 0), .1, 0, .5, .16, 100.1)
        self.assertEqual(decision['status'], 'unknown')
        self.assertFalse(decision['executable'])
