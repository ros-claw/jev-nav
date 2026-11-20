"""Cross-worktree replay integration; tests only, never physical motion."""

import importlib.util
import sys
import unittest
from pathlib import Path

import sweep_simulation


@unittest.skipIf(sys.version_info < (3, 11), "ROSClaw requires Python 3.11+")
class GridIntegrationTests(unittest.TestCase):
    def test_planar_ray_samples_cannot_be_used_as_verified_free_cells(self):
        root = Path(__file__).resolve().parents[1]
        rosclaw = root / "worktrees" / "rosclaw"
        sys.path.insert(0, str(rosclaw / "src"))
        from rosclaw.robot_pack.limo_planar_evidence import build_planar_evidence

        spec = importlib.util.spec_from_file_location(
            "planar_fixture", rosclaw / "tests/robot_pack/test_limo_planar_evidence.py"
        )
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        laser, config = fixture.fixture()
        observation = build_planar_evidence(laser, config, now=100.1)
        result = sweep_simulation.check_sweep(
            observation, (0, 0, 0), 0.1, 0, 0.5, 0.2, 100.1
        )
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["executable"])

    def test_unknown_projection_cannot_authorize_simulated_motion(self):
        root = Path(__file__).resolve().parents[1]
        rosclaw = root / "worktrees" / "rosclaw"
        sys.path.insert(0, str(rosclaw / "src"))
        spec = importlib.util.spec_from_file_location(
            "geometry_fixture", rosclaw / "tests/robot_pack/test_limo_local_geometry.py"
        )
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        laser, depth, grid = fixture.fixture()
        observation = fixture.build_evidence_grid(laser, depth, grid, now=100.1)
        result = sweep_simulation.check_sweep(
            observation, (0, 0, 0), 0.1, 0, 0.5, 0.2, 100.1
        )
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["executable"])
