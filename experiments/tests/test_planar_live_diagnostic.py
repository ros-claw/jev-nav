"""Cross-worktree integration of non-authoritative planar ray observations."""

import copy
import sys
import unittest

import planar_live_diagnostic
from tests.test_calibration_live_diagnostic import fixture


@unittest.skipIf(sys.version_info < (3, 11), "ROSClaw requires Python 3.11+")
class PlanarLiveDiagnosticTests(unittest.TestCase):
    def test_synthetic_scan_produces_only_ray_and_hit_counts(self):
        bundle, _ = fixture()
        bundle["schema_version"] = "limo.planar_scan.v1"
        bundle["scan"]["transform"] = {
            "source": "laser",
            "target": "base_link",
            "stamp": 100.0,
            "static": False,
            "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.2], [0, 0, 0, 1]],
        }
        bundle.pop("depth")
        bundle.pop("camera_info")
        result = planar_live_diagnostic.analyze(bundle, now=100.1)
        self.assertGreater(result["valid_rays_processed"], 0)
        self.assertGreater(result["ray_sampled_cell_count"], 0)
        self.assertFalse(result["safety_clearance_verified"])
        self.assertFalse(result["transform_verified"])
        self.assertFalse(result["executable"])
        self.assertNotIn("free_cells", result)

    def test_missing_or_mismatched_base_tf_fails_closed(self):
        base, _ = fixture()
        base["schema_version"] = "limo.planar_scan.v1"
        for transform in (
            None,
            {
                "source": "other",
                "target": "base_link",
                "stamp": 100,
                "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
        ):
            bundle = copy.deepcopy(base)
            bundle["scan"]["transform"] = transform
            with self.assertRaises(ValueError):
                planar_live_diagnostic.analyze(bundle, now=100.1)
