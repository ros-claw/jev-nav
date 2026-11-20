"""Synthetic comparison only; no ROS, calibration authority, or motor control."""

import math
import unittest

from calibration_compare import compare_models


class CalibrationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.depth = [[2.0] * 31 for _ in range(5)]
        self.points = [(float(x), 0.0, 1.0) for x in (-5, -4, -3, 3, 4, 5)]
        for x, _, _ in self.points:
            self.depth[2][int(2 * x + 15)] = 1.0
        self.models = {
            "candidate_a": {"fx": 2.0, "fy": 1.0, "cx": 15.0, "cy": 2.0},
            "candidate_b": {"fx": 3.0, "fy": 1.0, "cx": 15.0, "cy": 2.0},
        }

    def test_separate_residuals_do_not_claim_a_verified_model(self):
        result = compare_models(self.points, self.depth, self.models)
        self.assertEqual(result["candidates"]["candidate_a"]["median_abs_error_m"], 0.0)
        self.assertEqual(result["candidates"]["candidate_b"]["median_abs_error_m"], 1.0)
        self.assertEqual(
            result["candidates"]["candidate_b"]["median_expected_depth_m"], 1.0
        )
        self.assertEqual(
            result["candidates"]["candidate_b"]["median_measured_depth_m"], 2.0
        )
        self.assertEqual(
            result["candidates"]["candidate_b"]["median_signed_error_m"], 1.0
        )
        bands = result["candidates"]["candidate_b"]["distance_bands"]
        self.assertEqual(bands["0.05-2m"]["matched_samples"], 6)
        self.assertEqual(bands["0.05-2m"]["median_signed_error_m"], 1.0)
        self.assertEqual(bands["4-6m"]["matched_samples"], 0)
        self.assertFalse(result["calibration_verified"])
        self.assertFalse(result["executable"])
        self.assertNotIn("winner", result)

    def test_sparse_occluded_and_duplicate_samples_do_not_produce_confident_score(self):
        self.depth[2][5] = None
        self.depth[2][7] = float("nan")
        points = self.points + [self.points[0]] * 20 + [(10000.0, 0.0, 1.0)]
        result = compare_models(points, self.depth, {"a": self.models["candidate_a"]})
        candidate = result["candidates"]["a"]
        self.assertEqual(candidate["status"], "insufficient_samples")
        self.assertEqual(candidate["matched_samples"], 4)
        self.assertIsNone(candidate["median_abs_error_m"])

    def test_invalid_points_do_not_become_matches(self):
        result = compare_models(
            [(0, 0, 0), (math.nan, 0, 1), (0, 0, -1)],
            self.depth,
            {"a": self.models["candidate_a"]},
        )
        self.assertEqual(result["candidates"]["a"]["matched_samples"], 0)
        self.assertEqual(result["candidates"]["a"]["status"], "insufficient_samples")

    def test_same_pixel_uses_nearest_point_independent_of_input_order(self):
        further_point = (self.points[0][0] * 2, 0.0, 2.0)
        model = {"a": self.models["candidate_a"]}
        nearer_first = compare_models(self.points + [further_point], self.depth, model)
        farther_first = compare_models([further_point] + self.points, self.depth, model)
        self.assertEqual(nearer_first["candidates"], farther_first["candidates"])
        self.assertEqual(nearer_first["candidates"]["a"]["p90_abs_error_m"], 0.0)

    def test_malformed_models_and_oversized_data_are_rejected(self):
        for models in (
            {},
            {"bad": {"fx": math.nan, "fy": 1, "cx": 1, "cy": 1}},
            {"bad": {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "frame": "fake"}},
        ):
            with self.assertRaises(ValueError):
                compare_models(self.points, self.depth, models)
        with self.assertRaises(ValueError):
            compare_models(self.points * 1001, self.depth, self.models)
        with self.assertRaises(ValueError):
            compare_models(self.points, [[1.0], [1.0, 2.0]], self.models)
