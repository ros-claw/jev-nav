"""Tests for the scan evidence component, runnable on both Python 2 and 3."""

from __future__ import division  # noqa: UP010 - Python 2 tests

import importlib
import math
import os
import struct
import sys
import unittest

WORKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker")


def fusion(**options):
    path = os.path.join(WORKER, "limo_scan_fusion.py")
    assert os.path.isfile(path), "front-only fusion component is missing"
    sys.path.insert(0, WORKER)
    try:
        module = importlib.import_module("limo_scan_fusion")
    finally:
        sys.path.pop(0)
    sensor_pose = options.pop("sensor_pose", (0.0, 0.0, 0.0))
    return module.RollingScanFusion(sensor_pose=sensor_pose, **options)


def scan(stamp, ranges, angle_min=-math.pi / 2, angle_increment=math.pi / 2):
    return {
        "stamp": stamp,
        "ranges": ranges,
        "angle_min": angle_min,
        "angle_increment": angle_increment,
        "range_min": 0.1,
        "range_max": 10.0,
    }


class ScanFusionTests(unittest.TestCase):
    def test_only_valid_front_sector_rays_contribute_to_coverage(self):
        evidence = fusion()
        packet = scan(10.0, [0.2, 1.0, 2.0, float("nan"), 0.2], -math.pi, math.pi / 2)
        self.assertTrue(evidence.add_scan(packet, (0, 0, 0), 10.0, 10.0))
        result = evidence.snapshot((0, 0, 0), 10.0, 10.0)
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["executable"])
        self.assertEqual(result["coverage"]["front_samples"], 3)
        self.assertEqual(result["coverage"]["latest_valid_samples"], 2)
        self.assertAlmostEqual(result["coverage"]["latest_valid_fraction"], 2 / 3)
        self.assertEqual(len(result["rays"]), 2)
        self.assertEqual(len(result["occupied_hits"]), 2)

    def test_sensor_offset_and_odometry_translation_rotation_are_applied(self):
        evidence = fusion(sensor_pose=(0.2, 0.1, math.pi / 2))
        self.assertTrue(evidence.add_scan(scan(10, [2], 0, 0.1), (1, 2, math.pi / 2), 10, 10))
        result = evidence.snapshot((0.5, 1, 0), 10.1, 10.1)
        ray = result["rays"][0]
        self.assertAlmostEqual(ray["start_base"][0], 0.3)
        self.assertAlmostEqual(ray["start_base"][1], 1.2)
        self.assertAlmostEqual(ray["hit_base"][0], -1.6)
        self.assertAlmostEqual(ray["hit_base"][1], 1.2)
        rotated = evidence.snapshot((0.5, 1, math.pi / 2), 10.2, 10.2)
        self.assertAlmostEqual(rotated["rays"][0]["hit_base"][0], 1.2)
        self.assertAlmostEqual(rotated["rays"][0]["hit_base"][1], 1.6)

    def test_invalid_readings_do_not_create_samples_or_hits(self):
        evidence = fusion()
        values = [0, -1, 0.099, 0.1, 9.9, 10, 11, float("inf"), float("nan")]
        self.assertTrue(evidence.add_scan(scan(10, values, -0.4, 0.1), (0, 0, 0), 10, 10))
        result = evidence.snapshot((0, 0, 0), 10, 10)
        self.assertEqual(result["coverage"]["front_samples"], 9)
        self.assertEqual(result["coverage"]["latest_valid_samples"], 2)
        self.assertEqual(len(result["rays"]), 2)
        self.assertEqual(len(result["occupied_hits"]), 2)

    def test_raw_bin_diagnostics_count_only_front_sector_and_sum_to_front_samples(self):
        evidence = fusion()
        values = [float("nan"), 0, float("nan"), 0, 0.5, 12, float("inf"), -0.0, 1]
        self.assertTrue(evidence.add_scan(scan(10, values, -math.pi, math.pi / 4),
                                          (0, 0, 0), 10, 10))
        coverage = evidence.snapshot((0, 0, 0), 10, 10)["coverage"]
        self.assertEqual(coverage["front_samples"], 5)
        self.assertEqual(coverage.get("latest_nan_bins"), 1)
        self.assertEqual(coverage.get("latest_zero_bins"), 1)
        self.assertEqual(coverage.get("latest_finite_valid_samples"), 1)
        self.assertEqual(coverage.get("latest_other_bins"), 2)
        self.assertEqual(coverage["latest_valid_samples"], 1)
        self.assertEqual(coverage["front_samples"], sum(coverage[name] for name in (
            "latest_nan_bins", "latest_zero_bins", "latest_finite_valid_samples",
            "latest_other_bins")))

    def test_raw_bin_diagnostics_do_not_count_booleans_or_invalid_numbers_as_zero_or_hits(self):
        evidence = fusion()
        values = [float("nan"), -float("nan"), 0, -0.0, 0.1, 9.9, -1, 0.099, 10, 11,
                  float("inf"), -float("inf"), None, False, True, "nan", 10 ** 400]
        self.assertTrue(evidence.add_scan(scan(10, values, -0.5, 0.05), (0, 0, 0), 10, 10))
        result = evidence.snapshot((0, 0, 0), 10, 10)
        coverage = result["coverage"]
        self.assertEqual(coverage.get("latest_nan_bins"), 2)
        self.assertEqual(coverage.get("latest_zero_bins"), 2)
        self.assertEqual(coverage.get("latest_finite_valid_samples"), 2)
        self.assertEqual(coverage.get("latest_other_bins"), 11)
        self.assertEqual(coverage["front_samples"], 17)
        self.assertEqual(len(result["rays"]), 2)
        self.assertEqual(len(result["occupied_hits"]), 2)

    def test_raw_bin_diagnostics_describe_last_accepted_scan_even_when_stale_or_faulted(self):
        evidence = fusion()
        initial = evidence.snapshot((0, 0, 0), 10, 10)["coverage"]
        self.assertIsNone(initial["scan_stamp"])
        self.assertEqual(initial.get("latest_nan_bins"), 0)
        self.assertTrue(evidence.add_scan(scan(10.1, [1, 0, float("nan")]),
                                          (0, 0, 0), 10.1, 10.1))
        self.assertTrue(evidence.add_scan(scan(10.2, [0, float("nan"), float("inf")]),
                                          (0, 0, 0), 10.2, 10.2))
        latest = evidence.snapshot((0, 0, 0), 10.2, 10.2)
        self.assertEqual(latest["status"], "no_valid_front_rays")
        self.assertEqual(len(latest["occupied_hits"]), 1)
        self.assertEqual(latest["coverage"].get("latest_nan_bins"), 1)
        self.assertEqual(latest["coverage"].get("latest_zero_bins"), 1)
        self.assertEqual(latest["coverage"].get("latest_finite_valid_samples"), 0)
        self.assertEqual(latest["coverage"].get("latest_other_bins"), 1)
        self.assertFalse(evidence.add_scan(scan(10.3, [2], 0, 0), (0, 0, 0), 10.3, 10.3))
        fault = evidence.snapshot((0, 0, 0), 10.3, 10.3)
        self.assertEqual(fault["status"], "invalid_scan")
        self.assertFalse(fault["coverage"]["sensor_fresh"])
        self.assertEqual(fault["coverage"]["scan_stamp"], 10.2)
        self.assertEqual(fault["coverage"].get("latest_nan_bins"), 1)
        self.assertEqual(fault["coverage"].get("latest_zero_bins"), 1)
        self.assertTrue(evidence.add_scan(scan(10.4, [float("nan"), 0, 2]),
                                          (0, 0, 0), 10.4, 10.4))
        stale = evidence.snapshot((0, 0, 0), 11.0, 11.0)
        self.assertEqual(stale["status"], "sensor_stale")
        self.assertFalse(stale["coverage"]["sensor_fresh"])
        self.assertEqual(stale["coverage"]["scan_stamp"], 10.4)
        self.assertEqual(stale["coverage"].get("latest_nan_bins"), 1)
        self.assertEqual(stale["coverage"].get("latest_zero_bins"), 1)
        self.assertEqual(stale["coverage"].get("latest_finite_valid_samples"), 1)

    def test_stationary_complementary_scans_report_coverage_union(self):
        evidence = fusion()
        self.assertTrue(evidence.add_scan(scan(10, [1, 0, 0]), (0, 0, 0), 10, 10))
        self.assertTrue(evidence.add_scan(scan(10.1, [0, 2, 0]), (0, 0, 0), 10.1, 10.1))
        result = evidence.snapshot((0, 0, 0), 10.1, 10.1)
        coverage = result["coverage"]
        self.assertEqual(coverage["latest_valid_samples"], 1)
        self.assertEqual(coverage["same_pose_valid_samples"], 2)
        self.assertAlmostEqual(coverage["same_pose_valid_fraction"], 2 / 3)
        self.assertEqual(coverage["same_pose_scans"], 2)

    def test_union_requires_exact_matching_scan_geometry_and_pose(self):
        evidence = fusion()
        self.assertTrue(evidence.add_scan(scan(10, [1, 0, 0]), (0, 0, 0), 10, 10))
        self.assertTrue(evidence.add_scan(scan(10.1, [0, 1, 0]), (0.01, 0, 0), 10.1, 10.1))
        self.assertTrue(evidence.add_scan(scan(10.2, [0, 0, 1], -1.5, 1.5),
                                          (0.01, 0, 0), 10.2, 10.2))
        result = evidence.snapshot((0.01, 0, 0), 10.2, 10.2)
        self.assertEqual(result["coverage"]["same_pose_valid_samples"], 1)
        self.assertEqual(result["coverage"]["same_pose_scans"], 1)

    def test_a_new_farther_return_does_not_erase_recent_obstacle(self):
        evidence = fusion()
        self.assertTrue(evidence.add_scan(scan(10, [1], 0, 0.1), (0, 0, 0), 10, 10))
        self.assertTrue(evidence.add_scan(scan(10.1, [3], 0, 0.1), (0, 0, 0), 10.1, 10.1))
        result = evidence.snapshot((0, 0, 0), 10.1, 10.1)
        self.assertEqual([hit["point_base"][0] for hit in result["occupied_hits"]], [1, 3])

    def test_sensor_outage_disables_rays_and_expired_hits_disappear(self):
        evidence = fusion(max_evidence_age=1.0, max_scan_age=0.5)
        self.assertTrue(evidence.add_scan(scan(10, [1], 0, 0.1), (0, 0, 0), 10, 10))
        stale = evidence.snapshot((0, 0, 0), 10.6, 10.6)
        self.assertEqual(stale["status"], "sensor_stale")
        self.assertFalse(stale["coverage"]["sensor_fresh"])
        self.assertEqual(stale["rays"], [])
        self.assertEqual(len(stale["occupied_hits"]), 1)
        expired = evidence.snapshot((0, 0, 0), 11.01, 11.01)
        self.assertEqual(expired["rays"], [])
        self.assertEqual(expired["occupied_hits"], [])

    def test_invalid_current_scan_does_not_reuse_previous_rays(self):
        evidence = fusion()
        self.assertTrue(evidence.add_scan(scan(10, [1], 0, 0.1), (0, 0, 0), 10, 10))
        self.assertTrue(evidence.add_scan(scan(10.1, [0], 0, 0.1), (0, 0, 0), 10.1, 10.1))
        result = evidence.snapshot((0, 0, 0), 10.1, 10.1)
        self.assertEqual(result["status"], "no_valid_front_rays")
        self.assertEqual(result["rays"], [])
        self.assertEqual(result["coverage"]["latest_valid_samples"], 0)
        self.assertEqual(len(result["occupied_hits"]), 1)

    def test_scan_with_stale_or_future_time_or_misaligned_pose_is_rejected(self):
        for stamp, pose_stamp, now in ((10, 10, 10.6), (11, 11, 10), (10, 9.9, 10),
                                        (10, 10.01, 10)):
            evidence = fusion()
            self.assertFalse(evidence.add_scan(scan(stamp, [1], 0, 0.1),
                                               (0, 0, 0), pose_stamp, now))
            result = evidence.snapshot((0, 0, 0), now, now)
            self.assertNotEqual(result["status"], "ok")
            self.assertEqual(result["rays"], [])

    def test_stale_current_pose_disables_all_transformed_evidence(self):
        evidence = fusion()
        self.assertTrue(evidence.add_scan(scan(10, [1], 0, 0.1), (0, 0, 0), 10, 10))
        result = evidence.snapshot((0, 0, 0), 10, 10.2)
        self.assertEqual(result["status"], "pose_stale")
        self.assertEqual(result["rays"], [])
        self.assertEqual(result["occupied_hits"], [])
        self.assertIsNone(result["coverage"]["scan_stamp"])
        self.assertEqual(result["coverage"]["latest_finite_valid_samples"], 0)

    def test_clock_regression_cannot_revive_expired_observations(self):
        evidence = fusion()
        self.assertTrue(evidence.add_scan(scan(10, [1], 0, 0.1), (0, 0, 0), 10, 10))
        self.assertEqual(evidence.snapshot((0, 0, 0), 11.1, 11.1)["occupied_hits"], [])
        backwards = evidence.snapshot((0, 0, 0), 10.1, 10.1)
        self.assertNotEqual(backwards["status"], "ok")
        self.assertEqual(backwards["rays"], [])
        self.assertEqual(backwards["occupied_hits"], [])

    def test_malformed_packet_latches_a_fault_and_preserves_unexpired_hits(self):
        evidence = fusion()
        self.assertTrue(evidence.add_scan(scan(10, [1], 0, 0.1), (0, 0, 0), 10, 10))
        self.assertFalse(evidence.add_scan(scan(10.1, [2], 0, 0), (0, 0, 0), 10.1, 10.1))
        fault = evidence.snapshot((0, 0, 0), 10.1, 10.1)
        self.assertEqual(fault["status"], "invalid_scan")
        self.assertFalse(fault["coverage"]["sensor_fresh"])
        self.assertEqual(fault["rays"], [])
        self.assertEqual(len(fault["occupied_hits"]), 1)
        self.assertTrue(evidence.add_scan(scan(10.2, [3], 0, 0.1), (0, 0, 0), 10.2, 10.2))
        self.assertEqual(evidence.snapshot((0, 0, 0), 10.2, 10.2)["status"], "ok")

    def test_capacity_limit_fails_closed_without_dropping_unexpired_obstacles(self):
        evidence = fusion(max_scans=1, max_evidence_age=0.5, max_scan_age=0.5)
        self.assertTrue(evidence.add_scan(scan(10, [1], 0, 0.1), (0, 0, 0), 10, 10))
        self.assertFalse(evidence.add_scan(scan(10.1, [2], 0, 0.1), (0, 0, 0), 10.1, 10.1))
        fault = evidence.snapshot((0, 0, 0), 10.1, 10.1)
        self.assertEqual(fault["status"], "capacity_exceeded")
        self.assertEqual(fault["rays"], [])
        self.assertEqual(fault["occupied_hits"][0]["point_base"], (1, 0))
        self.assertTrue(evidence.add_scan(scan(10.51, [2], 0, 0.1),
                                          (0, 0, 0), 10.51, 10.51))
        result = evidence.snapshot((0, 0, 0), 10.51, 10.51)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([hit["point_base"] for hit in result["occupied_hits"]], [(2, 0)])

    def test_descending_scan_angles_produce_only_real_front_samples(self):
        evidence = fusion()
        packet = scan(10, [1, 2, 3, 4, 5], math.pi, -math.pi / 2)
        self.assertTrue(evidence.add_scan(packet, (0, 0, 0), 10, 10))
        result = evidence.snapshot((0, 0, 0), 10, 10)
        self.assertEqual(result["coverage"]["front_samples"], 3)
        self.assertEqual(result["coverage"]["latest_valid_samples"], 3)
        self.assertEqual(len(result["rays"]), 3)
        self.assertAlmostEqual(result["rays"][0]["hit_base"][1], 2)
        self.assertAlmostEqual(result["rays"][2]["hit_base"][1], -4)

    def test_extreme_pose_angles_and_coordinates_fail_closed(self):
        with self.assertRaises(ValueError):
            fusion(sensor_pose=(0, 0, 1e308))
        evidence = fusion()
        self.assertFalse(evidence.add_scan(scan(10, [1], 0, 0.1), (0, 0, 1e308), 10, 10))
        self.assertTrue(evidence.add_scan(scan(10.1, [1], 0, 0.1), (0, 0, 0), 10.1, 10.1))
        for pose in ((1e16, 0, 0), (0, 0, 1e308)):
            result = evidence.snapshot(pose, 10.1, 10.1)
            self.assertNotEqual(result["status"], "ok")
            self.assertEqual(result["rays"], [])
            self.assertEqual(result["occupied_hits"], [])

    def test_arithmetic_overflow_in_coordinates_is_rejected(self):
        evidence = fusion(sensor_pose=(1e308, 0, 0))
        packet = scan(10, [1e307], 0, 0.1)
        packet["range_min"] = 1e307
        packet["range_max"] = 1.7e308
        self.assertFalse(evidence.add_scan(packet, (1e308, 0, 0), 10, 10))

        evidence = fusion()
        self.assertTrue(evidence.add_scan(packet, (0, 0, 0), 10, 10))
        result = evidence.snapshot((1.7e308, -1.7e308, math.pi / 4), 10, 10)
        self.assertNotEqual(result["status"], "ok")
        self.assertEqual(result["rays"], [])
        self.assertEqual(result["occupied_hits"], [])

    def test_float_subclass_cannot_hide_nonfinite_transformed_coordinates(self):
        class MisleadingNaN(float):
            def __lt__(self, other):
                return True

            def __gt__(self, other):
                return True

            def __rmul__(self, other):
                return self

            def __add__(self, other):
                return self

        class PoseCoordinate(float):
            def __rsub__(self, other):
                return MisleadingNaN(float("nan"))

        evidence = fusion()
        self.assertTrue(evidence.add_scan(scan(10, [1], 0, 0.1), (0, 0, 0), 10, 10))
        result = evidence.snapshot((PoseCoordinate(0), 0, 0), 10, 10)
        self.assertEqual(result["status"], "pose_invalid")
        self.assertEqual(result["rays"], [])
        self.assertEqual(result["occupied_hits"], [])

    def test_negative_pose_timestamp_is_never_valid(self):
        evidence = fusion()
        self.assertFalse(evidence.add_scan(scan(0, [1], 0, 0.1), (0, 0, 0), -0.01, 0))
        self.assertTrue(evidence.add_scan(scan(0.01, [1], 0, 0.1),
                                          (0, 0, 0), 0.01, 0.01))
        result = evidence.snapshot((0, 0, 0), -0.01, 0.01)
        self.assertNotEqual(result["status"], "ok")
        self.assertEqual(result["rays"], [])

    def test_float32_ros_angle_metadata_at_pi_is_accepted_without_rear_samples(self):
        def float32(value):
            return struct.unpack("f", struct.pack("f", value))[0]

        evidence = fusion()
        packet = scan(10, [1.0] * 361, float32(-math.pi), float32(math.pi / 180))
        self.assertTrue(evidence.add_scan(packet, (0, 0, 0), 10, 10))
        result = evidence.snapshot((0, 0, 0), 10, 10)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["coverage"]["front_samples"], 180)
        self.assertEqual(len(result["rays"]), 180)


if __name__ == "__main__":
    unittest.main()
