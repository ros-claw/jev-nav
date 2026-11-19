"""Offline depth-visibility diagnostic tests, runnable on Python 2 and 3."""

import unittest

import numpy as np


def diagnose(*args):
    try:
        from depth_visibility import diagnose_depth_visibility
    except ImportError:
        raise AssertionError("depth visibility diagnostics are unavailable")
    return diagnose_depth_visibility(*args)


def camera(width=3, height=3, fx=2.0, fy=2.0, cx=1.0, cy=1.0):
    return {"width": width, "height": height, "fx": fx, "fy": fy,
            "cx": cx, "cy": cy, "rectified": True}


def transform(x=0.1, y=0.0, z=0.2):
    # optical right/down/forward to base forward/left/up
    return [[0.0, 0.0, 1.0, x], [-1.0, 0.0, 0.0, y],
            [0.0, -1.0, 0.0, z], [0.0, 0.0, 0.0, 1.0]]


def limits():
    return {"min_depth_m": 0.125, "max_depth_m": 4.0, "near_margin_m": 0.125,
            "surface_tolerance_m": 0.0625, "free_margin_m": 0.125}


def run(image, centers, encoding="32FC1", intrinsics=None, matrix=None, config=None):
    h, w = image.shape
    return diagnose(image, encoding, camera(w, h) if intrinsics is None else intrinsics,
                    transform() if matrix is None else matrix, centers,
                    limits() if config is None else config)


class DepthVisibilityTests(unittest.TestCase):
    def test_target_before_at_and_behind_observed_surface(self):
        image = np.full((3, 3), 1.5, dtype=np.float32)
        result = run(image, [(1.1, 0, 0.2), (1.6, 0, 0.2), (2.1, 0, 0.2)])
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["executable"])
        self.assertFalse(result["geometric_clearance_verified"])
        self.assertTrue(result["unobserved_space"])
        self.assertEqual([v["status"] for v in result["voxels"]],
                         ["sampled_free", "measured", "occluded"])
        self.assertEqual(result["counts"], {"unknown": 0, "sampled_free": 1,
                                             "measured": 1, "occluded": 1})
        self.assertEqual(result["voxels"][0]["pixel_uv"], (1, 1))
        self.assertAlmostEqual(result["voxels"][0]["target_depth_m"], 1.0)
        self.assertAlmostEqual(result["voxels"][0]["observed_depth_m"], 1.5)

    def test_base_from_optical_rotation_translation_and_pixel_orientation(self):
        image = np.full((3, 3), float("nan"), dtype=np.float32)
        image[0, 2], image[2, 0] = 2.0, 0.5
        result = run(image, [(1.1, -0.5, 0.7), (1.1, 0.5, -0.3)])
        self.assertEqual([v["pixel_uv"] for v in result["voxels"]], [(2, 0), (0, 2)])
        self.assertEqual([v["status"] for v in result["voxels"]], ["sampled_free", "occluded"])

    def test_invalid_and_near_limit_observations_remain_unknown(self):
        image = np.array([[0.0, float("nan"), float("inf"), -1.0, 4.0, 0.25]],
                         dtype=np.float32)
        intr = camera(6, 1, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        centers = [(1.1, -i, 0.2) for i in range(6)]
        result = run(image, centers, intrinsics=intr)
        self.assertEqual(result["counts"]["unknown"], 6)
        self.assertTrue(all(item["status"] == "unknown" for item in result["voxels"]))
        self.assertEqual(result["counts"]["sampled_free"], 0)
        self.assertEqual([item["reason"] for item in result["voxels"]],
                         ["invalid_depth", "invalid_depth", "invalid_depth",
                          "invalid_depth", "depth_out_of_range", "depth_near_limit"])

    def test_targets_near_lower_limit_and_beyond_range_are_unknown(self):
        image = np.array([[0.5]], dtype=np.float32)
        intr = camera(1, 1, cx=0, cy=0)
        result = run(image, [(0.325, 0, 0.2), (0.375, 0, 0.2), (4.125, 0, 0.2),
                             (0.385, 0, 0.2)], intrinsics=intr,
                     matrix=transform(x=0.125))
        self.assertEqual([v["status"] for v in result["voxels"][:3]],
                         ["unknown", "unknown", "unknown"])
        self.assertEqual(result["voxels"][3]["status"], "sampled_free")

    def test_surface_tolerance_and_free_margin_have_an_unknown_buffer(self):
        image = np.array([[1.5]], dtype=np.float32)
        intr = camera(1, 1, cx=0, cy=0)
        target_depths = [1.4375, 1.375, 1.374, 1.5625, 1.563]
        centers = [(depth, 0, 0.2) for depth in target_depths]
        result = run(image, centers, intrinsics=intr, matrix=transform(x=0))
        self.assertEqual([v["status"] for v in result["voxels"]],
                         ["measured", "unknown", "sampled_free", "measured", "occluded"])

    def test_out_of_frame_behind_camera_and_behind_base_are_unknown(self):
        image = np.ones((3, 3), dtype=np.float32)
        outside = run(image, [(1.1, 3, 0.2), (1.1, 0, 3)])
        self.assertEqual(outside["counts"]["unknown"], 2)
        self.assertTrue(all(v["pixel_uv"] is None for v in outside["voxels"]))
        behind_camera = run(image, [(1, 0, 0.2)], matrix=transform(x=2.0))
        self.assertEqual(behind_camera["voxels"][0]["status"], "unknown")
        behind_base = run(image, [(-0.1, 0, 0.2)], matrix=transform(x=-1.1))
        self.assertEqual(behind_base["voxels"][0]["status"], "unknown")

    def test_uint16_millimeters_and_float32_meters_produce_same_classifications(self):
        intr = camera(3, 1, fx=1, fy=1, cx=0, cy=0)
        centers = [(1.1, 0, 0.2), (1.1, -1, 0.2), (1.1, -2, 0.2)]
        integer = run(np.array([[1500, 1000, 0]], dtype=np.uint16), centers,
                      encoding="16UC1", intrinsics=intr)
        floating = run(np.array([[1.5, 1.0, 0.0]], dtype=np.float32), centers,
                       intrinsics=intr)
        self.assertEqual([v["status"] for v in integer["voxels"]],
                         ["sampled_free", "measured", "unknown"])
        self.assertEqual([v["status"] for v in integer["voxels"]],
                         [v["status"] for v in floating["voxels"]])

    def test_big_endian_uint16_and_noncontiguous_depth_views_are_supported(self):
        source = np.array([[9999, 1500, 9999], [9999, 1000, 9999]], dtype=">u2")
        image = source[:, 1:2]
        self.assertFalse(image.flags.c_contiguous)
        intr = camera(1, 2, fx=1, fy=1, cx=0, cy=0)
        result = run(image, [(1.1, 0, 0.2), (1.1, 0, -0.8)],
                     encoding="16UC1", intrinsics=intr)
        self.assertEqual([v["pixel_uv"] for v in result["voxels"]], [(0, 0), (0, 1)])
        self.assertEqual([v["status"] for v in result["voxels"]], ["sampled_free", "measured"])
        self.assertAlmostEqual(result["voxels"][0]["observed_depth_m"], 1.5)

    def test_no_centers_is_not_a_coverage_claim(self):
        result = run(np.ones((3, 3), dtype=np.float32), [])
        self.assertEqual(result["status"], "no_centers")
        self.assertFalse(result["executable"])
        self.assertFalse(result["geometric_clearance_verified"])
        self.assertTrue(result["unobserved_space"])
        self.assertEqual(result["voxel_count"], 0)
        self.assertEqual(sum(result["counts"].values()), 0)

    def test_invalid_inputs_fail_closed_and_do_not_issue_results(self):
        image = np.ones((3, 3), dtype=np.float32)
        malformed = [
            (image, "16UC1", camera(), transform(), [(1, 0, 0)], limits()),
            (image.astype(np.float64), "32FC1", camera(), transform(), [(1, 0, 0)], limits()),
            (image, "32FC1", dict(camera(), rectified=False), transform(), [(1, 0, 0)], limits()),
            (image, "32FC1", dict(camera(), fx=0), transform(), [(1, 0, 0)], limits()),
            (image, "32FC1", camera(), transform(), [(1, 0, float("nan"))], limits()),
            (image, "32FC1", camera(), transform(), [(1, 0, 0)],
             dict(limits(), free_margin_m=0.01)),
        ]
        for args in malformed:
            result = diagnose(*args)
            self.assertEqual(result["status"], "invalid_input")
            self.assertFalse(result["executable"])
            self.assertFalse(result["geometric_clearance_verified"])
            self.assertEqual(result["voxels"], [])

    def test_invalid_nonrigid_or_mirrored_transform_is_rejected(self):
        image = np.ones((3, 3), dtype=np.float32)
        matrices = [transform(), transform(), transform(), transform()]
        matrices[0][0][2] = 2.0
        matrices[1][1][0] = 1.0
        matrices[2][3][0] = 1.0
        matrices[3][0][3] = float("nan")
        for matrix in matrices:
            result = run(image, [(1, 0, 0)], matrix=matrix)
            self.assertEqual(result["status"], "invalid_input")
            self.assertFalse(result["executable"])


if __name__ == "__main__":
    unittest.main()
