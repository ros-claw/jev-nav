"""Synthetic end-to-end optical projection diagnostic; no robot control."""

import base64
import copy
import struct
import unittest
from unittest.mock import patch

from calibration_live_diagnostic import analyze


def fixture():
    width, height = 7, 3
    depth = struct.pack("<" + "H" * (width * height), *([1000] * (width * height)))
    camera_frame = "camera_color_optical_frame"
    bundle = {
        "ok": True,
        "schema_version": "limo.calibration_diagnostic.v1",
        "command_dispatched": False,
        "diagnostic_only": True,
        "calibration_verified": False,
        "transform_verified": False,
        "usable_for_real_execution": False,
        "deskewed": False,
        "scan": {
            "stamp": 100.0,
            "frame_id": "laser",
            "angle_min": -0.5,
            "angle_increment": 0.2,
            "range_min": 0.05,
            "range_max": 10.0,
            "time_increment": 0.001,
            "scan_time": 0.1,
            "ranges": [1.0] * 6,
        },
        "depth": {
            "stamp": 100.01,
            "frame_id": camera_frame,
            "width": width,
            "height": height,
            "step": width * 2,
            "encoding": "16UC1",
            "is_bigendian": False,
            "data_base64": base64.b64encode(depth).decode(),
        },
        "camera_info": {
            "stamp": 100.01,
            "frame_id": "",
            "width": width,
            "height": height,
            "frame_matches_image": False,
            "dimensions_match_image": True,
            "distortion_model": "plumb_bob",
            "matrices": {
                "K": [4, 0, 3, 0, 4, 1, 0, 0, 1],
                "D": [0] * 5,
                "R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "P": [4, 0, 3, 0, 0, 4, 1, 0, 0, 0, 1, 0],
            },
        },
        "transform": {
            "source": "laser",
            "target": camera_frame,
            "stamp": 100.0,
            "matrix": [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
        },
    }
    sdk = {
        "ok": True,
        "command_dispatched": False,
        "calibration_verified": False,
        "usable_for_real_execution": False,
        "parameters": {"l_intr_p": [4, 4, 3, 1], "r_intr_p": [3, 3, 3, 1]},
    }
    return bundle, sdk


class LiveDiagnosticTests(unittest.TestCase):
    def test_native_depth_frame_can_be_compared_without_claiming_calibration(self):
        bundle, sdk = fixture()
        bundle["depth"]["frame_id"] = "camera_depth_optical_frame"
        bundle["camera_info"].update(frame_id="camera_depth_optical_frame", frame_matches_image=True)
        bundle["transform"]["target"] = "camera_depth_optical_frame"
        result = analyze(bundle, sdk, now=100.1)
        self.assertFalse(result["calibration_verified"])
        self.assertFalse(result["executable"])
        self.assertTrue(result["camera_info_frame_matches_image"])

    def test_synthetic_bundle_produces_compact_non_authoritative_comparison(self):
        bundle, sdk = fixture()
        result = analyze(bundle, sdk, now=100.1)
        self.assertFalse(result["calibration_verified"])
        self.assertFalse(result["executable"])
        self.assertFalse(result["camera_info_frame_matches_image"])
        self.assertFalse(result["scan_deskewed"])
        self.assertEqual(result["laser_valid_points"], 6)
        self.assertEqual(result["depth_distribution"]["valid_pixels"], 21)
        self.assertEqual(result["depth_distribution"]["median_m"], 1.0)
        self.assertEqual(
            result["depth_distribution"]["most_common_1cm"][0]["count"], 21
        )
        self.assertEqual(
            set(result["candidates"]), {"camera_info", "sdk_left", "sdk_right"}
        )
        self.assertNotIn("data_base64", str(result))
        self.assertNotIn("winner", result)

    def test_mismatched_authority_time_frame_and_rigid_transform_rejected(self):
        base, sdk = fixture()
        mutations = [
            lambda b: b.update(calibration_verified=True),
            lambda b: b.update(diagnostic_only=False),
            lambda b: b["depth"].update(stamp=90),
            lambda b: b["transform"].update(target="base_link"),
            lambda b: b["transform"]["matrix"][0].__setitem__(1, 1),
            lambda b: b["depth"].update(data_base64="corrupt"),
            lambda b: b["camera_info"].update(frame_matches_image=True),
        ]
        for mutation in mutations:
            bundle = copy.deepcopy(base)
            mutation(bundle)
            with self.assertRaises(ValueError):
                analyze(bundle, sdk, now=100.1)

    def test_big_endian_float_depth_is_supported(self):
        bundle, sdk = fixture()
        bundle["depth"].update(
            encoding="32FC1",
            is_bigendian=True,
            step=28,
            data_base64=base64.b64encode(
                struct.pack(">" + "f" * 21, *([1.0] * 21))
            ).decode(),
        )
        self.assertEqual(analyze(bundle, sdk, now=100.1)["laser_valid_points"], 6)

    def test_oversized_encoded_data_is_rejected_before_decode(self):
        bundle, sdk = fixture()
        bundle["depth"]["data_base64"] = "A" * 2800001
        with (
            patch(
                "calibration_live_diagnostic.base64.b64decode",
                side_effect=AssertionError("decoder called"),
            ),
            self.assertRaises(ValueError),
        ):
            analyze(bundle, sdk, now=100.1)
