"""Synthetic temporal summary; repeated frames never certify calibration."""

import copy
import unittest

from calibration_live_diagnostic import analyze
from calibration_series import summarize
from tests.test_calibration_live_diagnostic import fixture


def frame(stamp):
    bundle, sdk = fixture()
    bundle["scan"]["stamp"] = stamp
    bundle["depth"]["stamp"] = stamp + 0.01
    bundle["camera_info"]["stamp"] = stamp + 0.01
    bundle["transform"]["stamp"] = stamp
    return analyze(bundle, sdk, now=stamp + 0.1)


class CalibrationSeriesTests(unittest.TestCase):
    def test_correlated_frames_produce_diagnostic_summary_only(self):
        result = summarize([frame(100), frame(101), frame(102)])
        self.assertEqual(result["successful_frames"], 3)
        self.assertEqual(result["temporal_status"], "summarized")
        self.assertFalse(result["executable"])
        self.assertFalse(result["calibration_verified"])
        self.assertFalse(result["independent_accuracy_verified"])
        self.assertNotIn("winner", result)

    def test_partial_failure_and_insufficient_frames_are_explicit(self):
        result = summarize([frame(100), {"status": "unavailable", "executable": False}])
        self.assertEqual(result["successful_frames"], 1)
        self.assertEqual(result["failed_frames"], 1)
        self.assertEqual(result["temporal_status"], "insufficient_frames")

    def test_duplicate_time_and_false_authority_are_rejected(self):
        first = frame(100)
        duplicate = copy.deepcopy(first)
        with self.assertRaises(ValueError):
            summarize([first, duplicate])
        second = frame(101)
        second["executable"] = True
        with self.assertRaises(ValueError):
            summarize([first, second])

    def test_sequence_size_is_bounded(self):
        with self.assertRaises(ValueError):
            summarize([])
        with self.assertRaises(ValueError):
            summarize([frame(100)] * 6)
