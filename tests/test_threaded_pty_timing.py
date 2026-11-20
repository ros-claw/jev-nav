"""Timing assertions for the isolated PTY test, without ROS or devices."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'standalone'))
from threaded_relay_pty_check import check_expiry_during_pause


class ExpiryTimingTest(unittest.TestCase):
    def test_expiry_stop_during_pause(self):
        check_expiry_during_pause({'start': 10.0, 'end': 10.3}, 10.1, 10.14,
                                  [(10.05, .02, 0, 'relay'), (10.16, 0, 0, 'relay')], 10.1)

    def test_stop_only_after_pause_is_rejected(self):
        with self.assertRaises(RuntimeError):
            check_expiry_during_pause({'start': 10.0, 'end': 10.3}, 10.1, 10.31, [], 10.1)

    def test_late_nonzero_is_rejected(self):
        with self.assertRaises(RuntimeError):
            check_expiry_during_pause({'start': 10.0, 'end': 10.3}, 10.1, 10.14,
                                      [(10.19, .02, 0, 'relay')], 10.1)
        with self.assertRaises(RuntimeError):
            check_expiry_during_pause({'start': 10.0, 'end': 10.3}, 10.1, 10.14, [], 10.19)


if __name__ == '__main__':
    unittest.main()
