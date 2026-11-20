import copy
import unittest

from mapless_preflight import inspect_preflight


def healthy_fixture():
    return {
        "runtime": {
            "running": True,
            "privilege_separated": True,
            "supervision_state": "ARMED",
            "emergency_stop_latched": False,
            "registered_executors": ["limo.explore:REAL"],
            "robot_pack": {"loaded": True, "signature_status": "valid"},
            "ledger": {"integrity_verified": True, "write_failed": False},
            "recovery": {"required": False},
            "interaction_plane": {"real_confirmation_ready": True},
        },
        "observations": {
            "scan_stamp": 100.0,
            "odom_stamp": 100.0,
            "status_stamp": 100.0,
            "base_healthy": True,
            "transform_verified": True,
            "geometry_coverage_verified": True,
        },
        "physical_evidence": {
            "independent_stop_verified": True,
            "command_loss_stop_verified": True,
            "braking_profile_verified": True,
        },
    }


class MaplessPreflightTest(unittest.TestCase):
    def test_even_complete_untrusted_inputs_cannot_authorize_execution(self):
        result = inspect_preflight(healthy_fixture(), now=100.5)
        self.assertEqual(result["unmet_reported_conditions"], [])
        self.assertFalse(result["real_execution_authorized"])
        self.assertFalse(result["evidence_authenticated"])

    def test_neither_map_nor_amcl_is_required(self):
        result = inspect_preflight(healthy_fixture(), now=100.0)
        self.assertEqual(result["unmet_reported_conditions"], [])

    def test_each_absent_or_false_safety_fact_is_a_blocker(self):
        fixture = healthy_fixture()
        for section in fixture:
            for key, value in fixture[section].items():
                altered = copy.deepcopy(fixture)
                altered[section][key] = None
                with self.subTest(section=section, key=key, value=value):
                    self.assertTrue(
                        inspect_preflight(altered, now=100.0)[
                            "unmet_reported_conditions"
                        ]
                    )

    def test_daemon_watchdog_does_not_prove_chassis_command_loss_stop(self):
        fixture = healthy_fixture()
        fixture["runtime"]["watchdog"] = {"healthy": True}
        fixture["physical_evidence"]["command_loss_stop_verified"] = False
        self.assertIn(
            "command_loss_stop_unverified",
            inspect_preflight(fixture, now=100.0)["unmet_reported_conditions"],
        )

    def test_stale_future_boolean_and_nonfinite_stamps_are_rejected(self):
        for stamp in (95.0, 101.0, True, float("nan"), float("inf"), "100", 10**1000):
            fixture = healthy_fixture()
            fixture["observations"]["scan_stamp"] = stamp
            with self.subTest(stamp=stamp):
                self.assertIn(
                    "scan_not_fresh",
                    inspect_preflight(fixture, now=100.0)["unmet_reported_conditions"],
                )

    def test_malformed_payload_fails_closed(self):
        for value in (None, [], {"runtime": None}, "data"):
            with self.subTest(value=value):
                report = inspect_preflight(value, now=100.0)
                self.assertTrue(report["unmet_reported_conditions"])
                self.assertFalse(report["real_execution_authorized"])


if __name__ == "__main__":
    unittest.main()
