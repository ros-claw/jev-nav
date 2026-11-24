import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

module_path = Path(__file__).resolve().parents[1] / "standalone" / "record_navigation.py"
spec = importlib.util.spec_from_file_location("record_navigation", str(module_path))
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


class RecordingTest(unittest.TestCase):
    def test_subscribes_to_the_command_before_a_relay_advertises_it(self):
        topics, cameras = recorder.recording_topics("limo_native_probe_000000000001", ["/visual_odom"])
        self.assertIn("/cmd_vel", topics)
        self.assertIn("/visual_odom", topics)
        self.assertIn("/limo_native_probe_000000000001/rolling_costmap/costmap/snapshot/fresh_grid", topics)
        self.assertNotIn("/camera/depth/image_raw", topics)
        self.assertTrue(all(destination in topics for _, _, destination in cameras))

    def test_compact_recording_keeps_motion_and_sparse_depth(self):
        namespace = "limo_native_probe_000000000001"
        topics, cameras = recorder.recording_topics(namespace, [])
        required = {"/cmd_vel", "/odom", "/scan", "/limo_status", "/rosclaw_limo/command_watchdog",
                    "/tf_static", "/rtabmap/mapGraph", "/camera/depth/camera_info",
                    "/" + namespace + "/tf", "/" + namespace + "/recording/depth"}
        self.assertTrue(required.issubset(topics))
        self.assertFalse({"/imu", "/camera/color/camera_info", "/rtabmap/grid_map",
                          "/" + namespace + "/recording/color"}.intersection(topics))
        self.assertEqual(cameras, (("/camera/depth/image_raw", "0.5",
                                    "/" + namespace + "/recording/depth"),))
        self.assertIn("/imu", recorder.recording_topics(namespace, ["/imu"])[0])

    def test_budget_counts_open_and_finalized_segments_without_deleting_either(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "sensors_0.bag").write_bytes(b"01234")
            (path / "sensors_1.bag.active").write_bytes(b"56789")
            (path / "timeline.jsonl").write_bytes(b"motion")
            self.assertEqual(recorder.bag_bytes(path), 10)
            self.assertEqual(recorder.storage_problem(path, budget_bytes=10), "bag_budget_reached")
            self.assertEqual(len(list(path.iterdir())), 3)

    def test_recorder_process_is_the_bag_writer_so_wait_includes_file_closure(self):
        command = recorder.bag_command(Path("/runs/session"), ["/odom", "/cmd_vel"])
        self.assertEqual(command[:3], ["rosrun", "rosbag", "record"])
        self.assertEqual(command[command.index("-O") + 1], "/runs/session/sensors")
        self.assertEqual(command[-2:], ["/odom", "/cmd_vel"])


if __name__ == "__main__":
    unittest.main()
