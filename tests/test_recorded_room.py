import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

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
        self.assertNotIn("--max-splits", command)
        self.assertEqual(command[-2:], ["/odom", "/cmd_vel"])

    def test_rolling_segments_leave_room_for_the_active_bag(self):
        for budget in (16, 65, 512, 2048):
            with self.subTest(budget=budget):
                command = recorder.bag_command(Path("/runs/session"), ["/odom"],
                                               rolling_budget_mib=budget)
                self.assertIn("--split", command)
                split_mib = int(command[command.index("--size") + 1])
                max_splits = int(command[command.index("--max-splits") + 1])
                self.assertTrue(0 < split_mib <= 64)
                self.assertGreater(max_splits, 0)
                self.assertLess((max_splits + 1) * split_mib, budget)
                self.assertEqual(command[command.index("--min-space") + 1], "256M")

    def test_rolling_requires_roam_in_the_navigation_arguments(self):
        with TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(module_path), "--output-root", directory,
                                     "--rolling", "--", "--static-room", "--seconds", "30"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 2)
            self.assertIn(b"--rolling requires --roam", result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_interrupted_rolling_session_finalizes_the_bag_and_manifest(self):
        launch_process = recorder.launch

        def launch(directory, name, command, result=None):
            if name == "rosbag":
                self.assertIn("--max-splits", command)
                script = ("from pathlib import Path\n"
                          "import signal, sys, time\n"
                          "active = Path(sys.argv[1]) / 'sensors_0.bag.active'\n"
                          "def finish(signum, frame):\n"
                          "    active.rename(active.with_suffix(''))\n"
                          "    sys.exit(0)\n"
                          "signal.signal(signal.SIGINT, finish)\n"
                          "active.write_bytes(b'sensor samples')\n"
                          "time.sleep(30)\n")
                command = [sys.executable, "-c", script, str(directory)]
            else:
                command = [sys.executable, "-c", "import time; time.sleep(30)"]
            process = launch_process(directory, name, command, result)
            if name == "navigation":
                os.kill(os.getpid(), signal.SIGINT)
            return process

        with TemporaryDirectory() as directory:
            args = [str(module_path), "--output-root", directory, "--budget-mib", "16",
                    "--rolling", "--", "--roam"]
            previous = {event: signal.getsignal(event) for event in (signal.SIGINT, signal.SIGTERM)}
            try:
                with patch.object(sys, "argv", args), patch.object(recorder, "launch", side_effect=launch):
                    self.assertEqual(recorder.main(), 2)
            finally:
                for event, handler in previous.items():
                    signal.signal(event, handler)
            session, = Path(directory).iterdir()
            manifest = json.loads((session / "manifest.json").read_text())
            self.assertEqual(manifest["phase"], "finished")
            self.assertEqual(manifest["recording_stop"], "interrupted")
            self.assertTrue(manifest["rolling"])
            self.assertTrue(manifest["bag_finalized"])
            self.assertEqual(manifest["recorder_exit"], 0)
            self.assertEqual(manifest["bags"], [{"name": "sensors_0.bag", "bytes": 14}])


if __name__ == "__main__":
    unittest.main()
