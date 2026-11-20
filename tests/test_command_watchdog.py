"""Compile and exercise the driver timeout policy without ROS or hardware."""

import subprocess
import time
from pathlib import Path


def test_driver_watchdog_timeout_latches_and_emits_stop(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "limo-watchdog-test"
    subprocess.run(
        ["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
         "-I", str(root / "worker"),
         str(root / "tests/fixtures/limo_command_watchdog_test.cpp"), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    before = time.monotonic()
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    after = time.monotonic()
    # Relay and driver compare their monotonic timestamps on the same host.
    assert before <= float(result.stdout) <= after
