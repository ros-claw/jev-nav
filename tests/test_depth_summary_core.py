import importlib.util
import struct
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "src/limo_ros_mcp/workers/depth_summary_core.py"
spec = importlib.util.spec_from_file_location("depth_summary_core", path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_16bit_depth_uses_meters_and_counts_unknown():
    data = struct.pack("<6H", 1000, 0, 2000, 1500, 3000, 2500)
    result = core.summarize(data, 3, 2, 6, "16UC1", False)
    assert result["sectors"]["left"]["minimum_m"] == 1.0
    assert result["sectors"]["center"]["minimum_m"] == 3.0
    assert result["sectors"]["center"]["valid_ratio"] == 0.5
    assert result["sectors"]["right"]["minimum_m"] == 2.0
    assert result["safety_clearance_verified"] is False


def test_big_endian_float_and_invalid_values():
    data = struct.pack(">6f", float("nan"), 0.0, 2.0, float("inf"), 1.5, 30.0)
    result = core.summarize(data, 3, 2, 12, "32FC1", True)
    assert result["sectors"]["left"]["minimum_m"] is None
    assert result["sectors"]["left"]["valid_ratio"] == 0
    assert result["sectors"]["center"]["minimum_m"] == 1.5
    assert result["sectors"]["right"]["minimum_m"] == 2.0


def test_vertical_bands_keep_near_floor_separate_from_upper_scene():
    data = struct.pack("<9H", 3000, 3000, 3000, 2000, 2000, 2000, 500, 500, 500)
    result = core.summarize(data, 3, 3, 6, "16UC1", False)
    assert result["bands"]["top"]["center"]["minimum_m"] == 3.0
    assert result["bands"]["bottom"]["center"]["minimum_m"] == 0.5
    assert result["bands"]["bottom"]["center"]["valid_ratio"] == 1


@pytest.mark.parametrize(
    "width,height,step,encoding,data",
    [
        (3, 2, 5, "16UC1", bytes(12)),
        (3, 2, 6, "16UC1", bytes(11)),
        (3, 2, 6, "rgb8", bytes(12)),
        (5000, 5000, 10000, "16UC1", bytes(1)),
    ],
)
def test_invalid_image_fails_closed(width, height, step, encoding, data):
    with pytest.raises(ValueError):
        core.summarize(data, width, height, step, encoding, False)
