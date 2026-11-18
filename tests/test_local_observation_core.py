"""Pure, offline tests for a bounded read-only local observation bundle."""

import base64
import importlib.util
import struct
from pathlib import Path

import pytest

source = Path(__file__).resolve().parents[1] / "src/limo_ros_mcp/workers/local_observation_core.py"
spec = importlib.util.spec_from_file_location("local_observation_core", source)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def bundle():
    scan = {
        "stamp": 100.0,
        "frame_id": "laser",
        "angle_min": 0.0,
        "angle_increment": 0.1,
        "range_min": 0.1,
        "range_max": 10.0,
        "ranges": [1.0, float("inf"), float("nan")],
    }
    depth = {
        "stamp": 100.05,
        "frame_id": "optical",
        "width": 3,
        "height": 2,
        "step": 6,
        "encoding": "16UC1",
        "is_bigendian": False,
        "data": struct.pack("<6H", 0, 1000, 0, 0, 0, 0),
    }
    info = {
        'stamp': 100.05,
        "frame_id": "optical",
        "width": 3,
        "height": 2,
        "K": [1, 0, 1, 0, 1, 1, 0, 0, 1],
        "D": [0, 0, 0, 0, 0],
        "R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "P": [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        "binning_x": 0,
        "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "width": 0, "height": 0, "do_rectify": False},
    }
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return scan, depth, info, identity, identity


def test_valid_bundle_preserves_raw_depth_and_marks_unverified():
    scan, depth, info, scan_tf, depth_tf = bundle()
    result = core.make_bundle(scan, depth, info, scan_tf, depth_tf, now=100.1)
    assert base64.b64decode(result["depth"]["data_base64"]) == depth["data"]
    assert result["scan"]["ranges"] == [1.0, None, None]
    assert result["depth"]["camera_info"]["fx"] == 1
    assert result["depth"]["transform"]["source"] == "optical"
    assert result["calibration_verified"] is False
    assert result["usable_for_real_execution"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s, d, i: i.update(frame_id="other"),
        lambda s, d, i: i.update(stamp=99.0),
        lambda s, d, i: i.update(stamp=100.4),
        lambda s, d, i: i.update(width=640),
        lambda s, d, i: i.update(D=[0, 0.01, 0, 0, 0]),
        lambda s, d, i: i.update(K=[0] * 9),
        lambda s, d, i: i.update(P=[0] * 12),
        lambda s, d, i: i.update(R=[1, -.001, .003, .001, 1, .006, -.003, -.006, 1]),
        lambda s, d, i: i.update(binning_x=2),
        lambda s, d, i: i["roi"].update(x_offset=1),
        lambda s, d, i: i["roi"].update(do_rectify=True),
        lambda s, d, i: d.update(data=b"incomplete"),
        lambda s, d, i: d.update(encoding="rgb8"),
        lambda s, d, i: d.update(stamp=100.4),
        lambda s, d, i: s.update(stamp=97.0),
    ],
)
def test_malformed_stale_or_mismatched_inputs_are_rejected(mutation):
    scan, depth, info, scan_tf, depth_tf = bundle()
    mutation(scan, depth, info)
    with pytest.raises(ValueError):
        core.make_bundle(scan, depth, info, scan_tf, depth_tf, now=100.1)
