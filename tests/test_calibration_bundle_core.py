"""Synthetic diagnostic capture: never a calibration or motion authorization."""

import base64
import importlib.util
import sys
from pathlib import Path

import pytest

from tests.test_local_observation_core import bundle

source = Path(__file__).resolve().parents[1] / "src/limo_ros_mcp/workers/calibration_bundle_core.py"
sys.path.insert(0, str(source.parent))
spec = importlib.util.spec_from_file_location("calibration_bundle_core", source)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def fixture():
    scan, depth, info, transform, _ = bundle()
    scan.update(time_increment=0.001, scan_time=0.1)
    depth["frame_id"] = "camera_color_optical_frame"
    info.update(frame_id="", distortion_model="plumb_bob")
    return scan, depth, info, transform


def test_mismatched_info_frame_is_preserved_as_untrusted_diagnostic():
    scan, depth, info, transform = fixture()
    output = core.make_diagnostic(scan, depth, info, transform, now=100.1)
    assert output["depth"]["frame_id"] == "camera_color_optical_frame"
    assert output["camera_info"]["frame_id"] == ""
    assert output["camera_info"]["frame_matches_image"] is False
    assert base64.b64decode(output["depth"]["data_base64"]) == depth["data"]
    assert output["transform"]["source"] == "laser"
    assert output["transform"]["target"] == "camera_color_optical_frame"
    assert output["diagnostic_only"] is True
    assert output["calibration_verified"] is False
    assert output["usable_for_real_execution"] is False
    assert output["deskewed"] is False


def test_native_depth_frame_is_preserved_as_untrusted_diagnostic():
    scan, depth, info, transform = fixture()
    depth["frame_id"] = "camera_depth_optical_frame"
    info["frame_id"] = "camera_depth_optical_frame"
    output = core.make_diagnostic(scan, depth, info, transform, now=100.1)
    assert output["camera_info"]["frame_matches_image"] is True
    assert output["transform"]["target"] == "camera_depth_optical_frame"
    assert output["calibration_verified"] is False
    assert output["usable_for_real_execution"] is False


def test_laser_to_base_transform_is_preserved_only_as_unverified_diagnostic():
    scan, depth, info, transform = fixture()
    output = core.make_diagnostic(scan, depth, info, transform, now=100.1, scan_to_base=transform)
    assert output["base_transform"]["source"] == "laser"
    assert output["base_transform"]["target"] == "base_link"
    assert output["base_transform"]["stamp"] == scan["stamp"]
    assert output["transform_verified"] is False


def test_invalid_laser_to_base_transform_is_rejected():
    scan, depth, info, transform = fixture()
    bad = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    with pytest.raises(ValueError):
        core.make_diagnostic(scan, depth, info, transform, now=100.1, scan_to_base=bad)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s, d, i, t: d.update(stamp=99.0),
        lambda s, d, i, t: i.update(stamp=99.0),
        lambda s, d, i, t: d.update(data=b"bad"),
        lambda s, d, i, t: d.update(is_bigendian=2),
        lambda s, d, i, t: d.update(frame_id="unrecognized_camera"),
        lambda s, d, i, t: s.update(time_increment=-1),
        lambda s, d, i, t: s.update(ranges=[1.0] * 10001),
        lambda s, d, i, t: i.update(K=[float("nan")] * 9),
        lambda s, d, i, t: t[0].__setitem__(0, -1),
    ],
)
def test_malformed_data_and_reflected_transform_are_rejected(mutation):
    scan, depth, info, transform = fixture()
    mutation(scan, depth, info, transform)
    with pytest.raises(ValueError):
        core.make_diagnostic(scan, depth, info, transform, now=100.1)
