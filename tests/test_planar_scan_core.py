"""Read-only scan and TF payload tests; no camera or actuator is required."""

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.test_calibration_bundle_core import fixture

source = Path(__file__).resolve().parents[1] / "src/limo_ros_mcp/workers/planar_scan_core.py"
sys.path.insert(0, str(source.parent))
spec = importlib.util.spec_from_file_location("planar_scan_core", source)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_scan_only_payload_preserves_unknown_and_never_authorizes_motion():
    scan, _depth, _info, transform = fixture()
    output = core.make_planar_scan(scan, transform, now=100.1)
    assert output["scan"]["ranges"] == [1.0, None, None]
    assert output["scan"]["transform"]["source"] == "laser"
    assert output["scan"]["transform"]["target"] == "base_link"
    assert output["diagnostic_only"] is True
    assert output["usable_for_real_execution"] is False
    assert output["transform_verified"] is False
    assert "depth" not in output and "camera_info" not in output


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s, t: s.update(stamp=80),
        lambda s, t: s.update(angle_increment=0),
        lambda s, t: s.update(ranges=[1.0] * 10001),
        lambda s, t: s.update(time_increment=-1),
        lambda s, t: t[0].__setitem__(0, -1),
    ],
)
def test_bad_or_stale_scan_and_transform_are_rejected(mutation):
    scan, _depth, _info, transform = fixture()
    mutation(scan, transform)
    with pytest.raises(ValueError):
        core.make_planar_scan(scan, transform, now=100.1)
