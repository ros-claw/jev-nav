"""Fixed scan-only subprocess boundary, exercised with synthetic responses."""

import json
import subprocess

import pytest

from limo_ros_mcp.roscli import RosCliReadOnlyClient
from tests.test_calibration_bundle_core import fixture
from tests.test_planar_scan_core import core


def response():
    scan, _depth, _info, transform = fixture()
    output = core.make_planar_scan(scan, transform, now=100.1)
    output.update(ok=True, protocol="rosclaw.limo.planar-scan-diagnostic.v1")
    return output


def test_client_uses_fixed_read_only_worker(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(response()), stderr="")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("limo_ros_mcp.roscli.Path.is_file", lambda _path: True)
    monkeypatch.setattr("subprocess.run", run)
    result = RosCliReadOnlyClient({}).planar_scan()
    assert calls[0][0][1].endswith("workers/limo_planar_scan_worker.py")
    assert result["diagnostic_only"] is True
    assert result["usable_for_real_execution"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(diagnostic_only=False),
        lambda r: r.update(usable_for_real_execution=True),
        lambda r: r.update(transform_verified=True),
        lambda r: r["scan"]["transform"].update(target="other"),
        lambda r: r["scan"].update(ranges=[]),
    ],
)
def test_client_rejects_false_authority_or_incorrect_contract(monkeypatch, mutate):
    output = response()
    mutate(output)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("limo_ros_mcp.roscli.Path.is_file", lambda _path: True)
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, **kw: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(output), stderr=""
        ),
    )
    with pytest.raises(RuntimeError):
        RosCliReadOnlyClient({}).planar_scan()
