"""Protocol tests for the fixed read-only diagnostic worker."""

import json
import subprocess

import pytest

from limo_ros_mcp.roscli import RosCliReadOnlyClient
from tests.test_calibration_bundle_core import core, fixture


def response():
    scan, depth, info, transform = fixture()
    result = core.make_diagnostic(scan, depth, info, transform, now=100.1)
    result.update(ok=True, protocol="rosclaw.limo.calibration-diagnostic.v1")
    return result


def test_diagnostic_cli_only_invokes_fixed_worker(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(response()), stderr="")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("limo_ros_mcp.roscli.Path.is_file", lambda _path: True)
    monkeypatch.setattr("subprocess.run", run)
    output = RosCliReadOnlyClient({}).calibration_diagnostic()
    assert output["diagnostic_only"] is True
    assert output["usable_for_real_execution"] is False
    assert calls[0][0][1].endswith("workers/limo_calibration_diagnostic_worker.py")
    assert json.loads(calls[0][1]["input"]) == {
        "protocol": "rosclaw.limo.calibration-diagnostic.v1",
        "timeout_sec": 5.0,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(diagnostic_only=False),
        lambda r: r.update(calibration_verified=True),
        lambda r: r.update(usable_for_real_execution=True),
        lambda r: r["depth"].update(data_base64="bad"),
        lambda r: r["transform"].update(target="base_link"),
    ],
)
def test_diagnostic_cli_rejects_false_authority_or_malformed_payload(monkeypatch, mutate):
    output = response()
    mutate(output)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("limo_ros_mcp.roscli.Path.is_file", lambda _path: True)
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(output), stderr=""
        ),
    )
    with pytest.raises(RuntimeError):
        RosCliReadOnlyClient({}).calibration_diagnostic()
