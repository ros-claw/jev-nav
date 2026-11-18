"""Read-only SDK calibration service inspection with synthetic responses."""

import math
import subprocess

import pytest
import yaml

from limo_ros_mcp.peripherals import PeripheralInspector


def parameters():
    return {
        "success": True,
        "message": "",
        "l_intr_p": [478.0, 479.0, 320.0, 200.0],
        "r_intr_p": [500.0, 501.0, 321.0, 240.0],
        "r2l_r": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
        "r2l_t": [0.01, 0, 0],
        "l_k": [0, 0, 0, 0, 0],
        "r_k": [0, 0, 0, 0, 0],
    }


def test_sdk_parameters_use_one_fixed_getter_and_do_not_claim_calibration():
    calls = []

    def run(command, timeout):
        calls.append((command, timeout))
        return 0, yaml.safe_dump(parameters()), ""

    result = PeripheralInspector(text_runner=run).dabai_camera_parameters()
    assert calls == [(["rosservice", "call", "/camera/get_camera_params"], 5.0)]
    assert result["ok"] is True
    assert result["parameters"]["l_intr_p"][0] == 478.0
    assert result["parameters"]["r_intr_p"][0] == 500.0
    assert result["calibration_verified"] is False
    assert result["usable_for_real_execution"] is False
    assert result["command_dispatched"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(success=False),
        lambda p: p.pop("r2l_r"),
        lambda p: p.update(l_intr_p=[1, 2]),
        lambda p: p.update(r_intr_p=[1, 2, math.nan, 4]),
        lambda p: p.update(r2l_t=[True, 0, 0]),
    ],
)
def test_invalid_service_results_cannot_be_used(mutation):
    response = parameters()
    mutation(response)
    inspector = PeripheralInspector(
        text_runner=lambda _command, _timeout: (0, yaml.safe_dump(response), "")
    )
    result = inspector.dabai_camera_parameters()
    assert result["ok"] is False
    assert result["usable_for_real_execution"] is False
    assert "parameters" not in result


@pytest.mark.parametrize(
    "failure",
    [
        lambda _command, _timeout: (1, "", "service unavailable"),
        lambda _command, _timeout: (0, "broken: [", ""),
        lambda _command, _timeout: (0, "x" * 20000, ""),
        lambda _command, _timeout: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("rosservice", 5)
        ),
    ],
)
def test_unavailable_or_oversized_service_fails_closed(failure):
    result = PeripheralInspector(text_runner=failure).dabai_camera_parameters()
    assert result["ok"] is False
    assert result["command_dispatched"] is False


def test_caller_cannot_choose_arbitrary_service():
    with pytest.raises(TypeError):
        PeripheralInspector().dabai_camera_parameters(service="/motor/command")
