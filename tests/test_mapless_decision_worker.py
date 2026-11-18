import base64
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "worker" / "limo_mapless_decision.py"
spec = importlib.util.spec_from_file_location("limo_mapless_decision", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_decision_receives_image_and_live_velocity():
    request = {
        "image_b64": base64.b64encode(b"\xff\xd8jpeg").decode(),
        "captured_at": 100.0,
        "laser": {"front_min_m": 3.0},
        "base": {"linear_x_mps": 0.1, "angular_z_radps": 0.2},
    }

    def transport(payload, key, timeout):
        assert payload["model"] == "gpt-6-luna"
        assert payload["input"][0]["content"][2]["data"]["base"]["linear_x_mps"] == 0.1
        assert key == "secret"
        return {"answers": [{"name": "next_action", "type": "choice", "choice": "left"}]}

    answer = module.decide(request, "secret", transport=transport, clock=lambda: 101.0)
    assert answer == {"choice": "left", "captured_at": 100.0, "ok": True}


def test_outdated_or_invalid_responses_do_not_update_the_control_target():
    request = {
        "image_b64": base64.b64encode(b"\xff\xd8jpeg").decode(),
        "captured_at": 100.0,
        "laser": {},
        "base": {},
    }
    assert (
        module.decide(request, "secret", transport=lambda *a: {}, clock=lambda: 110.0)["ok"]
        is False
    )
    assert (
        module.decide(
            request,
            "secret",
            transport=lambda *a: {
                "answers": [{"name": "next_action", "type": "choice", "choice": "back"}]
            },
            clock=lambda: 101.0,
        )["ok"]
        is False
    )
