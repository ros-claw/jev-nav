import base64
import importlib.util
import socket
import urllib.error
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


def test_forward_lidar_gaps_and_unobserved_rear_are_sent_to_the_model():
    view = {"center": {"valid_rays": 20, "total_rays": 40}, "rear_valid_rays": 0}
    request = {
        "image_b64": base64.b64encode(b"\xff\xd8jpeg").decode(),
        "captured_at": 100.0,
        "laser": {"forward_view": view},
        "base": {},
    }

    def transport(payload, key, timeout):
        assert payload["input"][0]["content"][2]["data"]["laser"]["forward_view"] == view
        instructions = payload["questions"][0]["instructions"].lower()
        assert "rear" in instructions and "unknown" in instructions
        return {"answers": [{"name": "next_action", "type": "choice", "choice": "forward"}]}

    assert module.decide(request, "secret", transport=transport, clock=lambda: 101.0)["ok"]


def test_forward_trial_only_offers_forward_or_stop_and_preserves_stop():
    request = {
        "image_b64": base64.b64encode(b"\xff\xd8jpeg").decode(),
        "captured_at": 100.0,
        "laser": {},
        "base": {},
        "forward_only": True,
        "forward_segment": {"max_distance_m": 0.04, "max_speed_mps": 0.02},
    }

    def reply(choice):
        def transport(payload, key, timeout):
            assert [item["value"] for item in payload["questions"][0]["choices"]] == ["forward", "stop"]
            assert payload["input"][0]["content"][2]["data"]["forward_segment"] == request["forward_segment"]
            return {"answers": [{"name": "next_action", "type": "choice", "choice": choice}]}
        return module.decide(request, "secret", transport=transport, clock=lambda: 101.0)

    assert reply("forward")["ok"]
    assert reply("stop") == {"choice": "stop", "captured_at": 100.0, "ok": True}
    assert not reply("left")["ok"]


def test_navigation_offers_only_locally_checked_segments():
    request = {
        "image_b64": base64.b64encode(b"\xff\xd8jpeg").decode(),
        "captured_at": 100.0,
        "laser": {},
        "base": {},
        "navigation_choices": ["left", "stop"],
    }

    def reply(choice):
        def transport(payload, key, timeout):
            assert [item["value"] for item in payload["questions"][0]["choices"]] == ["left", "stop"]
            return {"answers": [{"name": "next_action", "type": "choice", "choice": choice}]}
        return module.decide(request, "secret", transport=transport, clock=lambda: 101.0)

    assert reply("left")["ok"]
    assert not reply("forward")["ok"]
    request['navigation_choices'] = ['left']
    assert not reply("left")["ok"]


def test_room_model_selects_a_route_with_a_verified_start():
    laser = {"forward_view": {"rear_valid_rays": 0, "base_hits": {"right": {"nearest_m": 0.20}}},
             "blocked_segments": {"right": "occupied", "left": "unknown"}}
    request = {
        "image_b64": base64.b64encode(b"\xff\xd8jpeg").decode(),
        "captured_at": 100.0,
        "laser": laser,
        "base": {},
        "navigation_choices": ["forward", "stop"],
        "exploration_goal": "circulate_room",
    }

    def transport(payload, key, timeout):
        assert [item["value"] for item in payload["questions"][0]["choices"]] == [
            "circulate_left", "circulate_right"]
        context = payload["input"][0]["content"][2]["data"]
        assert context["checked_start_segments"] == ["forward"]
        assert context["laser"] == laser
        return {"answers": [{"name": "next_action", "type": "choice", "choice": "circulate_left"}]}

    assert module.decide(request, "secret", transport=transport, clock=lambda: 101.0)["choice"] == "circulate_left"


def test_room_model_can_select_only_the_verified_pivot_side():
    request = {
        "image_b64": base64.b64encode(b"\xff\xd8jpeg").decode(),
        "captured_at": 100.0,
        "laser": {},
        "base": {},
        "navigation_choices": ["pivot_right", "stop"],
        "exploration_goal": "circulate_room",
    }

    def transport(payload, key, timeout):
        raise AssertionError("an only checked turn does not require a model round trip")

    assert module.decide(request, "secret", transport=transport, clock=lambda: 101.0)["choice"] == "circulate_right"


def test_confirmed_map_targets_are_choices_and_recalled_experience_stays_context():
    request = {
        "image_b64": base64.b64encode(b"\xff\xd8jpeg").decode(),
        "captured_at": 100.0, "laser": {}, "base": {},
        "navigation_choices": ["left", "right", "stop"], "exploration_goal": "circulate_room",
        "exploration_targets": [
            {"id": "8-left", "action": "left", "map_id": "room", "revision": 3,
             "anchor": 8, "x": .1, "y": .4, "visits": 1},
            {"id": "8-right", "action": "right", "map_id": "room", "revision": 3,
             "anchor": 8, "x": .1, "y": -.4, "visits": 0},
        ],
    }

    def transport(payload, key, timeout):
        assert [item["value"] for item in payload["questions"][0]["choices"]] == ["8-left", "8-right"]
        context = payload["input"][0]["content"][2]["data"]
        assert context["exploration_targets"] == request["exploration_targets"]
        side = "8-right" if context.get("historical_context") else "8-left"
        return {"answers": [{"name": "next_action", "type": "choice", "choice": side}]}

    def decide():
        return module.decide(request, "secret", transport=transport, clock=lambda: 101.0)

    assert decide()["target_id"] == "8-left"
    request["exploration_targets"][0]["checked_follow_on"] = []
    request["exploration_targets"][1]["checked_follow_on"] = ["forward", "right"]
    assert decide()["ok"]
    request["exploration_targets"][1]["checked_follow_on"] = ["not_a_route"]
    assert not decide()["ok"]
    request["exploration_targets"][1]["checked_follow_on"] = ["forward", "right"]
    request["historical_context"] = "Room anchor 8. The left approach previously stalled."
    assert decide()["target_id"] == "8-right" and decide()["choice"] == "right"
    request["exploration_targets"][1]["action"] = "forward"
    assert not decide()["ok"]


def test_transport_failures_return_only_sanitized_classification():
    request = {
        "image_b64": base64.b64encode(b"\xff\xd8jpeg").decode(),
        "captured_at": 100.0,
        "laser": {},
        "base": {},
    }
    for error, label in (
        (socket.timeout("secret socket message"), "DecisionsTimeout"),
        (urllib.error.HTTPError("https://example.invalid/private", 429, "secret response", None, None),
         "DecisionsHttp429"),
    ):
        def transport(*_):
            raise error
        assert module.decide(request, "secret", transport=transport, clock=lambda: 101.0) == {
            "choice": "stop", "ok": False, "error_class": label,
        }
