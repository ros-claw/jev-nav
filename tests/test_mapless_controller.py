import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "worker" / "limo_mapless_controller.py"
spec = importlib.util.spec_from_file_location("limo_mapless_controller", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sensors(stamp):
    return {
        "scan_at": stamp,
        "status_at": stamp,
        "odom_at": stamp,
        "depth_at": stamp,
        "front": 3.0,
        "left": 2.0,
        "right": 2.0,
        "rear": 2.0,
        "depth_min": 1.0,
        "depth_valid_ratio": 0.2,
        "error_code": 0,
        "motion_mode": 0,
        "battery_voltage": 12.0,
        "linear_x_mps": 0.0,
        "angular_z_radps": 0.0,
    }


def test_new_model_choice_changes_direction_without_a_stop():
    controller = module.MaplessController(started_at=10.0, deadline=20.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    forward = controller.tick(sensors(10.2), now=10.2)
    assert forward[0] > 0 and forward[1] == 0
    assert controller.set_decision("left", observed_at=10.21, now=10.25)
    left = controller.tick(sensors(10.3), now=10.3)
    assert left[0] > 0 and left[1] > 0


def test_stale_sensor_or_obstacle_stops_and_fresh_model_can_replan():
    controller = module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    assert controller.tick(sensors(10.2), now=10.2)[0] > 0
    blocked = sensors(10.3)
    blocked["front"] = 0.3
    assert controller.tick(blocked, now=10.3) == (0.0, 0.0)
    assert controller.tick(sensors(10.3), now=11.2) == (0.0, 0.0)
    assert controller.set_decision("left", observed_at=11.2, now=11.2)
    assert controller.tick(sensors(11.3), now=11.3)[1] > 0


def test_expired_or_replayed_decision_does_not_replace_newer_decision():
    controller = module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.1)
    assert not controller.set_decision("left", observed_at=10.0, now=10.2)
    assert not controller.set_decision("left", observed_at=10.2, now=19.0)
    assert controller.tick(sensors(19.0), now=19.0) == (0.0, 0.0)


def test_direction_reversal_checks_the_previous_turn_direction():
    controller = module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("left", observed_at=10.0, now=10.0)
    for now in (10.2, 10.4, 10.6):
        linear, angular = controller.tick(sensors(now), now=now)
    assert linear > 0 and angular > 0
    assert controller.set_decision("right", observed_at=10.61, now=10.61)
    observation = sensors(10.65)
    observation["left"] = 0.2
    observation["angular_z_radps"] = angular
    assert controller.tick(observation, now=10.65) == (0.0, 0.0)


def test_front_obstacle_does_not_keep_forward_command_during_a_turn():
    controller = module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    assert controller.tick(sensors(10.2), now=10.2)[0] > 0
    assert controller.set_decision("left", observed_at=10.21, now=10.21)
    observation = sensors(10.25)
    observation["front"] = 0.3
    observation["linear_x_mps"] = 0.06
    assert controller.tick(observation, now=10.25) == (0.0, 0.0)


def test_turn_requires_clearance_behind_the_body():
    controller = module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("left", observed_at=10.0, now=10.0)
    observation = sensors(10.2)
    observation["rear"] = 0.2
    assert controller.tick(observation, now=10.2) == (0.0, 0.0)


def test_forward_requires_clearance_at_the_front_corners():
    controller = module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("forward", observed_at=10.0, now=10.0)
    observation = sensors(10.2)
    observation["left"] = 0.2
    assert controller.tick(observation, now=10.2) == (0.0, 0.0)


def test_turn_does_not_sweep_through_a_close_front_obstacle():
    controller = module.MaplessController(started_at=10.0, deadline=30.0)
    assert controller.set_decision("left", observed_at=10.0, now=10.0)
    observation = sensors(10.2)
    observation["front"] = 0.18
    assert controller.tick(observation, now=10.2) == (0.0, 0.0)


def test_unexpected_measured_speed_stops_the_controller():
    for key, value in (("linear_x_mps", 0.5), ("angular_z_radps", 1.0)):
        controller = module.MaplessController(started_at=10.0, deadline=30.0)
        assert controller.set_decision("forward", observed_at=10.0, now=10.0)
        observation = sensors(10.2)
        observation[key] = value
        assert controller.tick(observation, now=10.2) == (0.0, 0.0)
