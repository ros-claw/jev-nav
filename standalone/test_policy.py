"""Offline regression checks for the experimental forward sampling policy."""

import math
import os
import sys

import pytest

worker = os.path.join(os.path.dirname(__file__), "..", "worker")
if not os.path.isdir(worker):
    worker = os.path.join(os.path.dirname(__file__), "..", "worktrees", "limo-mcp", "worker")
sys.path.insert(0, worker)
from limo_scan_fusion import RollingScanFusion  # noqa: E402
from policy import ForwardPolicy, assess_front  # noqa: E402


def sample(now=10.0):
    rays = []
    for i in range(33):
        crossing_y = -0.15 + i * 0.30 / 32.0
        start = (0.105, 0.0)
        scale = (2.0 - start[0]) / (0.60 - start[0])
        rays.append({"start_base": start, "hit_base": (2.0, crossing_y * scale), "stamp": now})
    hits = [{"point_base": ray["hit_base"], "stamp": ray["stamp"]} for ray in rays]
    return {"status": "ok", "coverage": {
        "sensor_fresh": True, "scan_stamp": now, "front_samples": 33,
        "latest_valid_samples": 33, "latest_valid_fraction": 1.0,
    }, "rays": rays, "occupied_hits": hits}


def scan(stamp, ranges, angle_min=-0.3, angle_increment=0.6 / 32):
    return {"stamp": stamp, "ranges": ranges, "angle_min": angle_min,
            "angle_increment": angle_increment, "range_min": 0.1, "range_max": 10.0}


def fusion_scene(first, last, old_ranges, new_ranges):
    fusion = RollingScanFusion((0.105, 0.0, 0.0))
    pose = (0.0, 0.0, 0.0)
    assert fusion.add_scan(scan(first, old_ranges), pose, first, first)
    assert fusion.add_scan(scan(last, new_ranges), pose, last, last)
    return fusion.snapshot(pose, last, last)


def healthy_state(now=10.2, linear=0.0):
    return {"stamp": now, "motion_mode": 0, "error_code": 0, "battery_voltage": 12.0,
            "linear_x_mps": linear, "angular_z_radps": 0.0}


def healthy_depth(now=10.2):
    return {"stamp": now, "valid_ratio": 0.8, "min_m": 1.5,
            "valid_grid": [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]]}


def output(scene, state=None, depth=None, decision=None):
    policy = ForwardPolicy(started_at=10.0, deadline=20.0, speed=0.03)
    return policy.step(scene, state if state is not None else healthy_state(),
                       depth if depth is not None else healthy_depth(),
                       decision if decision is not None else {"choice": "forward", "stamp": 10.0}, 10.2)


def test_fresh_front_samples_can_be_a_candidate_without_certifying_clearance():
    result = assess_front(sample(), 10.1)
    assert result["candidate"] is True
    assert result["geometric_clearance_verified"] is False


def test_missing_directional_samples_are_not_filled_by_old_rays():
    previous = [2.0] * 33
    latest = [2.0] * 33
    latest[12:21] = [0.0] * 9
    scene = fusion_scene(10.0, 10.2, previous, latest)
    assert scene["coverage"]["latest_valid_fraction"] > 0.5
    assert assess_front(scene, 10.2)["candidate"] is False


def test_latest_single_valid_ray_cannot_borrow_old_coverage():
    latest = [0.0] * 33
    latest[16] = 2.0
    scene = fusion_scene(10.0, 10.2, [2.0] * 33, latest)
    assert scene["coverage"]["latest_valid_samples"] == 1
    assert assess_front(scene, 10.2)["candidate"] is False


def test_measured_42_percent_grid_occupancy_is_rejected_even_when_rays_are_spread():
    latest = [0.0] * 33
    for index in (0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32):
        latest[index] = 2.0
    scene = fusion_scene(10.0, 10.2, [2.0] * 33, latest)
    assert 0.42 < scene["coverage"]["latest_valid_fraction"] < 0.443
    assert assess_front(scene, 10.2)["candidate"] is False


def test_cached_snapshot_is_rejected_once_latest_scan_is_stale():
    assert assess_front(sample(10.0), 10.6)["candidate"] is False


def test_unexpired_retained_hit_blocks_even_after_policy_ray_age():
    previous = [2.0] * 33
    previous[16] = 0.3
    scene = fusion_scene(10.0, 10.7, previous, [2.0] * 33)
    assert any(hit["stamp"] == 10.0 and hit["point_base"][0] < 0.6
               for hit in scene["occupied_hits"])
    assert assess_front(scene, 10.7)["candidate"] is False


def test_fresh_retained_near_hit_blocks_after_slow_forward_motion():
    fusion = RollingScanFusion((0.105, 0.0, 0.0))
    previous = [2.0] * 181
    previous[-1] = 0.1
    initial, current = (0.0, 0.0, 0.0), (0.006, 0.0, 0.0)
    assert fusion.add_scan(scan(10.0, previous, -math.pi / 2, math.pi / 180), initial, 10.0, 10.0)
    assert fusion.add_scan(scan(10.2, [2.0] * 181, -math.pi / 2, math.pi / 180), current, 10.2, 10.2)
    scene = fusion.snapshot(current, 10.2, 10.2)
    assert any(0.09 < h["point_base"][0] < 0.1 and abs(h["point_base"][1]) < 0.15
               for h in scene["occupied_hits"])
    assert assess_front(scene, 10.2)["candidate"] is False


@pytest.mark.parametrize("x", [0.0, 0.05, 0.099])
def test_known_near_forward_points_are_not_ignored(x):
    scene = sample()
    scene["rays"].append({"start_base": (x - 0.1, 0.0), "hit_base": (x, 0.0), "stamp": 9.9})
    scene["occupied_hits"].append({"point_base": (x, 0.0), "stamp": 9.9})
    assert assess_front(scene, 10.1)["candidate"] is False


def test_known_point_behind_the_base_is_not_a_front_obstacle():
    scene = sample()
    scene["rays"].append({"start_base": (-0.1, 0.0), "hit_base": (-0.01, 0.0), "stamp": 9.9})
    scene["occupied_hits"].append({"point_base": (-0.01, 0.0), "stamp": 9.9})
    assert assess_front(scene, 10.1)["candidate"] is True


@pytest.mark.parametrize("coverage", [None, [], "invalid", {}])
def test_malformed_coverage_returns_invalid_evidence_instead_of_raising(coverage):
    scene = sample()
    scene["coverage"] = coverage
    assert assess_front(scene, 10.1)["candidate"] is False


def test_a_malformed_ray_cannot_be_silently_skipped():
    scene = sample()
    scene["rays"][0]["hit_base"] = (float("nan"), 0.0)
    assert assess_front(scene, 10.1)["candidate"] is False


def test_a_ray_hit_missing_from_occupied_hits_cannot_be_treated_as_clear():
    scene = sample()
    scene["rays"][16]["hit_base"] = (0.3, 0.0)
    assert assess_front(scene, 10.1)["candidate"] is False


def test_the_policy_commands_only_forward_when_the_gates_pass():
    result = output(sample(10.2))
    assert 0 < result["linear"] <= 0.03
    assert result["angular"] == 0.0


@pytest.mark.parametrize("velocity", [-0.0001, -0.1])
def test_reverse_measured_velocity_cannot_authorize_forward_command(velocity):
    assert output(sample(10.2), state=healthy_state(linear=velocity))["linear"] == 0.0


def test_decision_older_than_session_cannot_authorize_motion():
    assert output(sample(10.2), decision={"choice": "forward", "stamp": 9.9})["linear"] == 0.0


def test_later_stop_decision_cannot_be_replaced_by_older_forward_decision():
    policy = ForwardPolicy(10.0, 20.0)
    assert policy.step(sample(10.2), healthy_state(), healthy_depth(),
                       {"choice": "stop", "stamp": 10.15}, 10.2)["linear"] == 0.0
    assert policy.step(sample(10.2), healthy_state(), healthy_depth(),
                       {"choice": "forward", "stamp": 10.10}, 10.3)["linear"] == 0.0


def test_same_timestamp_cannot_replace_a_stop_with_a_forward_choice():
    policy = ForwardPolicy(10.0, 20.0)
    assert policy.step(sample(10.2), healthy_state(), healthy_depth(),
                       {"choice": "stop", "stamp": 10.15}, 10.2)["linear"] == 0.0
    assert policy.step(sample(10.2), healthy_state(), healthy_depth(),
                       {"choice": "forward", "stamp": 10.15}, 10.3)["linear"] == 0.0


@pytest.mark.parametrize("bad", [None, [], [[0.9] * 3] * 2,
                                  [[0.9] * 3, [0.9, float("nan"), 0.9], [0.9] * 3],
                                  [[0.9] * 3, [0.9, 1.2, 0.9], [0.9] * 3],
                                  [[0.9] * 3, [0.9, 0.1, 0.9], [0.9] * 3]])
def test_each_depth_region_must_have_a_valid_sufficient_ratio(bad):
    depth = healthy_depth()
    depth["valid_grid"] = bad
    assert output(sample(10.2), depth=depth)["linear"] == 0.0


def test_measured_depth_grid_is_rejected_even_if_global_depth_ratio_looks_good():
    depth = healthy_depth()
    depth["valid_grid"] = [[0.312, 0.0, 0.185], [0.469, 0.312, 0.228], [0.168, 0.160, 0.404]]
    assert output(sample(10.2), depth=depth)["linear"] == 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, None, 0.467])
def test_bad_or_close_depth_minimum_never_moves(value):
    depth = healthy_depth()
    depth["min_m"] = value
    assert output(sample(10.2), depth=depth)["linear"] == 0.0


def test_invalid_base_state_or_stop_decision_never_moves():
    state = healthy_state()
    state["error_code"] = 1
    assert output(sample(10.2), state=state)["linear"] == 0.0
    assert output(sample(10.2), decision={"choice": "stop", "stamp": 10.0})["linear"] == 0.0
