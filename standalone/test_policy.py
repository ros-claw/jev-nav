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


def crossing_scene(latest_y, older_y, latest_stamp=10.25, older_stamp=10.0):
    rays = [{"start_base": (0.1, 0.0), "hit_base": (1.1, 2.0 * y), "stamp": stamp}
            for stamp, values in ((latest_stamp, latest_y), (older_stamp, older_y))
            for y in values]
    return {"status": "ok", "coverage": {
        "sensor_fresh": True, "scan_stamp": latest_stamp, "front_samples": 14,
        "latest_valid_samples": len(latest_y),
        "latest_valid_fraction": float(len(latest_y)) / 14,
    }, "rays": rays,
        "occupied_hits": [{"point_base": ray["hit_base"], "stamp": ray["stamp"]} for ray in rays]}


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
            "linear_x_mps": linear, "linear_y_mps": 0.0, "angular_z_radps": 0.0}


def test_lateral_motion_cannot_authorize_forward_command():
    state = healthy_state()
    state['linear_y_mps'] = 0.02
    assert output(sample(10.2), state=state)['linear'] == 0.0
    state.pop('linear_y_mps')
    assert output(sample(10.2), state=state)['linear'] == 0.0


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


def test_sparse_padded_grid_uses_measured_cross_section_spacing_not_bin_fraction():
    latest = [0.0] * 33
    for index in (0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32):
        latest[index] = 2.0
    scene = fusion_scene(10.0, 10.2, [2.0] * 33, latest)
    assert 0.42 < scene["coverage"]["latest_valid_fraction"] < 0.443
    result = assess_front(scene, 10.2)
    assert result["candidate"] is True
    assert result["max_sample_gap_m"] < 0.035
    assert result["geometric_clearance_verified"] is False


def test_recent_scans_can_fill_small_sampling_gaps_but_not_a_large_latest_gap():
    previous = [2.0 if i % 4 == 0 else 0.0 for i in range(33)]
    latest = [2.0 if i % 4 == 2 else 0.0 for i in range(33)]
    scene = fusion_scene(10.0, 10.2, previous, latest)
    result = assess_front(scene, 10.2)
    assert result['candidate'] is True
    assert result['latest_max_sample_gap_m'] > 0.035
    assert result['max_sample_gap_m'] <= 0.035
    assert result['geometric_clearance_verified'] is False
    very_sparse = [2.0 if i in (0, 1, 31, 32) else 0.0 for i in range(33)]
    assert assess_front(fusion_scene(10.0, 10.2, [2.0] * 33, very_sparse), 10.2)['candidate'] is False
    assert assess_front(fusion_scene(10.0, 10.4, previous, latest), 10.4)['candidate'] is False
    assert assess_front(fusion_scene(10.0, 10.2, previous, latest), 10.46)['candidate'] is False


@pytest.mark.parametrize("older_stamp,now,expected", [
    (10.0, 10.25, True), (9.9999, 10.25, False),
    (10.0, 10.4499, True), (10.0, 10.4501, False),
])
def test_rolling_gap_fill_has_independent_scan_and_wall_age_limits(older_stamp, now, expected):
    latest_y = [-0.15, -0.09, -0.03, 0.03, 0.09, 0.15]
    older_y = [-0.12, -0.06, 0.0, 0.06, 0.12]
    result = assess_front(crossing_scene(latest_y, older_y, older_stamp=older_stamp), now)
    assert result["candidate"] is expected


def test_supervised_sampling_can_use_bounded_older_rays_with_latest_gap_still_checked():
    latest_y = [-0.15, -0.09, -0.03, 0.03, 0.09, 0.15]
    older_y = [-0.12, -0.06, 0.0, 0.06, 0.12]
    scene = crossing_scene(latest_y, older_y, latest_stamp=10.39, older_stamp=10.0)
    assert assess_front(scene, 10.5)['candidate'] is False
    result = assess_front(scene, 10.5, latest_age_limit=0.5,
                          history_age_limit=0.65, history_span_limit=0.4)
    assert result['candidate'] is True
    assert result['geometric_clearance_verified'] is False
    assert not assess_front(scene, 10.5, history_span_limit=0.401)['candidate']
    scene = crossing_scene([-0.15, -0.03, 0.09, 0.15], older_y, latest_stamp=10.39, older_stamp=10.0)
    assert not assess_front(scene, 10.5, latest_age_limit=0.5,
                            history_age_limit=0.65, history_span_limit=0.4)['candidate']


def test_latest_scan_gap_at_limit_passes_but_over_limit_does_not():
    older_y = [-0.14 + index * 0.01 for index in range(29)]
    at_limit = crossing_scene([-0.14, -0.07, 0, 0.07, 0.14], older_y)
    above_limit = crossing_scene([-0.14, -0.07001, 0, 0.07001, 0.14], older_y)
    assert assess_front(at_limit, 10.25, half_width=0.14)["candidate"] is True
    assert assess_front(above_limit, 10.25, half_width=0.14)["candidate"] is False


@pytest.mark.parametrize("max_gap", [0.035000001, 0.04, 0.05])
def test_max_gap_cannot_relax_the_required_35mm_limit(max_gap):
    result = assess_front(sample(), 10.1, max_gap=max_gap)
    assert result["candidate"] is False
    assert result["reason"] == "invalid_evidence"


def test_default_max_gap_boundary_remains_accepted():
    assert assess_front(sample(), 10.1, max_gap=0.035)["candidate"] is True


def test_smaller_caller_age_limit_also_expires_historical_gap_fill():
    latest_y = [-0.15, -0.09, -0.03, 0.03, 0.09, 0.15]
    older_y = [-0.12, -0.06, 0.0, 0.06, 0.12]
    scene = crossing_scene(latest_y, older_y, latest_stamp=10.25, older_stamp=10.125)
    at_limit = assess_front(scene, 10.25, max_age=0.125)
    beyond_limit = assess_front(scene, 10.2501, max_age=0.125)
    assert at_limit["candidate"] is True
    assert beyond_limit["candidate"] is False
    assert beyond_limit["reason"] == "forward_coverage_incomplete"


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


@pytest.mark.parametrize("x", [-0.18, -0.1, -0.01])
def test_retained_hit_in_padded_rear_footprint_blocks(x):
    scene = sample()
    scene["rays"].append({"start_base": (x - 0.1, 0.0), "hit_base": (x, 0.0), "stamp": 9.9})
    scene["occupied_hits"].append({"point_base": (x, 0.0), "stamp": 9.9})
    result = assess_front(scene, 10.1)
    assert result["candidate"] is False
    assert result["reason"] == "forward_obstacle"


@pytest.mark.parametrize("x,y", [(-0.18, -0.15), (-0.18, 0.15), (0.1, 0.14)])
def test_narrow_sample_width_cannot_shrink_padded_obstacle_check(x, y):
    scene = sample()
    scene["rays"].append({"start_base": (x - 0.1, y), "hit_base": (x, y), "stamp": 9.9})
    scene["occupied_hits"].append({"point_base": (x, y), "stamp": 9.9})
    assert assess_front(scene, 10.1, half_width=0.05)["reason"] == "forward_obstacle"


def test_known_point_behind_padded_footprint_is_not_a_forward_obstacle():
    scene = sample()
    scene["rays"].append({"start_base": (-0.3, 0.0), "hit_base": (-0.180001, 0.0), "stamp": 9.9})
    scene["occupied_hits"].append({"point_base": (-0.180001, 0.0), "stamp": 9.9})
    assert assess_front(scene, 10.1)["candidate"] is True


def test_rear_footprint_hit_blocks_after_history_coverage_window():
    scene = sample(now=10.25)
    scene["rays"].append({"start_base": (-0.2, 0.0), "hit_base": (-0.1, 0.0), "stamp": 9.7})
    scene["occupied_hits"].append({"point_base": (-0.1, 0.0), "stamp": 9.7})
    result = assess_front(scene, 10.25)
    assert result["candidate"] is False
    assert result["reason"] == "forward_obstacle"


@pytest.mark.parametrize("coverage", [None, [], "invalid", {}])
def test_malformed_coverage_returns_invalid_evidence_instead_of_raising(coverage):
    scene = sample()
    scene["coverage"] = coverage
    assert assess_front(scene, 10.1)["candidate"] is False


def test_a_malformed_ray_cannot_be_silently_skipped():
    scene = sample()
    scene["rays"][0]["hit_base"] = (float("nan"), 0.0)
    assert assess_front(scene, 10.1)["candidate"] is False


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), 10 ** 400])
def test_shared_hit_coordinates_still_require_finite_numeric_values(invalid):
    scene = sample()
    shared_point = (invalid, 0.1)
    scene["rays"][0]["hit_base"] = shared_point
    scene["occupied_hits"][0]["point_base"] = shared_point
    assert assess_front(scene, 10.1)["reason"] == "invalid_evidence"


def test_equal_but_distinct_hit_must_not_accept_boolean_as_a_coordinate():
    scene = sample()
    scene["rays"][0]["hit_base"] = (1.0, 0.1)
    scene["occupied_hits"][0]["point_base"] = (True, 0.1)
    assert assess_front(scene, 10.1)["reason"] == "invalid_evidence"


def test_old_hit_stamp_must_be_valid_even_if_equal_to_the_ray_stamp():
    scene = sample()
    scene["rays"].append({"start_base": (0.1, 1.0), "hit_base": (1.0, 2.0), "stamp": 1.0})
    scene["occupied_hits"].append({"point_base": (1.0, 2.0), "stamp": True})
    assert assess_front(scene, 10.1)["reason"] == "invalid_evidence"


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
