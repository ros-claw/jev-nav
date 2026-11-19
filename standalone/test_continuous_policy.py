"""Pure synthetic tests; no ROS, model network, relay, or robot."""
import copy

import pytest
from test_forward_trial import Clock, Source


def module():
    try:
        import continuous_policy
        return continuous_policy
    except ImportError:
        pytest.fail("continuous intent policy is not implemented", pytrace=False)


class Gate(object):
    def __init__(self, stamp, accepted=True, reason="ok"):
        self.stamp, self.accepted, self.reason = stamp, accepted, reason
        self.calls = 0

    def __call__(self, sample, now):
        self.calls += 1
        return {"candidate": self.accepted, "reason": self.reason,
                self.stamp: sample[self.stamp], "geometric_clearance_verified": False}


def make_policy(start=100.0, **kwargs):
    front = kwargs.pop("front_check", Gate("scan_at"))
    depth = kwargs.pop("depth_check", Gate("depth_at"))
    return module().ContinuousPolicy(start, epoch=1, front_check=front, depth_check=depth, **kwargs)


def decision(stamp=100.0, choice="forward", epoch=1, token="a" * 64, pose=(0.0, 0.0, 0.0)):
    return {"ok": True, "choice": choice, "epoch": epoch, "captured_at": stamp,
            "image_at": stamp, "scan_at": stamp, "image_pose": pose, "observation_id": token}


def tick(policy, clock, source, seconds=0.05, update=None, register=True):
    if (register and not policy.latched and isinstance(update, dict)
            and update.get("choice") == "forward"):
        policy.register_request(update, clock.now())
    clock.t += seconds
    return policy.step(source.snapshot(), update, clock.now())


def test_continuous_forward_reuses_fresh_decision_without_segment_restart():
    clock = Clock()
    source = Source(clock)
    policy = make_policy()
    waiting = policy.step(source.snapshot(), None, clock.now())
    assert waiting["linear"] == 0 and not waiting["latched"]
    values = []
    for index in range(36):
        result = tick(policy, clock, source, update=decision() if index == 0 else None)
        values.append(result["linear"])
        assert not result["latched"] and result["angular"] == 0
        assert result["intent_only"] is True and result["physical_stop_verified"] is False
        assert result["geometric_clearance_verified"] is False
        assert clock.now() < result["valid_until"] <= clock.now() + 0.15
    assert all(0 < value <= 0.02 for value in values)
    assert values[-1] == pytest.approx(0.02)
    assert all(0 <= b - a <= 0.00201 for a, b in zip(values, values[1:]))
    assert clock.now() > 101.5  # deliberately exceeds the old 0.3-second segments


def test_newer_forward_decision_replaces_old_without_stopping():
    clock = Clock()
    source = Source(clock)
    policy = make_policy(decision_ttl_sec=1.0)
    policy.step(source.snapshot(), None, clock.now())
    tick(policy, clock, source, update=decision())
    for _ in range(10):
        result = tick(policy, clock, source)
    assert result["linear"] > 0
    current = clock.now()
    update = decision(current, token="b" * 64)
    result = tick(policy, clock, source, update=update)
    assert result["linear"] > 0 and not result["latched"]
    for _ in range(10):
        result = tick(policy, clock, source)
    assert result["linear"] > 0 and not result["latched"]


@pytest.mark.parametrize("choice", ["stop", "left", "right"])
def test_new_nonforward_decision_stops_and_latches_even_if_old(choice):
    clock = Clock()
    source = Source(clock)
    policy = make_policy()
    policy.step(source.snapshot(), None, clock.now())
    tick(policy, clock, source, update=decision())
    result = tick(policy, clock, source, update=decision(choice=choice))
    assert result["linear"] == 0 and result["latched"]
    assert tick(policy, clock, source, update=decision(clock.now(), token="b" * 64))["linear"] == 0


def test_new_obstacle_or_sensor_loss_stops_and_never_restarts():
    for fault in ("obstacle", "missing"):
        clock = Clock()
        source = Source(clock)
        front = Gate("scan_at")
        policy = make_policy(front_check=front)
        policy.step(source.snapshot(), None, clock.now())
        tick(policy, clock, source, update=decision())
        if fault == "obstacle":
            front.accepted, front.reason = False, "forward_obstacle"
            stopped = tick(policy, clock, source)
        else:
            clock.t += 0.05
            stopped = policy.step(None, None, clock.now())
        assert stopped["linear"] == 0 and stopped["latched"]
        front.accepted = True
        assert tick(policy, clock, source, update=decision(clock.now(), token="b" * 64))["linear"] == 0


def test_front_veto_locks_before_another_depth_check():
    clock = Clock()
    source = Source(clock)
    front, depth = Gate("scan_at"), Gate("depth_at")
    policy = make_policy(front_check=front, depth_check=depth)
    policy.step(source.snapshot(), None, clock.now())
    tick(policy, clock, source, update=decision())
    prior = depth.calls
    front.accepted, front.reason = False, "forward_obstacle"
    result = tick(policy, clock, source)
    assert result["linear"] == 0 and result["latched"]
    assert depth.calls == prior


def test_expired_decision_locks_zero_without_reusing_it():
    clock = Clock()
    source = Source(clock)
    policy = make_policy(decision_ttl_sec=0.3)
    policy.step(source.snapshot(), None, clock.now())
    tick(policy, clock, source, update=decision())
    for _ in range(7):
        result = tick(policy, clock, source)
    assert result["linear"] == 0 and result["latched"] and result["reason"] == "decision_expired"


def test_duration_limit_stops_without_automatic_restart():
    clock = Clock()
    source = Source(clock)
    policy = make_policy(max_duration_sec=0.4)
    policy.step(source.snapshot(), None, clock.now())
    tick(policy, clock, source, update=decision())
    for _ in range(10):
        result = tick(policy, clock, source)
    assert result["linear"] == 0 and result["latched"] and result["reason"] == "duration_limit"


@pytest.mark.parametrize("field", ["snapshot_at", "scan_at", "odom_at", "status_at", "depth_at", "color_at"])
def test_any_stale_observation_latches(field):
    clock = Clock()
    source = Source(clock)
    policy = make_policy()
    policy.step(source.snapshot(), None, clock.now())
    tick(policy, clock, source, update=decision())
    clock.t += 0.05
    stale = source.snapshot()
    stale[field] -= 2.0
    result = policy.step(stale, None, clock.now())
    assert result["linear"] == 0 and result["latched"]


@pytest.mark.parametrize("problem", ["epoch", "mode", "fault", "lateral", "reverse", "pose_jump"])
def test_invalid_state_epoch_and_odom_discontinuity_latch(problem):
    clock = Clock()
    source = Source(clock)
    policy = make_policy()
    policy.step(source.snapshot(), None, clock.now())
    tick(policy, clock, source, update=decision())
    clock.t += 0.05
    bad = source.snapshot()
    if problem == "epoch":
        bad["epoch"] = 2
    elif problem == "mode":
        bad["state"]["motion_mode"] = 1
    elif problem == "fault":
        bad["state"]["error_code"] = 1
    elif problem == "lateral":
        bad["state"]["linear_y_mps"] = 0.02
    elif problem == "reverse":
        bad["state"]["linear_x_mps"] = -0.01
    else:
        bad["pose"] = (1.0, 0.0, 0.0)
    result = policy.step(bad, None, clock.now())
    assert result["linear"] == 0 and result["latched"]


def test_invalid_gate_output_or_exception_is_fail_closed():
    clock = Clock()
    source = Source(clock)
    for bad_gate in (lambda sample, now: True,
                     lambda sample, now: {"candidate": True, "reason": "ok", "scan_at": sample["scan_at"],
                                          "geometric_clearance_verified": True},
                     lambda sample, now: 1 / 0):
        policy = make_policy(front_check=bad_gate)
        result = policy.step(source.snapshot(), None, clock.now())
        assert result["linear"] == 0 and result["latched"]


def test_future_reordered_or_mutated_decision_cannot_replace_current_one():
    clock = Clock()
    source = Source(clock)
    for mutation in (lambda d: d.update(epoch=2), lambda d: d.update(image_at=999.0),
                     lambda d: d.update(observation_id="b" * 64), lambda d: d.update(image_pose=(0.5, 0, 0))):
        clock.t = 100.0
        policy = make_policy()
        policy.step(source.snapshot(), None, clock.now())
        tick(policy, clock, source, update=decision())
        bad = copy.deepcopy(decision())
        mutation(bad)
        result = tick(policy, clock, source, update=bad)
        assert result["linear"] == 0 and result["latched"]


def test_decision_capture_pose_must_match_retained_odometry_history():
    clock = Clock()
    source = Source(clock)
    policy = make_policy()
    policy.step(source.snapshot(), None, clock.now())
    for _ in range(10):
        tick(policy, clock, source)
    result = tick(policy, clock, source, update=decision(pose=(0.08, 0.0, 0.0)))
    assert result["linear"] == 0 and result["latched"]


def test_repeated_or_late_tick_invalidates_intent_and_latches():
    for gap in (0.0, 0.21):
        clock = Clock()
        source = Source(clock)
        policy = make_policy()
        policy.step(source.snapshot(), None, clock.now())
        tick(policy, clock, source, update=decision())
        result = tick(policy, clock, source, seconds=gap)
        assert result["linear"] == 0 and result["latched"]


def test_limits_are_explicit_and_cannot_be_disabled():
    for kwargs in ({"speed_mps": 0.021}, {"max_duration_sec": 0}, {"max_duration_sec": 301},
                   {"decision_ttl_sec": 8.01}, {"max_step_gap_sec": 0.21}):
        with pytest.raises(ValueError):
            make_policy(**kwargs)


def test_unregistered_model_result_cannot_authorize_forward_intent():
    clock = Clock()
    source = Source(clock)
    policy = make_policy()
    policy.step(source.snapshot(), None, clock.now())
    result = tick(policy, clock, source, update=decision(), register=False)
    assert result["linear"] == 0 and result["latched"] and result["reason"] == "decision_unregistered"


def test_registered_request_binds_model_result_to_exact_observation():
    clock = Clock()
    source = Source(clock)
    policy = make_policy()
    assert hasattr(policy, "register_request")
    observation = decision()
    assert policy.register_request(observation, clock.now())
    policy.step(source.snapshot(), None, clock.now())
    result = tick(policy, clock, source, update=observation)
    assert result["linear"] > 0 and not result["latched"]


@pytest.mark.parametrize("delay", [2.0, 6.0])
def test_inference_can_take_seconds_while_fresh_samples_continue(delay):
    clock = Clock()
    source = Source(clock)
    policy = make_policy()
    observation = decision()
    assert policy.register_request(observation, clock.now())
    assert policy.step(source.snapshot(), None, clock.now())["linear"] == 0
    for _ in range(int(delay / 0.05)):
        waiting = tick(policy, clock, source)
        assert waiting["linear"] == 0 and not waiting["latched"]
    result = tick(policy, clock, source, update=observation, register=False)
    assert result["linear"] > 0 and not result["latched"]


def test_model_failure_is_not_treated_as_pending_decision():
    clock = Clock()
    source = Source(clock)
    policy = make_policy()
    policy.step(source.snapshot(), None, clock.now())
    tick(policy, clock, source, update=decision())
    result = tick(policy, clock, source, update={"ok": False})
    assert result["linear"] == 0 and result["latched"] and result["reason"] == "decision_invalid"


def test_initial_model_wait_rejects_uncommanded_motion_even_if_pose_returns():
    clock = Clock()
    source = Source(clock)
    policy = make_policy()
    assert policy.register_request(decision(), clock.now())
    policy.step(source.snapshot(), None, clock.now())
    clock.t += 0.05
    moving = source.snapshot()
    moving["state"]["linear_x_mps"] = 0.01
    result = policy.step(moving, None, clock.now())
    assert result["linear"] == 0 and result["latched"]


def test_new_decision_can_be_validated_from_continuous_moving_pose_history():
    clock = Clock()
    source = Source(clock)
    policy = make_policy()
    assert policy.register_request(decision(), clock.now())
    policy.step(source.snapshot(), None, clock.now())
    tick(policy, clock, source, update=decision(), register=False)
    motion_started = clock.now()
    def moving_sample():
        sample = source.snapshot()
        distance = max(0.0, clock.now() - motion_started) * 0.015
        sample["pose"] = sample["image_pose"] = sample["depth_pose"] = (distance, 0.0, 0.0)
        sample["scene"]["coverage"]["scan_pose"] = sample["pose"]
        sample["state"]["linear_x_mps"] = 0.015
        return sample
    for _ in range(20):
        clock.t += 0.05
        assert policy.step(moving_sample(), None, clock.now())["linear"] > 0
    capture = clock.now()
    pose = moving_sample()["pose"]
    update = decision(capture, token="b" * 64, pose=pose)
    assert policy.register_request(update, clock.now())
    for _ in range(40):
        clock.t += 0.05
        assert policy.step(moving_sample(), None, clock.now())["linear"] > 0
    clock.t += 0.05
    result = policy.step(moving_sample(), update, clock.now())
    assert result["linear"] > 0 and not result["latched"]


@pytest.mark.parametrize("choice", ["stop", "left", "right"])
def test_nonforward_update_locks_before_calling_sensor_gates(choice):
    clock = Clock()
    source = Source(clock)
    front, depth = Gate("scan_at"), Gate("depth_at")
    policy = make_policy(front_check=front, depth_check=depth)
    policy.step(source.snapshot(), None, clock.now())
    tick(policy, clock, source, update=decision())
    before = (front.calls, depth.calls)
    result = tick(policy, clock, source, update=decision(choice=choice))
    assert result["linear"] == 0 and result["latched"]
    assert (front.calls, depth.calls) == before


def test_real_pure_front_and_projected_depth_gates_veto_new_raw_hit():
    from projected_depth_gate import ProjectedDepthGate
    mod = module()
    assert hasattr(mod, "front_scan_check")
    clock = Clock()
    source = Source(clock)
    def projected(sample):
        sample["depth"]["corridor"] = {"stamp": sample["depth_at"], "status": "observed",
            "executable": False, "geometric_clearance_verified": False, "calibration_verified": False,
            "unobserved_space": True, "sample_count": 200, "valid_count": 150,
            "invalid_count": 50, "sample_stride": 4, "ground_count": 100,
            "obstacle_count": 0, "unclassified_corridor_count": 0, "outside_corridor_count": 50}
        return sample
    policy = make_policy(front_check=mod.front_scan_check, depth_check=ProjectedDepthGate())
    assert policy.register_request(decision(), clock.now())
    policy.step(projected(source.snapshot()), None, clock.now())
    clock.t += 0.05
    result = policy.step(projected(source.snapshot()), decision(), clock.now())
    assert result["linear"] > 0 and not result["latched"]
    clock.t += 0.05
    obstacle = projected(source.snapshot())
    obstacle["raw_scan"]["ranges"][16] = 0.195
    result = policy.step(obstacle, None, clock.now())
    assert result["linear"] == 0 and result["latched"] and result["reason"] == "front_blocked"


def test_front_adapter_rejects_malformed_or_clearance_claiming_result(monkeypatch):
    import policy as front_policy
    mod = module()
    clock = Clock()
    sample = Source(clock).snapshot()
    monkeypatch.setattr(front_policy, "assess_front", lambda scene, now:
                        {"candidate": True, "reason": "ok", "geometric_clearance_verified": True})
    assert mod.front_scan_check(sample, clock.now())["candidate"] is False


def test_front_adapter_retains_older_near_obstacle_as_a_veto():
    mod = module()
    clock = Clock()
    sample = Source(clock).snapshot()
    old = sample["scan_at"] - 0.7
    sample["scene"]["rays"].append({"start_base": (0.0, 0.0), "hit_base": (0.09, 0.0), "stamp": old})
    sample["scene"]["occupied_hits"].append({"point_base": (0.09, 0.0), "stamp": old})
    checked = mod.front_scan_check(sample, clock.now())
    assert checked["candidate"] is False and checked["reason"] == "forward_obstacle"
