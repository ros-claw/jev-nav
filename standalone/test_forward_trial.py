"""Offline-only tests for a one-shot supervised forward calibration candidate."""

import copy
import threading
import time

import pytest
from test_policy import sample as scene_sample


def trial_module():
    try:
        import forward_trial
    except ImportError:
        pytest.fail("offline forward trial is not implemented", pytrace=False)
    return forward_trial


class Clock(object):
    def __init__(self):
        self.t = 100.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class Source(object):
    simulation_only = True

    def __init__(self, clock, mutate=None, trace_mutate=None):
        self.clock = clock
        self.mutate = mutate
        self.trace_mutate = trace_mutate
        self.calls = 0
        self.current_epoch = 1
        self.newer_observation = False

    def is_current_epoch(self, epoch):
        return self.current_epoch == epoch

    def is_current_sample(self, sample):
        return self.current_epoch == sample["epoch"] and not self.newer_observation

    def stationary_trace(self, start, end):
        stamps = [start]
        index = 1
        while start + index * 0.05 < end:
            stamps.append(start + index * 0.05)
            index += 1
        if end > start:
            stamps.append(end)
        trace = [{"stamp": stamp, "pose": (0.0, 0.0, 0.0), "linear_x_mps": 0.0, "linear_y_mps": 0.0,
                  "angular_z_radps": 0.0, "seq": i, "epoch": 1}
                 for i, stamp in enumerate(stamps)]
        return self.trace_mutate(trace) if self.trace_mutate else trace

    def snapshot(self):
        self.calls += 1
        now = self.clock.now()
        pose = (0.0, 0.0, 0.0)
        scene = scene_sample(now)
        scene["coverage"]["scan_pose"] = pose
        result = {
            "epoch": 1, "snapshot_at": now, "color_at": now, "scan_at": now,
            "status_at": now, "odom_at": now, "depth_at": now,
            "image_jpeg": b"\xff\xd8offline-frame\xff\xd9",
            "pose": pose, "image_pose": pose, "depth_pose": pose,
            "state": {"stamp": now, "motion_mode": 0, "vehicle_state": 0,
                      "control_mode": 1, "error_code": 0, "battery_voltage": 12.0,
                      "linear_x_mps": 0.0, "linear_y_mps": 0.0, "angular_z_radps": 0.0},
            "depth": {"stamp": now, "min_m": 1.5, "valid_grid": [[0.8] * 3 for _ in range(3)]},
            "raw_scan": {"stamp": now, "frame_id": "laser_link", "angle_min": -0.3,
                         "angle_increment": 0.6 / 32.0, "range_min": 0.1,
                         "range_max": 10.0, "time_increment": 0.0, "ranges": [2.0] * 33},
            "laser_extrinsics": (0.105, 0.0, 0.0),
            "scene": scene,
        }
        return self.mutate(result, self.calls) if self.mutate else result


class Model(object):
    simulation_only = True

    def __init__(self, clock, choice="forward", delay=0.0, wrong_id=False):
        self.clock, self.choice, self.delay, self.wrong_id = clock, choice, delay, wrong_id
        self.evidence = []

    def decide(self, evidence):
        self.evidence.append(copy.deepcopy(evidence))
        self.clock.t += self.delay
        return {"ok": True, "choice": self.choice,
                "observation_id": "wrong" if self.wrong_id else evidence["observation_id"]}


class Relay(object):
    simulation_only = True

    def __init__(self, clock, hard_deadline=True, fail_health_at=None, fail_send_at=None,
                 zero_on_parent_loss=True, independent_heartbeat=True, health_delay=0.0, on_health=None):
        self.clock = clock
        self.hard_deadline = hard_deadline
        self.zero_on_parent_loss = zero_on_parent_loss
        self.independent_heartbeat = independent_heartbeat
        self.health_delay = health_delay
        self.on_health = on_health
        self.fail_health_at, self.fail_send_at = fail_health_at, fail_send_at
        self.healthy_calls = 0
        self.frames = []
        self.stop_calls = 0
        self.closed = False
        self.deadline = None
        self.motion_limit = None
        self.calls_threads = []

    def arm(self, max_motion_sec):
        self.calls_threads.append(threading.current_thread().ident)
        self.motion_limit = max_motion_sec
        return {"ready_line": b"READY motion_limit_sec=0.300000\n", "simulation_only": True,
                "relative_motion_limit": self.hard_deadline,
                "latches_zero": True, "zero_on_parent_loss": self.zero_on_parent_loss,
                "independent_heartbeat": self.independent_heartbeat}

    def healthy(self):
        self.calls_threads.append(threading.current_thread().ident)
        self.healthy_calls += 1
        self.clock.t += self.health_delay
        if self.on_health is not None:
            self.on_health()
        return self.fail_health_at != self.healthy_calls

    def send_frame(self, sequence, stamp, linear, angular):
        self.calls_threads.append(threading.current_thread().ident)
        if self.fail_send_at == len(self.frames) + 1:
            raise IOError("synthetic relay failure")
        if self.deadline is None:
            self.deadline = stamp + self.motion_limit
        if stamp >= self.deadline:
            return False
        self.frames.append((sequence, stamp, linear, angular))
        return True

    def stop_and_latch(self):
        self.calls_threads.append(threading.current_thread().ident)
        self.stop_calls += 1
        return True

    def close(self):
        self.calls_threads.append(threading.current_thread().ident)
        self.closed = True


class RelativeContractRelay(Relay):
    """Offline fixture backed by the real relay's pure gate, without ROS."""
    def __init__(self, clock, ready_line=b"READY motion_limit_sec=0.300000\n"):
        Relay.__init__(self, clock)
        self.ready_line = ready_line
        self.gate = None

    def arm(self, max_motion_sec):
        import relay as relay_module
        assert max_motion_sec == 0.3
        self.gate = relay_module.RelayGate(self.clock.now(), max_motion_sec=max_motion_sec)
        return {"ready_line": self.ready_line, "simulation_only": True,
                "relative_motion_limit": True, "latches_zero": True,
                "zero_on_parent_loss": True, "independent_heartbeat": True}

    def send_frame(self, sequence, stamp, linear, angular):
        self.calls_threads.append(threading.current_thread().ident)
        frame = {"sequence": sequence, "stamp": stamp, "linear": linear, "angular": angular}
        if not self.gate.accept(frame, self.clock.now()):
            return False
        self.frames.append((sequence, stamp, linear, angular))
        return True


def test_relative_relay_ready_and_first_motion_limit_contract():
    clock = Clock()
    relay = RelativeContractRelay(clock)
    _, result, _, _, _, _ = run_trial(relay=relay, clock=clock)
    assert result["reason"] == "motion_limit"
    assert relay.frames and relay.stop_calls == 1
    first = relay.frames[0][1]
    assert all(frame[1] < first + 0.3 for frame in relay.frames)
    assert relay.gate.motion_started_at == first


@pytest.mark.parametrize("ready_line", [b"READY\n", b"READY motion_limit_sec=0.400000\n",
                                        b"READY motion_limit_sec=0.300000", b"FAULT motion_deadline\n"])
def test_relay_must_confirm_the_exact_raw_ready_line(ready_line):
    clock = Clock()
    relay = RelativeContractRelay(clock, ready_line=ready_line)
    _, result, _, _, _, _ = run_trial(relay=relay, clock=clock)
    assert result["reason"] == "relay_contract"
    assert not relay.frames and relay.stop_calls == 1


def test_relay_gate_relative_timer_starts_once_and_heartbeats_do_not_extend_it():
    import relay as relay_module
    gate = relay_module.RelayGate(10.0, max_motion_sec=0.3)
    assert gate.motion_started_at is None
    for sequence, stamp in enumerate((10.05, 10.15, 10.25, 10.34)):
        assert gate.accept({"sequence": sequence, "stamp": stamp,
                            "linear": 0.02, "angular": 0.0}, stamp)
    assert gate.motion_started_at == 10.05
    assert gate.command(10.351) == (0.0, 0.0)
    assert gate.reason == "motion_deadline"
    assert not gate.accept({"sequence": 4, "stamp": 10.36, "linear": 0.02, "angular": 0.0}, 10.36)


def test_lateral_motion_during_model_inference_is_rejected():
    def move_sideways(trace):
        for item in trace:
            item["linear_y_mps"] = 0.0
        trace[len(trace) // 2]["linear_y_mps"] = 0.02
        return trace
    clock = Clock()
    relay = Relay(clock)
    source = Source(clock, trace_mutate=move_sideways)
    _, result, _, _, _, _ = run_trial(source=source, model=Model(clock, delay=2), relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_stationary_trace_supports_bounded_live_callback_jitter():
    check = trial_module()._stationary_trace_valid
    trace = [{'stamp': stamp, 'seq': index, 'epoch': 1, 'pose': (0.0, 0.0, 0.0),
              'linear_x_mps': 0.0, 'linear_y_mps': 0.0, 'angular_z_radps': 0.0}
             for index, stamp in enumerate((10.0, 10.12, 10.24))]
    assert not check(trace, 10.0, 10.24, 1, (0.0, 0.0, 0.0))
    assert check(trace, 10.0, 10.24, 1, (0.0, 0.0, 0.0), max_gap_sec=0.15)
    assert not check(trace, 10.0, 10.24, 1, (0.0, 0.0, 0.0), max_gap_sec=0.2)


def test_missing_lateral_velocity_in_stationary_history_is_rejected():
    def missing_lateral(trace):
        for item in trace:
            item.pop("linear_y_mps", None)
        return trace
    clock = Clock()
    relay = Relay(clock)
    source = Source(clock, trace_mutate=missing_lateral)
    _, result, _, _, _, _ = run_trial(source=source, model=Model(clock, delay=2), relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_lateral_velocity_during_trial_stops_the_next_command():
    def sideways(sample, count):
        sample["state"]["linear_y_mps"] = 0.02 if count >= 5 else 0.0
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, sideways), relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert relay.frames and relay.stop_calls == 1


def run_trial(source=None, model=None, relay=None, clock=None, **kwargs):
    clock = clock or Clock()
    source = source or Source(clock)
    model = model or Model(clock)
    relay = relay or Relay(clock)
    trial = trial_module().ForwardTrial(source, model, relay, clock, **kwargs)
    return trial, trial.run(), source, model, relay, clock


def test_bounded_single_thread_forward_and_terminal_zero_latch():
    trial, result, source, model, relay, clock = run_trial(duration_sec=0.4)
    assert result["reason"] == "motion_limit"
    assert result["physical_stop_verified"] is False and result["stop_requested"] is True
    assert result["diagnostic_only"] is True
    assert model.evidence and model.evidence[0]["image_jpeg"].startswith(b"\xff\xd8")
    assert model.evidence[0]["latest_rays"] and model.evidence[0]["retained_hits"]
    assert model.evidence[0]["raw_scan"]["ranges"][16] == 2.0
    assert relay.frames and relay.stop_calls == 1 and relay.closed
    assert all(0 < frame[2] <= 0.02 and frame[3] == 0 for frame in relay.frames)
    assert all(frame[1] < relay.deadline for frame in relay.frames)
    assert [f[0] for f in relay.frames] == list(range(len(relay.frames)))
    assert all(b[1] > a[1] for a, b in zip(relay.frames, relay.frames[1:]))
    assert len(set(relay.calls_threads)) == 1
    assert trial.run()["reason"] == "already_used"
    assert relay.stop_calls == 1


def test_model_evidence_expiry_stops_the_remaining_motion_window():
    clock = Clock()
    model = Model(clock, delay=7.8)
    trial, result, source, model, relay, clock = run_trial(model=model, clock=clock)
    assert result["reason"] == "model_stale"
    assert relay.frames and relay.stop_calls == 1
    assert all(frame[1] - model.evidence[0]["color_at"] <= 8.0 for frame in relay.frames)


@pytest.mark.parametrize("choice", ["stop", "left", "right"])
def test_model_must_choose_forward_before_any_motion(choice):
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(model=Model(clock, choice=choice), relay=relay, clock=clock)
    assert result["reason"] == "model_not_forward"
    assert relay.frames == [] and relay.stop_calls == 1


def test_model_result_must_bind_to_the_same_image_and_scan():
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(model=Model(clock, wrong_id=True), relay=relay, clock=clock)
    assert result["reason"] == "model_invalid"
    assert not relay.frames and relay.stop_calls == 1


def test_truncated_image_cannot_be_used_as_a_scene_observation():
    def mutate(sample, _count):
        sample["image_jpeg"] = b"\xff\xd8truncated"
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, mutate), relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_model_cannot_change_the_expected_observation_digest():
    class MutatingModel(Model):
        def decide(self, evidence):
            evidence["observation_id"] = "tampered"
            return {"ok": True, "choice": "forward", "observation_id": "tampered"}
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(model=MutatingModel(clock), relay=relay, clock=clock)
    assert result["reason"] == "model_invalid"
    assert not relay.frames and relay.stop_calls == 1


def test_model_result_older_than_eight_seconds_is_rejected_before_relay_is_armed():
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(model=Model(clock, delay=8.1), relay=relay, clock=clock)
    assert result["reason"] == "model_stale"
    assert relay.deadline is None and relay.frames == [] and relay.stop_calls == 1


@pytest.mark.parametrize("delay", [2.0, 6.0])
def test_model_latency_up_to_six_seconds_can_run_when_odom_proves_stationary(delay):
    clock = Clock()
    relay = Relay(clock)
    _, result, _, model, _, _ = run_trial(model=Model(clock, delay=delay), relay=relay, clock=clock,
                                          duration_sec=0.4)
    assert result["reason"] == "motion_limit"
    assert relay.frames and relay.stop_calls == 1
    assert relay.frames[0][1] - model.evidence[0]["color_at"] >= delay


def test_motion_during_model_inference_is_rejected_even_if_pose_returns():
    def move_midway(trace):
        trace[len(trace) // 2]["pose"] = (0.02, 0.0, 0.0)
        return trace
    clock = Clock()
    relay = Relay(clock)
    source = Source(clock, trace_mutate=move_midway)
    _, result, _, _, _, _ = run_trial(source=source, model=Model(clock, delay=2), relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_incomplete_odom_history_cannot_prove_stationary_inference():
    clock = Clock()
    relay = Relay(clock)
    source = Source(clock, trace_mutate=lambda trace: trace[-2:])
    _, result, _, _, _, _ = run_trial(source=source, model=Model(clock, delay=6), relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


@pytest.mark.parametrize("problem", ["gap", "seq_reset", "epoch_change", "epoch_bool", "nonzero_speed"])
def test_stationary_history_rejects_discontinuity_or_movement(problem):
    def alter(trace):
        if problem == "gap":
            del trace[10:14]
        elif problem == "seq_reset":
            trace[10]["seq"] = trace[9]["seq"]
        elif problem == "epoch_change":
            trace[10]["epoch"] = 2
        elif problem == "epoch_bool":
            trace[10]["epoch"] = True
        else:
            trace[10]["linear_x_mps"] = 0.01
        return trace
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, trace_mutate=alter),
                                      model=Model(clock, delay=2), relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_epoch_change_during_model_wait_invalidates_the_decision():
    class EpochChangingModel(Model):
        def __init__(self, clock, source):
            super(EpochChangingModel, self).__init__(clock, delay=2)
            self.source = source
        def decide(self, evidence):
            result = super(EpochChangingModel, self).decide(evidence)
            self.source.current_epoch = 2
            return result
    clock = Clock()
    relay = Relay(clock)
    source = Source(clock)
    _, result, _, _, _, _ = run_trial(source=source, model=EpochChangingModel(clock, source),
                                      relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_relay_must_have_its_own_first_motion_deadline():
    clock = Clock()
    relay = Relay(clock, hard_deadline=False)
    _, result, _, _, _, _ = run_trial(relay=relay, clock=clock)
    assert result["reason"] == "relay_contract"
    assert not relay.frames and relay.stop_calls == 1


@pytest.mark.parametrize("kwargs", [{"zero_on_parent_loss": False}, {"independent_heartbeat": False}])
def test_relay_requires_independent_parent_loss_and_heartbeat_stops(kwargs):
    clock = Clock()
    relay = Relay(clock, **kwargs)
    _, result, _, _, _, _ = run_trial(relay=relay, clock=clock)
    assert result["reason"] == "relay_contract"
    assert not relay.frames and relay.stop_calls == 1


def test_sensor_freshness_is_rechecked_after_a_slow_relay_health_call():
    clock = Clock()
    relay = Relay(clock, health_delay=0.11)
    _, result, _, _, _, _ = run_trial(relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_slow_final_scene_validation_cannot_send_with_stale_odometry(monkeypatch):
    mod = trial_module()
    clock = Clock()
    relay = Relay(clock)
    real_check = mod._snapshot_reason
    calls = [0]
    def slow_check(sample, now):
        calls[0] += 1
        reason = real_check(sample, now)
        if calls[0] == 4:
            clock.t += 0.11
        return reason
    monkeypatch.setattr(mod, "_snapshot_reason", slow_check)
    _, result, _, _, _, _ = run_trial(relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_source_epoch_change_before_send_prevents_motion():
    clock = Clock()
    source = Source(clock)
    relay = Relay(clock, on_health=lambda: setattr(source, "current_epoch", 2))
    _, result, _, _, _, _ = run_trial(source=source, relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_newer_sensor_data_before_send_invalidates_the_old_snapshot():
    clock = Clock()
    source = Source(clock)
    relay = Relay(clock, on_health=lambda: setattr(source, "newer_observation", True))
    _, result, _, _, _, _ = run_trial(source=source, relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_sensor_loss_after_motion_stops_and_does_not_restart():
    clock = Clock()
    source = Source(clock, lambda sample, count: None if count >= 5 else sample)
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=source, relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert relay.frames and relay.stop_calls == 1


def test_new_latest_scan_obstacle_stops_forward_motion():
    def mutate(sample, count):
        if count >= 5:
            sample["scene"]["rays"][16]["hit_base"] = (0.30, 0.0)
            sample["scene"]["occupied_hits"][16]["point_base"] = (0.30, 0.0)
            sample["raw_scan"]["ranges"][16] = 0.195
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, mutate), relay=relay, clock=clock)
    assert result["reason"] == "front_blocked"
    assert relay.frames and relay.stop_calls == 1


def test_retained_near_obstacle_blocks_even_if_latest_scan_has_long_rays():
    def mutate(sample, count):
        old = sample["scan_at"] - 0.7
        sample["scene"]["rays"].append({"start_base": (0.0, 0.0),
                                         "hit_base": (0.09, 0.0), "stamp": old})
        sample["scene"]["occupied_hits"].append({"point_base": (0.09, 0.0), "stamp": old})
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, mutate), relay=relay, clock=clock)
    assert result["reason"] == "front_blocked"
    assert not relay.frames and relay.stop_calls == 1


def test_raw_latest_lidar_obstacle_blocks_even_if_derived_scene_misses_it():
    def mutate(sample, _count):
        sample["raw_scan"]["ranges"][16] = 0.195
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, mutate), relay=relay, clock=clock)
    assert result["reason"] == "front_blocked"
    assert not relay.frames and relay.stop_calls == 1


def test_raw_no_return_samples_cannot_be_replaced_by_claimed_scene_coverage():
    def mutate(sample, _count):
        sample["raw_scan"]["ranges"] = [10.0] * 33
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, mutate), relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_raw_lidar_hit_is_transformed_from_scan_pose_into_current_base():
    def mutate(sample, _count):
        sample["pose"] = sample["image_pose"] = sample["depth_pose"] = (0.01, 0.0, 0.0)
        sample["scene"]["coverage"]["scan_pose"] = (0.0, 0.0, 0.0)
        sample["raw_scan"]["ranges"][16] = 0.5
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, mutate), relay=relay, clock=clock)
    assert result["reason"] == "front_blocked"
    assert not relay.frames and relay.stop_calls == 1


def test_missing_raw_ranges_are_explicitly_unknown_in_model_evidence():
    def mutate(sample, _count):
        sample["raw_scan"]["ranges"][0] = float("nan")
        sample["scene"]["rays"].pop(0)
        sample["scene"]["occupied_hits"].pop(0)
        sample["scene"]["coverage"]["latest_valid_samples"] = 32
        sample["scene"]["coverage"]["latest_valid_fraction"] = 32.0 / 33
        return sample
    clock = Clock()
    _, result, _, model, relay, _ = run_trial(source=Source(clock, mutate), clock=clock, duration_sec=0.4)
    assert result["reason"] == "motion_limit" and relay.frames
    assert model.evidence[0]["raw_scan"]["ranges"][0] == "nan"
    assert model.evidence[0]["nonreturns_are_unknown"] is True


def test_odometry_displacement_limit_stops_before_another_command():
    def mutate(sample, count):
        if count >= 5:
            sample["pose"] = (0.021, 0.0, 0.0)
            sample["image_pose"] = sample["depth_pose"] = sample["pose"]
            sample["scene"]["coverage"]["scan_pose"] = sample["pose"]
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, mutate), relay=relay, clock=clock)
    assert result["reason"] in ("distance_limit", "odom_discontinuity")
    assert relay.frames and relay.stop_calls == 1
    assert len(relay.frames) < 5


def test_trial_does_not_start_when_odometry_already_reports_motion():
    def mutate(sample, _count):
        sample["state"]["linear_x_mps"] = 0.01
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, mutate), relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_odom_position_moving_backwards_stops_even_if_twist_reports_zero():
    def mutate(sample, count):
        if count >= 5:
            sample["pose"] = (-0.001, 0.0, 0.0)
            sample["image_pose"] = sample["depth_pose"] = sample["pose"]
            sample["scene"]["coverage"]["scan_pose"] = sample["pose"]
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, mutate), relay=relay, clock=clock)
    assert result["reason"] == "odom_discontinuity"
    assert relay.frames and relay.stop_calls == 1


@pytest.mark.parametrize("field", ["status_at", "odom_at", "scan_at", "depth_at", "color_at"])
def test_stale_sensor_evidence_cannot_authorize_motion(field):
    def mutate(sample, _count):
        sample[field] -= 2.0
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, mutate), relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert relay.frames == [] and relay.stop_calls == 1


def test_low_depth_coverage_is_not_relaxed_for_supervision():
    def mutate(sample, _count):
        sample["depth"]["valid_grid"] = [[0.312, 0.0, 0.185],
                                          [0.469, 0.312, 0.228], [0.168, 0.160, 0.404]]
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, mutate), relay=relay, clock=clock)
    assert result["reason"] == "sensor_unavailable"
    assert not relay.frames and relay.stop_calls == 1


def test_relay_fault_or_send_exception_latches_zero():
    for health, send in ((3, None), (None, 2)):
        clock = Clock()
        relay = Relay(clock, fail_health_at=health, fail_send_at=send)
        _, result, _, _, _, _ = run_trial(relay=relay, clock=clock)
        assert result["reason"] == "relay_failure"
        assert relay.frames and relay.stop_calls == 1


def test_send_exception_after_a_possible_side_effect_is_reported_as_motion_requested():
    class LostAcknowledgementRelay(Relay):
        def send_frame(self, sequence, stamp, linear, angular):
            super(LostAcknowledgementRelay, self).send_frame(sequence, stamp, linear, angular)
            raise IOError("synthetic acknowledgement loss")
    clock = Clock()
    relay = LostAcknowledgementRelay(clock)
    _, result, _, _, _, _ = run_trial(relay=relay, clock=clock)
    assert result["reason"] == "relay_failure"
    assert result["motion_requested"] is True and result["physical_stop_verified"] is False
    assert relay.frames and relay.stop_calls == 1


def test_exception_requests_stop_without_claiming_physical_stop():
    def fail(sample, count):
        if count >= 5:
            raise RuntimeError("synthetic failure")
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, fail), relay=relay, clock=clock)
    assert result["reason"] == "exception"
    assert result["stop_requested"] is True and result["physical_stop_verified"] is False
    assert relay.stop_calls == 1


def test_catchable_process_exit_still_requests_a_latched_zero():
    def exit_during_control(sample, count):
        if count >= 5:
            raise SystemExit(3)
        return sample
    clock = Clock()
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, exit_during_control), relay=relay, clock=clock)
    assert result["reason"] == "exception"
    assert relay.frames and relay.stop_calls == 1 and result["stop_requested"] is True


def test_monotonic_clock_regression_stops_the_trial():
    clock = Clock()
    def move_clock_back(sample, count):
        if count >= 5:
            clock.t -= 1.0
        return sample
    relay = Relay(clock)
    _, result, _, _, _, _ = run_trial(source=Source(clock, move_clock_back), relay=relay, clock=clock)
    assert result["reason"] == "clock_invalid"
    assert relay.frames and relay.stop_calls == 1


def test_frozen_clock_cannot_repeat_heartbeat_or_run_without_bound():
    class FrozenClock(Clock):
        def sleep(self, _seconds):
            pass
    clock = FrozenClock()
    relay = Relay(clock, fail_send_at=3)
    _, result, _, _, _, _ = run_trial(relay=relay, clock=clock)
    assert result["reason"] == "clock_invalid"
    assert len(relay.frames) == 1 and relay.stop_calls == 1


def test_failed_relay_zero_request_is_reported_without_claiming_stop():
    class FailingStopRelay(Relay):
        def stop_and_latch(self):
            self.stop_calls += 1
            return False
    clock = Clock()
    relay = FailingStopRelay(clock)
    _, result, _, _, _, _ = run_trial(relay=relay, clock=clock, duration_sec=0.4)
    assert result["reason"] == "relay_stop_failed" and result["trigger_reason"] == "motion_limit"
    assert result["stop_requested"] is False and result["physical_stop_verified"] is False
    assert relay.stop_calls == 1 and relay.closed


def test_invalid_trial_limits_and_nonfixture_adapters_are_rejected():
    mod = trial_module()
    clock = Clock()
    for kwargs in ({"duration_sec": 0.6001}, {"speed_mps": 0.0201}, {"speed_mps": -0.01}):
        with pytest.raises(ValueError):
            mod.ForwardTrial(Source(clock), Model(clock), Relay(clock), clock, **kwargs)
    relay = Relay(clock)
    relay.simulation_only = False
    with pytest.raises(ValueError):
        mod.ForwardTrial(Source(clock), Model(clock), relay, clock)
    assert relay.frames == [] and relay.stop_calls == 0


def test_concurrent_run_attempts_cannot_start_two_sessions():
    mod = trial_module()
    class DelayedUsedRead(mod.ForwardTrial):
        def __getattribute__(self, name):
            value = super(DelayedUsedRead, self).__getattribute__(name)
            if name == "_used":
                time.sleep(0.05)
            return value
    clock = Clock()
    relay = Relay(clock)
    trial = DelayedUsedRead(Source(clock), Model(clock), relay, clock, duration_sec=0.4)
    results = []
    threads = [threading.Thread(target=lambda: results.append(trial.run())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(result["reason"] for result in results) == ["already_used", "motion_limit"]
    assert relay.stop_calls == 1 and len(set(relay.calls_threads)) == 1
