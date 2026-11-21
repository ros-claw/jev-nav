"""Offline checks for clock, odometry, and multi-sensor alignment guards."""

import math

import pytest


def guards():
    try:
        import observation_guards
    except ImportError:
        pytest.fail("observation guards are not implemented", pytrace=False)
    return observation_guards


def clock_guard():
    return guards().ClockGuard(max_offset_jump_sec=0.02)


def odom_guard():
    return guards().OdomGuard(max_linear_mps=0.12, max_angular_radps=0.1,
                              position_slack_m=0.002, yaw_slack_rad=0.003,
                              max_gap_sec=0.3)


def aligned(stamps, samples, now=10.1, **overrides):
    options = {"max_sensor_age_sec": 0.5, "max_sensor_skew_sec": 0.1,
               "max_odom_skew_sec": 0.02, "max_translation_m": 0.01,
               "max_rotation_rad": 0.02, "max_linear_mps": 0.12,
               "max_angular_radps": 0.1}
    options.update(overrides)
    return guards().sensors_aligned(stamps, samples, now, **options)


def pose_sample(stamp, pose=(0.0, 0.0, 0.0)):
    return {"stamp": stamp, "pose": pose}


def test_clock_mapping_allows_continuous_clocks():
    guard = clock_guard()
    assert guard.observe(1000.0, 50.0)
    assert guard.observe(1000.1, 50.1)
    assert guard.observe(1000.2, 50.2)


@pytest.mark.parametrize("ros_now,mono_now", [
    (1005.1, 50.1), (995.1, 50.1), (1000.1, 49.9),
    (float("nan"), 50.1), (1000.1, float("inf")),
])
def test_clock_jump_regression_or_invalid_sample_latches_permanent_fault(ros_now, mono_now):
    guard = clock_guard()
    assert guard.observe(1000.0, 50.0)
    assert not guard.observe(ros_now, mono_now)
    assert not guard.observe(1000.2, 50.2)


def test_odometry_allows_plausible_translation_and_wrapped_yaw():
    guard = odom_guard()
    assert guard.observe(100, 10.0, (1.0, 2.0, math.pi - 0.004))
    assert guard.observe(101, 10.1, (1.01, 2.0, -math.pi + 0.004))


@pytest.mark.parametrize("seq,stamp,pose", [
    (0, 10.1, (0.0, 0.0, 0.0)),
    (100, 10.1, (0.0, 0.0, 0.0)),
    (101, 9.9, (0.0, 0.0, 0.0)),
    (101, 10.0, (0.0, 0.0, 0.0)),
    (101, 10.4, (0.0, 0.0, 0.0)),
    (101, 10.1, (0.03, 0.0, 0.0)),
    (101, 10.1, (0.0, 0.0, 0.04)),
    (101, 10.1, (float("nan"), 0.0, 0.0)),
    (101, 10.1, (0.0, 0.0, 4.0)),
])
def test_odom_sequence_time_pose_or_gap_fault_latches(seq, stamp, pose):
    guard = odom_guard()
    assert guard.observe(100, 10.0, (0.0, 0.0, 0.0))
    assert not guard.observe(seq, stamp, pose)
    assert not guard.observe(102, 10.2, (0.0, 0.0, 0.0))


def test_position_slack_cannot_accumulate_on_each_odom_sample():
    guard = odom_guard()
    assert guard.observe(100, 10.0, (0.0, 0.0, 0.0))
    accepted = []
    for index in range(1, 11):
        accepted.append(guard.observe(100 + index, 10.0 + 0.1 * index,
                                      (0.013 * index, 0.0, 0.0)))
    assert False in accepted
    assert not guard.observe(111, 11.1, (0.0, 0.0, 0.0))


def test_old_stationary_time_cannot_hide_a_later_excess_speed_burst():
    guard = odom_guard()
    assert guard.observe(100, 10.0, (0.0, 0.0, 0.0))
    for index in range(1, 101):
        assert guard.observe(100 + index, 10.0 + 0.1 * index, (0.0, 0.0, 0.0))
    burst = [guard.observe(200 + index, 20.0 + 0.1 * index, (0.013 * index, 0.0, 0.0))
             for index in range(1, 11)]
    assert False in burst


def test_sensor_alignment_accepts_fresh_nearby_times_and_poses():
    stamps = {"scan": 10.0, "color": 10.04, "depth": 10.08}
    samples = [pose_sample(10.0), pose_sample(10.04, (0.001, 0.0, 0.001)),
               pose_sample(10.08, (0.002, 0.0, 0.002))]
    assert aligned(stamps, samples)


@pytest.mark.parametrize("stamps", [
    {"scan": 10.0, "color": 10.0, "depth": 10.11},
    {"scan": 9.5, "color": 10.0, "depth": 10.0},
    {"scan": 10.0, "color": 10.0, "depth": 10.3},
    {"scan": 10.0, "color": float("nan"), "depth": 10.0},
    {"scan": 10.0, "color": 10.0},
])
def test_sensor_alignment_rejects_skew_stale_future_or_malformed_stamps(stamps):
    assert not aligned(stamps, [pose_sample(10.0), pose_sample(10.1)], now=10.2)


def test_each_sensor_needs_a_nearby_odom_sample():
    stamps = {"scan": 10.0, "color": 10.04, "depth": 10.08}
    assert not aligned(stamps, [pose_sample(10.0), pose_sample(10.08)])


def test_sensor_alignment_rejects_relative_translation_or_rotation():
    stamps = {"scan": 10.0, "color": 10.04, "depth": 10.08}
    for pose in ((0.02, 0.0, 0.0), (0.0, 0.0, 0.04)):
        samples = [pose_sample(10.0), pose_sample(10.04), pose_sample(10.08, pose)]
        assert not aligned(stamps, samples)


def test_sensor_alignment_handles_yaw_wrap():
    stamps = {"scan": 10.0, "color": 10.04, "depth": 10.08}
    samples = [pose_sample(10.0, (0.0, 0.0, math.pi - 0.004)),
               pose_sample(10.04, (0.0, 0.0, -math.pi + 0.004)),
               pose_sample(10.08, (0.0, 0.0, -math.pi + 0.006))]
    assert aligned(stamps, samples)


def test_odom_time_skew_is_included_in_translation_uncertainty():
    stamps = {"scan": 10.0, "color": 10.05, "depth": 10.05}
    samples = [pose_sample(10.01), pose_sample(10.04)]
    assert not aligned(stamps, samples, max_translation_m=0.001)


def test_sensor_alignment_rejects_bad_or_unordered_odom_history():
    stamps = {"scan": 10.0, "color": 10.04, "depth": 10.08}
    assert not aligned(stamps, [pose_sample(10.0), pose_sample(10.08), pose_sample(10.04)])
    assert not aligned(stamps, [pose_sample(10.0), pose_sample(10.04, (float("nan"), 0.0, 0.0)),
                                pose_sample(10.08)])


def test_local_geometry_alignment_does_not_require_a_model_frame():
    stamps = {'scan': 10., 'depth': 10.05}
    odometry = [pose_sample(10.), pose_sample(10.05)]
    assert not aligned(stamps, odometry)
    assert aligned(stamps, odometry, require_color=False)
    assert not aligned(stamps, [pose_sample(10.)], require_color=False)
