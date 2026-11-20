import pytest
from supervised_trial import (
    RealClock,
    make_relay,
    make_ros_monitor,
    parse_args,
    summarize_observations,
    trial_period,
    trial_exit_code,
    summarize_trial_observations,
)


def test_default_is_observation_only():
    args = parse_args([])
    assert args.execute is False and args.longer_forward is False


def test_longer_forward_trial_is_explicitly_selected():
    args = parse_args(['--longer-forward'])
    assert args.longer_forward is True and args.execute is False
    relay = make_relay(RealClock(), motion_limit_sec=2.0)
    assert relay.motion_limit_sec == 2.0 and relay.transport.process is None


def test_single_threaded_mode_requires_explicit_longer_trial():
    with pytest.raises(SystemExit):
        parse_args(['--single-threaded'])
    args = parse_args(['--longer-forward', '--single-threaded'])
    assert args.single_threaded and not args.execute
    assert trial_period(args) == 0.025
    assert trial_period(parse_args(['--longer-forward'])) == 0.05


def test_trial_uses_independent_heartbeat_without_starting_a_process_at_construction():
    from threaded_relay_adapter import ThreadedRelayAdapter
    relay = make_relay(RealClock())
    assert isinstance(relay, ThreadedRelayAdapter)
    assert relay.transport.process is None


def test_prepared_mode_requires_explicit_longer_trial():
    with pytest.raises(SystemExit):
        parse_args(['--prepared-forward'])
    with pytest.raises(SystemExit):
        parse_args(['--longer-forward', '--prepared-forward', '--single-threaded'])
    assert trial_period(parse_args(['--longer-forward', '--prepared-forward'])) == .025
    with pytest.raises(SystemExit):
        parse_args(['--longer-forward', '--preproject-depth'])
    assert parse_args(['--longer-forward', '--prepared-forward', '--preproject-depth']).preproject_depth


def test_depth_projection_process_requires_the_split_preprojection_path():
    mode = ['--longer-forward', '--prepared-forward', '--split-sensor-checks', '--depth-projection-process']
    with pytest.raises(SystemExit):
        parse_args(mode + ['--assume-rectified', '--ground-z=-.1'])
    with pytest.raises(SystemExit):
        parse_args(mode + ['--preproject-depth'])
    args = parse_args(mode + ['--preproject-depth', '--assume-rectified', '--ground-z=-.1'])
    assert args.depth_projection_process and not args.execute


def test_trial_monitor_allows_ros_rpc_scheduling_delay_without_changing_watchdog_freshness():
    guard = make_ros_monitor(RealClock(), 11139, 'http://localhost:11311')
    assert guard.backend.rpc_timeout == 0.2
    assert guard.poll_interval == 0.05
    assert guard.max_age == 0.3
    assert guard.bind_timeout == 0.35


@pytest.mark.parametrize('argv', [
    ['--execute'],
    ['--execute', '--driver-pid', '1'],
    ['--execute', '--driver-pid', '1', '--operator-site-reported'],
    ['--execute', '--driver-pid', '0', '--operator-site-reported', '--ground-z=-0.1', '--assume-rectified'],
    ['--execute', '--driver-pid', '2', '--operator-site-reported', '--ground-z=-0.1', '--assume-rectified'],
])
def test_motion_requires_explicit_runtime_evidence_and_projection_assumptions(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_motion_accepts_explicit_assumptions_and_process_identity():
    args = parse_args(['--execute', '--driver-pid', '11139', '--operator-site-reported',
                       '--ground-z=-0.1', '--assume-rectified', '--driver-exe=/tmp/candidate',
                       '--driver-sha256=' + 'a' * 64])
    assert args.execute and args.driver_pid == 11139 and args.ground_z == -0.1


def test_business_failure_and_no_measurable_odom_do_not_exit_successfully():
    assert trial_exit_code({'reason': 'relay_failure', 'motion_requested': True},
                           {'odom_displacement_m': .01, 'nonzero_cmd_vel_messages': 2}) != 0
    assert trial_exit_code({'reason': 'time_limit', 'motion_requested': True},
                           {'odom_displacement_m': .000001, 'nonzero_cmd_vel_messages': 2}) != 0
    observed = {'odom_displacement_m': .006, 'nonzero_cmd_vel_messages': 2,
                'measurement_scope': 'relay_callback_window', 'relay_caller_observed': True}
    assert trial_exit_code({'reason': 'motion_limit', 'motion_requested': True, 'stop_requested': True},
                           observed) == 0
    assert trial_exit_code({'reason': 'motion_limit', 'motion_requested': True},
                           {'odom_displacement_m': float('nan'), 'nonzero_cmd_vel_messages': 2}) != 0
    assert trial_exit_code({'reason': 'motion_limit', 'motion_requested': True},
                           {'odom_displacement_m': .006, 'nonzero_cmd_vel_messages': '2'}) != 0
    for extra in ({'stop_requested': False}, {'stop_requested': True, 'sampler_closed': False},
                  {'stop_requested': True, 'source_restore_failed': True}):
        result = {'reason': 'motion_limit', 'motion_requested': True}
        result.update(extra)
        assert trial_exit_code(result, observed) != 0


def test_trial_summary_excludes_warmup_and_nonrelay_commands():
    odom = [(1., 0., 0., 0.), (2., .2, 0., 0.), (2.95, .2, 0., 0.),
            (3.1, .2, 0., 0.), (3.15, .2, 0., 0.), (3.4, .2, 0., 0.)]
    commands = [(1.5, .02, 0., '/other'), (3., .02, 0., '/other'),
                (3.05, .02, 0., '/relay'), (3.2, 0., 0., '/relay')]
    summary = summarize_trial_observations(odom, commands, '/relay', 2.5, 3.2)
    assert summary['nonzero_cmd_vel_messages'] == 1
    assert summary['odom_displacement_m'] == pytest.approx(0.)
    assert summary['measurement_scope'] == 'relay_callback_window'
    assert summary['relay_caller_observed'] is True
    missing = summarize_trial_observations(odom, commands, '/wrong', 2.5, 3.2)
    assert missing['nonzero_cmd_vel_messages'] == 0 and missing['odom_displacement_m'] is None


def test_trial_summary_never_counts_motion_before_first_command_or_after_end():
    odom = [(9.95, 0., 0., 0.), (10.05, .01, 0., 0.), (10.2, .01, 0., 0.),
            (11.2, .02, 0., 0.)]
    commands = [(10., .02, 0., '/relay'), (10.3, 0., 0., '/relay')]
    summary = summarize_trial_observations(odom, commands, '/relay', 9.5, 11.)
    assert summary['odom_displacement_m'] == pytest.approx(0.)


def test_trial_summary_excludes_motion_after_relay_zero_even_before_run_returns():
    odom = [(10.12, 0., 0., 0.), (10.19, 0., 0., 0.), (10.8, .01, 0., 0.)]
    commands = [(10.1, .02, 0., '/relay'), (10.2, 0., 0., '/relay')]
    summary = summarize_trial_observations(odom, commands, '/relay', 10., 11.)
    assert summary['odom_displacement_m'] == pytest.approx(0.)


def test_observation_summary_distinguishes_command_from_reported_odometry():
    odom = [(1.0, 0.0, 0.0, 0.0), (1.1, 0.012, 0.0, 0.02), (1.2, 0.015, 0.0, 0.0)]
    commands = [(1.0, 0.0, 0.0), (1.05, 0.02, 0.0), (1.2, 0.0, 0.0)]
    summary = summarize_observations(odom, commands)
    assert summary['nonzero_cmd_vel_messages'] == 1
    assert summary['odom_displacement_m'] == pytest.approx(0.015)
    assert summary['max_abs_odom_speed_mps'] == 0.02
    assert summary['last_odom_speed_mps'] == 0.0
    assert summary['physical_stop_verified'] is False


def test_observation_summary_rejects_invalid_odom_instead_of_inventing_motion():
    summary = summarize_observations([(1.0, float('nan'), 0.0, 0.0)], [])
    assert summary['odom_displacement_m'] is None
    assert summary['last_odom_speed_mps'] is None
