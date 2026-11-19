import pytest
from supervised_trial import parse_args, summarize_observations


def test_default_is_observation_only():
    args = parse_args([])
    assert args.execute is False


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
