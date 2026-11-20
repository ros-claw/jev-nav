"""Frame timing and real gate replay; no robot, subprocess or ROS."""
from frame_replay import replay_frames


def frame(stream, stamp, duration=.01, clear=True):
    age = .3 if stream == 'scan' else .15
    return {'stream': stream, 'acquired': stamp, 'received': stamp,
            'completed': stamp + duration, 'valid_until': stamp + age,
            'clear': clear, 'reason': 'sampled'}


def test_replay_uses_only_processed_channels_and_real_relay_gate():
    frames = []
    for i in range(30):
        stamp = 10. + i*.04
        frames.extend([frame('scan', stamp), frame('depth', stamp)])
    result = replay_frames(frames, max_processing_lag=.2)
    assert result['motion_requested'] is False
    assert result['longest_segment_sec'] > .5
    assert result['accepted_commands'] > 0
    assert result['odom_continuity_checked'] is False
    assert result['base_status_checked'] is False


def test_unknown_depth_and_obstacle_never_generate_a_permit():
    frames = [frame('scan', 10.), frame('depth', 10., clear=None),
              frame('scan', 10.04, clear=False), frame('depth', 10.04)]
    frames[1]['valid_until'] = None
    frames[2]['valid_until'] = None
    frames[2]['reason'] = 'front_blocked'
    result = replay_frames(frames, max_processing_lag=.2)
    assert result['accepted_commands'] == 0
    assert result['final_state'] == 'fault'
