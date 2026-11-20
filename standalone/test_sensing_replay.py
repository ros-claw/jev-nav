"""Replay candidate observations through the real in-memory RelayGate."""
import pytest
from sensing_replay import replay


def observation(now, reason='ready'):
    return {'event': 'perception', 'at': now, 'perception_start': now - .03,
            'perception_end': now, 'reason': reason, 'scan_at': now - .1,
            'depth_at': now - .02, 'odom_at': now - .01, 'status_at': now - .1,
            'snapshot_at': now - .01, 'valid_until': now + .3}


def test_replay_renews_real_gate_at_original_sensor_deadlines():
    result = replay([observation(10 + i*.06) for i in range(12)])
    assert len(result['segments']) == 1
    segment = result['segments'][0]
    assert segment['last_command_at'] - segment['first_command_at'] > .6
    assert segment['stop_reason'] == 'intent_deadline'
    assert segment['commands'] > 12
    assert result['motion_requested'] is False


def test_invalid_observation_stops_then_new_session_requires_new_proof():
    events = [observation(10.), observation(10.06), observation(10.12, 'sensor_unavailable'),
              observation(10.20), observation(10.26)]
    result = replay(events)
    assert len(result['segments']) == 2
    assert result['segments'][0]['stop_reason'] == 'observation_rejected'
    assert result['segments'][1]['first_command_at'] >= 10.20


def test_old_or_malformed_acquisition_never_authorizes_even_in_replay():
    item = observation(10.)
    item['depth_at'] = 9.0
    assert replay([item])['segments'] == []


def test_future_processing_completion_is_not_replayed_as_already_complete():
    item = observation(10.)
    item['perception_end'] = 11.
    with pytest.raises(ValueError):
        replay([item])


def test_sensor_frame_acquired_after_processing_cannot_authorize():
    item = observation(10.)
    item['perception_start'] = 9.9
    item['perception_end'] = 9.95
    item['depth_at'] = 9.98
    assert replay([item])['segments'] == []
    item = observation(10.)
    del item['scan_at']
    assert replay([item])['segments'] == []
