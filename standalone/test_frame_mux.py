"""Processed per-process results and producer failure; no motion I/O."""
from frame_mux import FrameMux


def ingress(stream, sequence, at):
    return {'kind': 'ingress', 'stream': stream, 'sequence': sequence, 'epoch': 1,
            'acquired_at': at, 'received_at': at}


def result(stream, sequence, at, status='candidate'):
    return {'kind': 'result', 'stream': stream, 'sequence': sequence, 'epoch': 1,
            'acquired_at': at, 'received_at': at, 'completed_at': at + .01,
            'valid_until': at + (.3 if stream == 'scan' else .15) if status == 'candidate' else None,
            'status': status, 'reason': ('sampled_forward_corridor' if stream == 'scan'
                                         else 'sampled_depth_corridor')}


def test_mux_preserves_frame_order_and_faults_on_producer_loss():
    mux = FrameMux(max_processing_lag=.2)
    for stream in ('scan', 'depth'):
        assert mux.accept(ingress(stream, 0, 10.))
        assert mux.accept(result(stream, 0, 10.))
    assert mux.read(10.02)['state'] == 'ready'
    mux.channel_failed('scan')
    assert mux.read(10.03)['state'] == 'fault'


def test_mux_rejects_sequence_or_captured_time_changes():
    mux = FrameMux(max_processing_lag=.2)
    assert mux.accept(ingress('scan', 0, 10.))
    changed = result('scan', 0, 10.)
    changed['acquired_at'] = 10.005
    assert not mux.accept(changed)
    assert mux.read(10.02)['state'] == 'fault'
    mux = FrameMux(max_processing_lag=.2)
    assert not mux.accept(ingress('scan', 5, 10.))
    assert mux.read(10.02)['state'] == 'fault'


def test_unknown_revokes_existing_channel_and_is_not_a_fault():
    mux = FrameMux(max_processing_lag=.2)
    for stream in ('scan', 'depth'):
        assert mux.accept(ingress(stream, 0, 10.))
        assert mux.accept(result(stream, 0, 10.))
    assert mux.accept(ingress('depth', 1, 10.03))
    assert mux.accept(result('depth', 1, 10.03, 'unknown'))
    assert mux.read(10.05)['state'] == 'waiting'


def test_candidate_or_unknown_cannot_hide_a_reported_obstacle():
    mux = FrameMux(max_processing_lag=.2)
    assert mux.accept(ingress('scan', 0, 10.))
    bad = result('scan', 0, 10.)
    bad['reason'] = 'front_blocked'
    assert not mux.accept(bad)
    assert mux.read(10.02)['state'] == 'fault'


def test_valid_obstacle_record_preserves_stop_reason():
    mux = FrameMux(max_processing_lag=.2)
    assert mux.accept(ingress('scan', 0, 10.))
    blocked = result('scan', 0, 10., status='obstacle')
    blocked['reason'] = 'front_blocked'
    assert mux.accept(blocked)
    assert mux.read(10.02)['reason'] == 'front_blocked'
