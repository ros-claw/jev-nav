"""Synthetic contracts for a non-actuating processed-evidence store."""
import pytest
from processed_evidence import ProcessedEvidence


def ready_store():
    store = ProcessedEvidence({'scan': .3, 'depth': .15}, max_processing_lag=.1, max_pending=4)
    scan = store.ingest('scan', 10.0, 10.01, epoch=1)
    depth = store.ingest('depth', 10.0, 10.01, epoch=1)
    assert store.complete('scan', scan, True, 10.25, 10.04)
    assert store.complete('depth', depth, True, 10.14, 10.04)
    return store


def test_new_frame_does_not_discard_processed_result_but_caps_deadline():
    store = ready_store()
    store.ingest('depth', 10.05, 10.06, epoch=1)
    current = store.read(10.07, epoch=1)
    assert current['state'] == 'ready'
    assert current['valid_until'] == pytest.approx(10.11)
    assert current['pending'] == 1
    assert store.read(10.14, epoch=1)['state'] == 'waiting'


def test_pending_age_is_a_hard_deadline_and_never_refreshes_old_capture():
    store = ready_store()
    store.ingest('scan', 10.05, 10.06, epoch=1)
    assert store.read(10.10, epoch=1)['valid_until'] == pytest.approx(10.11)
    assert store.read(10.15, epoch=1)['state'] == 'waiting'


def test_completed_obstacle_or_independent_hazard_latches_fault():
    store = ready_store()
    seq = store.ingest('scan', 10.05, 10.06, epoch=1)
    assert store.complete('scan', seq, False, 10.20, 10.07, reason='obstacle')
    assert store.read(10.08, epoch=1)['reason'] == 'obstacle'
    assert store.read(10.09, epoch=1)['state'] == 'fault'
    other = ready_store()
    other.hazard('depth_obstacle')
    assert other.read(10.05, epoch=1)['reason'] == 'depth_obstacle'


def test_missing_producer_reorder_overflow_or_epoch_change_never_authorizes():
    store = ready_store()
    first = store.ingest('scan', 10.05, 10.06, epoch=1)
    second = store.ingest('scan', 10.07, 10.08, epoch=1)
    assert not store.complete('scan', second, True, 10.2, 10.09)
    assert store.read(10.10, epoch=1)['state'] == 'fault'
    assert not store.complete('scan', first, True, 10.2, 10.10)
    store = ready_store()
    store.producer_failed('producer_failed')
    assert store.read(10.05, epoch=1)['state'] == 'fault'
    store = ready_store()
    assert store.read(10.05, epoch=2)['state'] == 'fault'
    store = ready_store()
    for i in range(4):
        assert store.ingest('scan', 10.05 + i*.01, 10.06 + i*.01, epoch=1) is not None
    assert store.ingest('scan', 10.10, 10.11, epoch=1) is None
    assert store.read(10.12, epoch=1)['reason'] == 'backlog_overflow'


def test_forged_expiry_and_clock_regression_are_faults():
    store = ProcessedEvidence({'scan': .3, 'depth': .15}, max_processing_lag=.1)
    seq = store.ingest('depth', 10., 10.01, epoch=1)
    assert not store.complete('depth', seq, True, 10.2, 10.04)
    assert store.read(10.05, epoch=1)['state'] == 'fault'
    store = ready_store()
    assert store.read(10.05, epoch=1)['state'] == 'ready'
    assert store.read(10.04, epoch=1)['reason'] == 'clock_regression'


def test_future_completion_invalid_epoch_and_bad_stream_cannot_be_read_as_ready():
    store = ready_store()
    assert store.read(10.03, epoch=1)['state'] == 'fault'
    store = ready_store()
    assert store.read(10.05, epoch=True)['state'] == 'fault'
    store = ready_store()
    assert store.ingest([], 10.05, 10.06, epoch=1) is None
    assert store.read(10.07, epoch=1)['state'] == 'fault'


def test_unavailable_processed_frame_revokes_result_without_clearing_faults():
    store = ready_store()
    seq = store.ingest('scan', 10.05, 10.06, epoch=1)
    assert store.complete('scan', seq, None, 10.20, 10.07, reason='coverage_incomplete')
    assert store.read(10.08, epoch=1)['state'] == 'waiting'
    seq = store.ingest('scan', 10.09, 10.10, epoch=1)
    assert store.complete('scan', seq, True, 10.3, 10.11)
    depth = store.ingest('depth', 10.09, 10.10, epoch=1)
    assert store.complete('depth', depth, True, 10.2, 10.11)
    assert store.read(10.12, epoch=1)['state'] == 'ready'


def test_issued_deadline_already_bounds_a_frame_arriving_after_read():
    store = ProcessedEvidence({'scan': .5, 'depth': .5}, max_processing_lag=.1)
    for name in ('scan', 'depth'):
        seq = store.ingest(name, 10., 10.01, epoch=1)
        assert store.complete(name, seq, True, 10.4, 10.02)
    issued = store.read(10.03, epoch=1)['valid_until']
    assert issued <= 10.13
    assert store.ingest('scan', 10.04, 10.05, epoch=1) is not None
    assert issued <= 10.05 + .1


def test_delayed_ingress_cannot_make_issued_deadline_unsafe():
    store = ProcessedEvidence({'scan': .5, 'depth': .5}, max_processing_lag=.1)
    for name in ('scan', 'depth'):
        seq = store.ingest(name, 10., 10.01, epoch=1)
        assert store.complete(name, seq, True, 10.4, 10.02)
    issued = store.read(10.10, epoch=1)['valid_until']
    assert store.ingest('scan', 10.015, 10.02, epoch=1) is not None
    assert issued <= 10.02 + .1


def test_unknown_and_obstacle_results_do_not_need_a_fake_expiry():
    store = ready_store()
    seq = store.ingest('scan', 10.05, 10.06, epoch=1)
    assert store.complete('scan', seq, None, None, 10.07, reason='coverage_incomplete')
    assert store.read(10.08, epoch=1)['state'] == 'waiting'
    store = ready_store()
    seq = store.ingest('scan', 10.05, 10.06, epoch=1)
    assert store.complete('scan', seq, False, None, 10.07, reason='front_blocked')
    assert store.read(10.08, epoch=1)['reason'] == 'front_blocked'


def test_current_read_samples_clock_while_holding_evidence_lock():
    store = ready_store()
    def clock():
        acquired = store._lock.acquire(False)
        if acquired:
            store._lock.release()
        assert not acquired
        return 10.05
    result = store.read_current(clock, epoch=1)
    assert result['state'] == 'ready' and result['checked_at'] == 10.05
