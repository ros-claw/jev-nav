"""Exercise cached evidence together with the real in-memory RelayGate."""
import heapq

from processed_evidence import ProcessedEvidence
from relay import RelayGate


def drive_with_independent_workers(stall_scan=False):
    store = ProcessedEvidence({'scan': .3, 'depth': .15}, max_processing_lag=.1, max_pending=8)
    gate = RelayGate(100.0, max_motion_sec=4.0, require_intent_deadline=True)
    events, keys = [], {}
    for index in range(28):
        acquired = 100.0 + index * .11
        heapq.heappush(events, (acquired, 'scan_in', index))
        delay = .30 if stall_scan and index == 5 else .096
        heapq.heappush(events, (acquired + delay, 'scan_out', index))
    for index in range(100):
        acquired = 100.0 + index * .03
        heapq.heappush(events, (acquired, 'depth_in', index))
        heapq.heappush(events, (acquired + .008, 'depth_out', index))
    sequence, written, started = 0, [], False
    while events:
        now = events[0][0]
        if started:
            gate.command(now)
            if gate.stopped:
                break
        while events and events[0][0] == now:
            _, kind, index = heapq.heappop(events)
            stream = kind.split('_')[0]
            if kind.endswith('_in'):
                keys[(stream, index)] = store.ingest(stream, now, now, epoch=1)
            else:
                acquired = 100.0 + index * (.11 if stream == 'scan' else .03)
                deadline = acquired + (.3 if stream == 'scan' else .15)
                store.complete(stream, keys[(stream, index)], True, deadline, now)
        state = store.read(now, epoch=1)
        if state['state'] != 'ready':
            if started:
                break
            continue
        deadline = min(state['valid_until'], now + .16)
        frame = {'sequence': sequence, 'stamp': now, 'linear': .02,
                 'angular': 0., 'expires_at': deadline}
        if not gate.accept(frame, now):
            break
        sequence += 1
        started = True
        written.append(frame)
    return written, gate, state


def test_conservative_ingress_bound_reveals_96ms_scan_budget_gap():
    frames, gate, state = drive_with_independent_workers()
    assert frames and (gate.stopped or state['state'] != 'ready')
    assert frames[-1]['stamp'] - frames[0]['stamp'] < .2


def test_stalled_scan_does_not_keep_renewing_old_evidence():
    frames, gate, state = drive_with_independent_workers(stall_scan=True)
    assert frames and (gate.stopped or state['state'] != 'ready')
    assert frames[-1]['stamp'] - frames[0]['stamp'] < 1.0
