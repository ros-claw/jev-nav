import socket

from memory_bridge import MemoryBridge


def test_events_and_bounded_recall_use_the_configured_scope():
    calls = []
    def transport(path, payload, timeout):
        calls.append((path, payload))
        if path.endswith('/prepare'):
            return {'status': 'ready', 'content': 'A previous left turn was blocked.',
                    'receipt': {'selected': [{'artifact_address': {'scope_id': 'room-scope',
                                                                  'artifact': {'artifact_id': 'm-7',
                                                                               'revision': 2}}}]}}
        return {'source': {'name': 'content', 'source_id': 'limo-run-8'}}

    bridge = MemoryBridge('http://127.0.0.1:8000', 'room-scope', transport=transport)
    try:
        assert bridge.record('limo-run-8', 'Left arc was blocked', {'event_type': 'occupied_detour'})
        result = bridge.prepare('room low obstacle turning')
        assert result['status'] == 'ready' and result['content']
        assert result['references'] == [('m-7', 2)]
    finally:
        assert bridge.close()
    assert sorted(path for path, _ in calls) == ['/v1/context/prepare', '/v1/sources/content']
    assert all(payload['scope_id'] == 'room-scope' for _, payload in calls)


def test_disconnected_service_returns_no_history_and_converts_transport_errors():
    def transport(*args):
        raise socket.timeout('private transport diagnostics')

    bridge = MemoryBridge('http://127.0.0.1:8000', 'room-scope', transport=transport)
    try:
        assert bridge.prepare('room') == {'status': 'unavailable'}
        bridge.record('limo-run-9', 'Contact was reported', {'pose_valid': False})
    finally:
        assert bridge.close()
    assert bridge.counters['write_failed'] == 1


def test_experience_candidate_uses_the_acknowledged_source_and_stays_pending_review():
    calls = []

    def transport(path, payload, timeout):
        calls.append((path, payload))
        if path == '/v1/sources/content':
            return {'source': {'name': 'content', 'source_id': 'stored-event-12'}}
        return {'candidate_id': 'candidate-12', 'version': 1}

    bridge = MemoryBridge('http://127.0.0.1:8000', 'room-scope', transport=transport)
    proposal = {'situation': 'Approaching room anchor 7', 'action': 'Forward command',
                'outcome': 'Measured translation was below threshold', 'lesson': 'Inspect this approach again'}
    try:
        assert bridge.record('limo-run-12', 'No progress on approach', {'pose_valid': True},
                             experience=proposal)
    finally:
        assert bridge.close()
    assert [path for path, _ in calls] == ['/v1/sources/content', '/v1/experience/propose']
    assert calls[-1][1] == {'scope_id': 'room-scope', 'proposal': proposal,
                            'source_refs': [{'name': 'content', 'source_id': 'stored-event-12'}],
                            'artifact_refs': []}
    assert list(bridge.pending_candidates) == [('limo-run-12', 'candidate-12', 1)]
