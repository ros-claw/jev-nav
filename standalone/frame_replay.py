"""Replay processed frame timing with the actual RelayGate, entirely in memory.

Input is diagnostic evidence, not a geometry or pose safety proof. The replay
never creates a publisher or device connection; its commands are hypothetical.
"""
from __future__ import division

from processed_evidence import ProcessedEvidence
from relay import RelayGate


def replay_frames(frames, max_processing_lag=.2):
    ages = {'scan': .3, 'depth': .15}
    store = ProcessedEvidence(ages, max_processing_lag=max_processing_lag, max_pending=8)
    events = []
    for index, frame in enumerate(frames):
        if (not isinstance(frame, dict) or frame.get('stream') not in ages
                or frame.get('clear') is not None and type(frame.get('clear')) is not bool
                or not 0 <= frame['acquired'] <= frame['received'] <= frame['completed']):
            raise ValueError('invalid frame record')
        events.append((frame['received'], 0, index))
        events.append((frame['completed'], 1, index))
    events.sort()
    result = {'motion_requested': False, 'hypothetical_only': True, 'accepted_commands': 0,
              'longest_segment_sec': 0., 'segments': [], 'trace': [], 'final_state': 'waiting',
              'odom_continuity_checked': False, 'base_status_checked': False}
    ids, gate, segment, sequence = {}, None, None, 0
    index = 0
    while index < len(events):
        now = events[index][0]
        if gate is not None:
            gate.command(now)
            if gate.stopped:
                segment['stop_reason'] = gate.reason
                result['segments'].append(segment)
                gate = segment = None
        while index < len(events) and events[index][0] == now:
            _, kind, frame_index = events[index]
            frame = frames[frame_index]
            name = frame['stream']
            if kind == 0:
                ids[frame_index] = store.ingest(name, frame['acquired'], frame['received'], epoch=1)
            else:
                store.complete(name, ids.get(frame_index), frame['clear'], frame['valid_until'], now,
                               reason=frame.get('reason', 'sensor_failure'))
            result['trace'].append({'at': now, 'event': 'received' if kind == 0 else 'processed',
                                    'stream': name, 'frame_index': frame_index,
                                    'acquired': frame['acquired'], 'clear': frame['clear'] if kind else None})
            index += 1
        state = store.read(now, epoch=1)
        result['final_state'] = state['state']
        if state['state'] != 'ready':
            if gate is not None:
                gate.stop(state['reason'])
                segment['stop_reason'] = gate.reason
                result['segments'].append(segment)
                gate = segment = None
            continue
        if gate is None:
            gate = RelayGate(now, max_motion_sec=30., require_intent_deadline=True)
            sequence = 0
            segment = {'first_at': now, 'last_at': now, 'commands': 0}
        command = {'sequence': sequence, 'stamp': now, 'linear': .02, 'angular': 0.,
                   'expires_at': min(state['valid_until'], now + .16)}
        if not gate.accept(command, now):
            segment['stop_reason'] = gate.reason
            result['segments'].append(segment)
            gate = segment = None
            continue
        sequence += 1
        result['accepted_commands'] += 1
        segment['last_at'], segment['commands'] = now, segment['commands'] + 1
        result['trace'].append({'at': now, 'event': 'hypothetical_command',
                                'expires_at': command['expires_at']})
    if gate is not None:
        gate.command(gate.intent_deadline)
        segment['stop_reason'] = gate.reason
        result['segments'].append(segment)
    result['longest_segment_sec'] = max(
        [value['last_at'] - value['first_at'] for value in result['segments']] or [0.])
    return result
