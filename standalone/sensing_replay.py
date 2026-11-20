#!/usr/bin/env python2
"""Replay captured-frame timestamps through RelayGate, with no motion I/O.

This is an optimistic diagnostic: it does not replay newer raw obstacles,
pose changes, full geometry, the ROS driver, or actual transport scheduling.
The captured-frame assessments are not motion permits. Each rejected or
expired result ends its in-memory relay; later valid results start a new one.
"""
from __future__ import division, print_function

import argparse
import json
import math
import sys

from relay import RelayGate

try:
    NUMBER_TYPES = (int, long, float)  # noqa: F821
except NameError:
    NUMBER_TYPES = (int, float)


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
            and not math.isnan(value) and not math.isinf(value))


def _deadline(item, now):
    if item.get('reason') != 'ready':
        return None
    limits = {'scan_at': .3, 'depth_at': .15, 'odom_at': .15,
              'status_at': 1., 'snapshot_at': .25}
    values = []
    for name, age in limits.items():
        stamp = item.get(name)
        if not _finite(stamp) or not 0 <= stamp <= item['perception_end']:
            return None
        values.append(stamp + age)
    coverage = item.get('valid_until')
    if not _finite(coverage):
        return None
    deadline = min(values + [coverage])
    return deadline if now < deadline else None


def replay(events, period=.025):
    """Use the real RelayGate; replayed nonzero frames never leave memory."""
    if not _finite(period) or not 0 < period <= .05:
        raise ValueError('invalid period')
    result = {'motion_requested': False, 'hypothetical_only': True, 'segments': [],
              'rejected_observations': 0, 'accepted_observations': 0}
    gate = None
    segment = None
    next_tick = None
    proof = None
    sequence = 0
    last_event = None

    def finish(reason, now):
        segment['stop_reason'] = reason
        segment['stop_at'] = now
        result['segments'].append(segment)

    for item in events:
        if not isinstance(item, dict) or item.get('event') != 'perception' or not _finite(item.get('at')):
            raise ValueError('invalid timeline event')
        now = item['at']
        start, end = item.get('perception_start'), item.get('perception_end')
        if not _finite(start) or not _finite(end) or not 0 <= start <= end <= now:
            raise ValueError('invalid processing timeline')
        if last_event is not None and now <= last_event:
            raise ValueError('nonmonotonic timeline')
        last_event = now
        while gate is not None and next_tick < now:
            # The child deadline can occur before the next intended heartbeat.
            check_at = min(next_tick, gate.intent_deadline)
            gate.command(check_at)
            if gate.stopped:
                finish(gate.reason, check_at)
                gate = segment = None
                break
            expiry = min(proof, next_tick + .16)
            frame = {'sequence': sequence, 'stamp': next_tick, 'linear': .02,
                     'angular': 0., 'expires_at': expiry}
            if not gate.accept(frame, next_tick):
                finish(gate.reason, next_tick)
                gate = segment = None
                break
            sequence += 1
            segment['commands'] += 1
            segment['last_command_at'] = next_tick
            next_tick += period
        if gate is not None:
            gate.command(now)
            if gate.stopped:
                finish(gate.reason, now)
                gate = segment = None
        deadline = _deadline(item, now)
        if deadline is None:
            result['rejected_observations'] += 1
            if gate is not None:
                gate.stop('observation_rejected')
                finish(gate.reason, now)
                gate = segment = None
            continue
        result['accepted_observations'] += 1
        proof = deadline
        if gate is None:
            gate = RelayGate(now, max_motion_sec=30., require_intent_deadline=True)
            sequence = 0
            segment = {'first_command_at': now, 'last_command_at': now, 'commands': 0}
        # New proof updates the child deadline immediately, rather than
        # waiting for a fixed tick when the prior deadline is near.
        frame = {'sequence': sequence, 'stamp': now, 'linear': .02,
                 'angular': 0., 'expires_at': min(deadline, now + .16)}
        if not gate.accept(frame, now):
            finish(gate.reason, now)
            gate = segment = None
            continue
        sequence += 1
        segment['commands'] += 1
        segment['last_command_at'] = now
        next_tick = now + period
    if gate is not None:
        # Do not artificially renew beyond the final recorded evidence.
        gate.command(gate.intent_deadline)
        finish(gate.reason, gate.intent_deadline)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('timeline')
    args = parser.parse_args(argv)
    with open(args.timeline) as stream:
        events = [json.loads(line) for line in stream]
    result = replay(events)
    spans = [segment['last_command_at'] - segment['first_command_at'] for segment in result['segments']]
    result['max_segment_sec'] = max(spans) if spans else 0.
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
