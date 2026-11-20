"""Build a non-authoritative SHADOW step request from an advisory decision."""
import math
import re

_HASH = re.compile(r'^sha256:[0-9a-f]{64}$')
_ID = re.compile(r'^[A-Za-z0-9_-]{1,128}$')
_CHOICES = frozenset(('forward', 'left', 'right', 'stop'))


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('expected finite numeric value')
    return float(value)


def make_intent(decision, action_id, body_snapshot_hash, *, wall_now, monotonic_now, duration):
    """Translate times for the same host/boot only; this does not create a permit."""
    if not isinstance(decision, dict) or decision.get('status') != 'ok' or decision.get('executable') is not False:
        raise ValueError('decision is not a valid advisory result')
    choice, round_id, observation_hash = (decision.get(k) for k in ('choice', 'round_id', 'observation_hash'))
    if not isinstance(choice, str) or choice not in _CHOICES:
        raise ValueError('unsupported decision')
    for value in (action_id, round_id):
        if not isinstance(value, str) or not _ID.fullmatch(value):
            raise ValueError('invalid action or round id')
    for value in (body_snapshot_hash, observation_hash):
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ValueError('invalid body or observation hash')
    wall, mono, seconds = _number(wall_now), _number(monotonic_now), _number(duration)
    expiry_wall = _number(decision.get('expires_wall_time'))
    if wall <= 0 or mono < 0 or not 0 < seconds <= .5:
        raise ValueError('invalid clock or duration')
    # Preserve a small margin when translating wall-time validity to monotonic time.
    lifetime = min(.95, expiry_wall - wall - .05)
    if lifetime < seconds:
        raise ValueError('decision expires before the bounded step could finish')
    return {'schema_version': 'limo.mapless-step.v1', 'action_id': action_id, 'body_id': 'limo',
            'body_snapshot_hash': body_snapshot_hash, 'observation_hash': observation_hash,
            'round_id': round_id, 'primitive': choice, 'duration_sec': seconds,
            'created_monotonic': mono, 'expires_monotonic': mono + lifetime}
