"""Synthetic replay controller only; no transport, ROS, daemon or hardware API."""
import importlib.util
import math
from pathlib import Path

import sweep_simulation

PRIMITIVES = {'forward': (.1, 0), 'left': (0, .3), 'right': (0, -.3)}
FOOTPRINT = (.18, .18, .13)  # configured differential footprint plus 0.02 m padding
_SWEEP_PATH = Path(__file__).resolve().parent / 'worktrees/limo-mcp/worker/limo_forward_sweep.py'
_SPEC = importlib.util.spec_from_file_location('limo_forward_sweep', _SWEEP_PATH)
_FORWARD_SWEEP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FORWARD_SWEEP)


class ReplayController:
    """A deterministic simulation of short decisions with a latched stop."""
    def __init__(self):
        self._active = False
        self._last = None
        self._deadline = None
        self._primitive = (0, 0)
        self._reason = 'idle'

    def begin(self, decision, current_round, now, expires_at):
        self._active = False
        self._reason = 'invalid_decision'
        try:
            if not math.isfinite(now) or not math.isfinite(expires_at) or not now < expires_at <= now + 1:
                return False
            if (decision.get('status') != 'ok' or decision.get('executable') is not False
                    or not current_round or decision.get('round_id') != current_round
                    or decision.get('choice') not in PRIMITIVES):
                return False
            self._primitive = PRIMITIVES[decision['choice']]
            self._last = now
            self._deadline = expires_at
            self._active = True
            self._reason = 'active'
            return True
        except (TypeError, ValueError, AttributeError):
            return False

    def _stop(self, reason):
        self._active = False
        self._reason = reason
        return {'simulated_linear_mps': 0, 'simulated_angular_radps': 0,
                'executable': False, 'reason': reason}

    def step(self, now, grid, pose):
        if not self._active:
            return self._stop(self._reason)
        if not isinstance(now, (int, float)) or not math.isfinite(now) or now < self._last:
            return self._stop('invalid_clock')
        if now >= self._deadline:
            return self._stop('decision_expired')
        if now - self._last > .25:
            return self._stop('lease_expired')
        linear, angular = self._primitive
        if linear > 0 and angular == 0:
            sweep = _FORWARD_SWEEP.check_forward_sweep(
                grid, pose, FOOTPRINT, linear * .1, now, max_age=.25)
        else:
            radius = math.hypot(max(FOOTPRINT[:2]), FOOTPRINT[2])
            sweep = sweep_simulation.check_sweep(
                grid, pose, linear, angular, .1, radius, now, max_age=.25)
        if sweep['status'] != 'clear_in_simulation':
            return self._stop(sweep['status'])
        self._last = now
        return {'simulated_linear_mps': linear, 'simulated_angular_radps': angular,
                'executable': False, 'reason': 'active'}
