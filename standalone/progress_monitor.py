"""Compare motion reported on the output topic with measured progress."""
from __future__ import division

from navigation import _relative


PROGRESS_FAULTS = ('no_progress', 'slip', 'pose_uncertain')
MAX_COMMAND_GAP = .25


class ProgressMonitor(object):
    def __init__(self, require_independent=False):
        self.require_independent = require_independent
        self.command_at = 0.
        self.epoch = None
        self.fault = None
        self.diagnostics = {}
        self._reset()

    def _reset(self):
        self.previous = self.wheel_anchor = self.independent_anchor = None
        self.expected = 0.

    def update(self, now, commands, wheel, epoch, independent=None):
        if self.fault:
            return self.fault
        if wheel is None or self.epoch != epoch:
            self._reset()
        self.epoch = epoch
        for at, linear, angular in commands:
            if at <= self.command_at or at > now:
                continue
            self.command_at = at
            if wheel is None or now - at >= MAX_COMMAND_GAP:
                self._reset()
                continue
            if linear:
                axis, speed = 'x', linear
            elif not linear and abs(angular) >= .025:
                axis, speed = 'yaw', angular
            else:
                self._reset()
                continue
            direction = 1 if speed > 0. else -1
            previous = self.previous
            if previous and previous[:2] == (axis, direction) and 0. < at - previous[2] < MAX_COMMAND_GAP:
                self.expected += (at - previous[2]) * previous[3]
            else:
                self.expected = 0.
                self.wheel_anchor, self.independent_anchor = wheel, independent
            self.previous = axis, direction, at, abs(speed)
        if self.previous is None or now - self.command_at >= MAX_COMMAND_GAP:
            self._reset()
            return None
        axis, direction = self.previous[:2]
        if self.expected + 1e-9 < (.06 if axis == 'x' else .15):
            return None
        wheel_progress = direction * getattr(_relative(self.wheel_anchor, wheel), axis)
        independent_progress = None
        if independent is not None and self.independent_anchor is not None:
            frame, anchor = self.independent_anchor
            if frame == independent[0]:
                independent_progress = direction * getattr(_relative(anchor, independent[1]), axis)
        minimum = self.expected * .2
        if independent_progress is None and self.require_independent:
            self.fault = 'pose_uncertain'
        elif wheel_progress < minimum:
            self.fault = 'no_progress' if independent_progress is None or independent_progress < minimum else 'pose_uncertain'
        elif independent_progress is not None and independent_progress < minimum:
            self.fault = 'slip'
        self.diagnostics = {'axis': axis, 'expected': self.expected, 'wheel': wheel_progress,
                            'independent': independent_progress, 'reason': self.fault}
        self.wheel_anchor, self.independent_anchor = wheel, independent
        self.expected = 0.
        return self.fault
