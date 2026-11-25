"""Compare motion reported on the output topic with measured progress."""
from __future__ import division

from navigation import _relative


PROGRESS_FAULTS = ('no_progress', 'slip', 'pose_uncertain')
MAX_COMMAND_GAP = .25
MIN_MEASURED_SLOW_YAW_RAD = .02


class ProgressMonitor(object):
    def __init__(self, require_independent=False, track_segment=False):
        self.require_independent = require_independent
        self.track_segment = track_segment
        self.segment = None
        self.command_at = 0.
        self.epoch = None
        self.fault = None
        self.diagnostics = {}
        self._reset()

    def _reset(self):
        self.previous = self.wheel_anchor = self.independent_anchor = None
        self.expected = 0.
        self._previous_yaw_surplus = 0.

    def update(self, now, commands, wheel, epoch, independent=None, segment=None):
        if self.fault:
            return self.fault
        if self.track_segment and (segment is None or segment != self.segment or self.epoch != epoch):
            self._reset()
            self.segment, self.epoch = segment, epoch
            self.command_at = max(self.command_at, now)
            return None
        if (wheel is None and not self.track_segment) or self.epoch != epoch:
            self._reset()
        self.epoch = epoch
        if wheel is None and self.previous is not None:
            self.previous = self.previous[:3] + (0.,)
        for at, linear, angular in commands:
            if at <= self.command_at or at > now:
                continue
            self.command_at = at
            previous = self.previous
            if wheel is None or now - at >= MAX_COMMAND_GAP:
                if self.track_segment and previous is not None:
                    self.previous = previous[:2] + (at, 0.)
                else:
                    self._reset()
                continue
            if linear:
                axis, speed = 'x', linear
            elif abs(angular) >= .025:
                axis, speed = 'yaw', angular
            elif self.track_segment and previous is not None:
                if 0. < at - previous[2] < MAX_COMMAND_GAP:
                    self.expected += (at - previous[2]) * previous[3]
                self.previous = previous[:2] + (at, 0.)
                continue
            else:
                self._reset()
                continue
            direction = 1 if speed > 0 else -1
            same_axis = previous is not None and previous[:2] == (axis, direction)
            if same_axis and 0. < at - previous[2] < MAX_COMMAND_GAP:
                self.expected += (at - previous[2]) * previous[3]
            elif not (self.track_segment and same_axis):
                self.expected = 0.
                self._previous_yaw_surplus = 0.
                self.wheel_anchor, self.independent_anchor = wheel, independent
            self.previous = axis, direction, at, abs(speed)
        if wheel is None or self.previous is None or now - self.command_at >= MAX_COMMAND_GAP:
            if self.track_segment and self.previous is not None:
                self.previous = self.previous[:3] + (0.,)
            else:
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
        slow_yaw_progress = False
        if self.track_segment and axis == 'yaw' and independent is None and not self.require_independent:
            # A pulsed pivot can have one slow window after a strong one.
            # Require real motion in both windows and consume the credit once.
            slow_yaw_progress = (MIN_MEASURED_SLOW_YAW_RAD <= wheel_progress < minimum
                                 and wheel_progress + self._previous_yaw_surplus >= minimum)
            self._previous_yaw_surplus = max(0., wheel_progress - minimum)
        else:
            self._previous_yaw_surplus = 0.
        if independent_progress is None and self.require_independent:
            self.fault = 'pose_uncertain'
        elif wheel_progress < minimum and not slow_yaw_progress:
            self.fault = 'no_progress' if independent_progress is None or independent_progress < minimum else 'pose_uncertain'
        elif independent_progress is not None and independent_progress < minimum:
            self.fault = 'slip'
        self.diagnostics = {'axis': axis, 'expected': self.expected, 'wheel': wheel_progress,
                            'independent': independent_progress, 'reason': self.fault}
        if slow_yaw_progress:
            self.diagnostics['combined_yaw_windows'] = True
        self.wheel_anchor, self.independent_anchor = wheel, independent
        self.expected = 0.
        return self.fault
