"""Run local control with an asynchronous Decisions worker."""
from __future__ import division

import threading

from navigation import PIVOTS, Command, MAX_RESPONSE_AGE, Navigation, ready, sweep_block, sweep_clear


ROOM_DIRECTIONS = {'circulate_left': 'left', 'circulate_right': 'right'}


class NavigationRuntime(object):
    def __init__(self, decide, publish, navigation=None, static_room=False):
        self.navigation = Navigation() if navigation is None else navigation
        self.decide, self.publish = decide, publish
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._job = self._result = None
        self._next_request = 0.
        self._last_grid = None
        self.static_room = static_room
        self.room_plan = None
        self.room_continuation = None
        self._worker = threading.Thread(target=self._work, name='navigation-decisions')
        self._worker.daemon = True
        self._worker.start()

    def _work(self):
        while not self._closed.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._closed.is_set():
                return
            request = self._job
            try:
                choice = self.decide(request, self._closed.is_set)
            except Exception:
                choice = None
            self._result = (request, choice)

    def cycle(self, view, now):
        """Called by one periodic controller. Returns decision and motion telemetry."""
        result = self._result
        decision = None
        self.room_continuation = None
        if result is not None:
            request, choice = result
            waiting_for_sensors = (choice not in (None, 'stop') and request is self.navigation.pending
                                   and 0 <= now - request.at < MAX_RESPONSE_AGE and not ready(view, now))
            if not waiting_for_sensors:
                self._result = self._job = None
                direction = ROOM_DIRECTIONS.get(choice) if self.static_room else None
                action = self._room_action(request, direction, view) if direction else choice
                accepted = self.navigation.accept(request, action, view, now)
                if self.static_room:
                    self.room_plan = (request.sequence, direction) if accepted and direction else None
                decision = (request.sequence, choice, accepted)
                if not accepted or choice == 'stop':
                    self._next_request = now + 1.

        command = self.navigation.step(view, now)
        survey = (self.room_plan is not None and command.reason == 'path_unknown'
                  and (self.navigation.goal.action in ('left', 'right') or self.navigation.goal.action in PIVOTS)
                  and sweep_clear(view.grid, view.pose, self.navigation.length, 0.))
        if survey:
            self.navigation.stop('path_unknown')
            self._last_grid = None
        if command.reason in ('path_tracking_lost', 'clock_or_odometry_reset'):
            self.room_plan = None
        published = self.publish(command)
        if published is not None:
            command = published
            self.navigation.speed = command.linear
        offered = None
        if (not self._closed.is_set() and not self.navigation.fault and self._job is None and view is not None
                and now >= self._next_request and view.grid_at != self._last_grid):
            self._last_grid = view.grid_at
            offered = self.navigation.request(view, now)
            if self.room_plan is not None and not survey:
                source, direction = self.room_plan
                pivot = self.navigation.prepare_pivot(view, now, direction, offered)
                if pivot is not None:
                    self.room_continuation = (self.navigation.sequence, pivot, source)
                    offered = None
            if offered is not None and self.room_plan is not None:
                source, direction = self.room_plan
                action = 'forward' if survey else self._room_action(offered, direction, view)
                if action and self.navigation.accept(offered, action, view, now):
                    self.room_continuation = (offered.sequence, action, source)
                    offered = None
                else:
                    self.room_plan = None
            if offered is not None:
                self._job = offered
                self._wake.set()
        return command, offered, decision

    def _room_action(self, request, direction, view):
        blocked = self.navigation.candidate_blocks.get(direction)
        away = 'right' if direction == 'left' else 'left'
        if direction in request.choices:
            # Look farther for already observed obstacles. Both executable
            # choices still require a strict, shorter swept-area check.
            if (view is not None and away in request.choices
                    and sweep_block(view.grid, request.anchor, .60, self.navigation.curvature[direction],
                                    occupied_only=True)
                    and not sweep_block(view.grid, request.anchor, .60, self.navigation.curvature[away],
                                        occupied_only=True)):
                return away
            return direction
        if blocked and blocked[0] == 'occupied' and away in request.choices:
            return away
        return 'forward' if 'forward' in request.choices else None

    def close(self, publish_stop=True):
        self._closed.set()
        self._wake.set()
        try:
            if publish_stop:
                self.publish(Command(0., 0., 'stop', 'shutdown'))
        finally:
            self._worker.join(1.)
        return not self._worker.is_alive()
