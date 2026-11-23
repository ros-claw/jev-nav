"""Run local control with an asynchronous Decisions worker."""
from __future__ import division

import threading

from navigation import (PIVOTS, ROOM_OBSTACLE_LOOKAHEAD, Command, MAX_RESPONSE_AGE, Navigation,
                        ready, sweep_block, sweep_clear)


ROOM_DIRECTIONS = {'circulate_left': 'left', 'circulate_right': 'right'}
ROOM_ROUTE_REVIEW_SEC = 45.


def offered_direction(request, direction):
    return direction and any(start in request.choices for start in ('forward', direction, 'pivot_' + direction))


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
        self.review_origin = None
        self._review_after = 0.
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
            current_request = request is self.navigation.pending or self.review_origin is not None
            waiting_for_sensors = (choice not in (None, 'stop') and current_request
                                   and 0 <= now - request.at < MAX_RESPONSE_AGE and not ready(view, now))
            if not waiting_for_sensors:
                self._result = self._job = None
                direction = ROOM_DIRECTIONS.get(choice) if self.static_room else None
                reviewing = self.review_origin is not None
                if reviewing:
                    accepted = self._review_route(request, choice, direction, view, now)
                    self.review_origin = None
                else:
                    action = self._room_action(request, direction, view) if direction else choice
                    accepted = self.navigation.accept(request, action, view, now)
                    if self.static_room:
                        # A current route preference can select a new segment when its original sweep changed.
                        keep_direction = (current_request and not self.navigation.fault
                                          and 0 <= now - request.at < MAX_RESPONSE_AGE
                                          and self.navigation.anchor_current(request, view, now)
                                          and offered_direction(request, direction))
                        if direction and (accepted or keep_direction):
                            self.room_plan = request.sequence, direction
                            accepted = True
                        elif accepted and choice == 'stop':
                            self.room_plan = None
                if accepted and direction:
                    self._review_after = now + ROOM_ROUTE_REVIEW_SEC
                decision = (request.sequence, choice, accepted)
                if (not accepted and not reviewing) or (accepted and choice == 'stop'):
                    self._next_request = now + 1.

        command = self.navigation.step(view, now)
        survey = (self.room_plan is not None and command.reason == 'path_unknown'
                  and (self.navigation.goal.action in ('left', 'right') or self.navigation.goal.action in PIVOTS)
                  and sweep_clear(view.grid, view.pose, self.navigation.length, 0.))
        if survey:
            self.navigation.stop('path_unknown')
            self._last_grid = None
        if command.reason == 'clock_or_odometry_reset':
            self.room_plan = None
        elif command.reason == 'path_tracking_lost' and self.room_plan is not None:
            self._last_grid = None
        published = self.publish(command)
        if published is not None:
            command = published
            self.navigation.speed = command.linear
        offered = None
        if (not self._closed.is_set() and not self.navigation.fault and view is not None
                and (self._job is None or self.review_origin is not None and self.room_plan is not None)
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
                    if self._job is None and now >= self._review_after:
                        self.review_origin = source
                        self._review_after = now + ROOM_ROUTE_REVIEW_SEC
                    else:
                        offered = None
                else:
                    offered = None
            if offered is not None and self._job is None:
                self._job = offered
                self._wake.set()
            elif offered is not None:
                offered = None
        return command, offered, decision

    def _review_route(self, request, choice, direction, view, now):
        if (self.room_plan is None or self.room_plan[0] != self.review_origin
                or not ready(view, now) or request.epoch != view.epoch
                or not 0 <= now - request.at < MAX_RESPONSE_AGE):
            return False
        if offered_direction(request, direction):
            self.room_plan = request.sequence, direction
            return True
        if choice == 'stop':
            self.navigation.stop('model_stop')
            self.room_plan = None
            return True
        return False

    def _room_action(self, request, direction, view):
        blocked = self.navigation.candidate_blocks.get(direction)
        away = 'right' if direction == 'left' else 'left'
        for side in (direction, away):
            pivot = 'pivot_' + side
            if pivot in request.choices:
                return pivot
        if direction in request.choices:
            # Look farther for already observed obstacles. Both executable
            # choices still require a strict, shorter swept-area check.
            if (view is not None and away in request.choices
                    and sweep_block(view.grid, request.anchor, ROOM_OBSTACLE_LOOKAHEAD, self.navigation.curvature[direction],
                                    occupied_only=True)
                    and not sweep_block(view.grid, request.anchor, ROOM_OBSTACLE_LOOKAHEAD, self.navigation.curvature[away],
                                        occupied_only=True)):
                return away
            return direction
        if away in request.choices and ('forward' not in request.choices or blocked and blocked[0] == 'occupied'):
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
