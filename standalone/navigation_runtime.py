"""Run local control with an asynchronous Decisions worker."""
from __future__ import division

import math
import threading
from collections import namedtuple

from navigation import (ROOM_OBSTACLE_LOOKAHEAD, ROOM_TRACK_LOOKAHEAD, Command, MAX_RESPONSE_AGE, Navigation,
                        _relative, observation_deadline, ready, sweep_block)


ROOM_DIRECTIONS = {'circulate_left': 'left', 'circulate_right': 'right'}
ROOM_ROUTE_REVIEW_SEC = 45.
ROOM_PREDICTION_PERIOD = .25
ROOM_PREDICTION_BUDGET = .055
ROOM_DETOUR_COOLDOWN = 1.
ROOM_DETOUR_HOLD_M = .24
ROOM_DETOUR_HOLD_SEC = 20.
ROOM_EXIT_COVERAGE_M = .20
MappedChoice = namedtuple('MappedChoice', 'action target_id')


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
        self._blocked_replacement = None
        self.static_room = static_room
        self.room_plan = None
        self.room_continuation = None
        self.room_replacement = False
        self.review_origin = None
        self._review_after = 0.
        self._predict_after = 0.
        self._prediction_grid = None
        self.early_detour = None
        self.occupied_detour = None
        self._detour = None
        self.detour_continuation = False
        self.selected_target_id = None
        self.selected_target_stage = None
        self._settling = None
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
        self.room_replacement = False
        self.early_detour = None
        self.occupied_detour = None
        self.detour_continuation = False
        self.selected_target_id = None
        self.selected_target_stage = None
        if result is not None:
            request, choice = result
            mapped = choice if isinstance(choice, MappedChoice) else None
            if mapped is not None:
                choice = mapped.action
            current_request = request is self.navigation.pending or self.review_origin is not None
            waiting_for_sensors = (choice not in (None, 'stop') and current_request
                                   and 0 <= now - request.at < MAX_RESPONSE_AGE and not ready(view, now))
            if not waiting_for_sensors:
                self._result = self._job = None
                direction = ROOM_DIRECTIONS.get(choice) if self.static_room else None
                if mapped is not None and self.static_room:
                    self.selected_target_id = mapped.target_id
                    if choice in ('left', 'right', 'pivot_left', 'pivot_right'):
                        direction = choice.rsplit('_', 1)[-1]
                    elif choice == 'forward':
                        direction = (self.room_plan[1] if self.room_plan is not None else
                                     'right' if 'right' in request.choices and 'left' not in request.choices else 'left')
                reviewing = self.review_origin is not None
                if mapped is not None:
                    self.selected_target_stage = 'mapped_route_preference' if reviewing else 'mapped_target'
                if reviewing:
                    accepted = (mapped is None or self._target_current(request, choice, view, now))
                    accepted = accepted and self._review_route(request, choice, direction, view, now)
                    self.review_origin = None
                else:
                    action = self._room_action(request, direction, view) if direction and mapped is None else choice
                    accepted = self.navigation.accept(request, action, view, now)
                    if self.static_room:
                        # A current route preference can select a new segment when its original sweep changed.
                        keep_direction = (current_request and not self.navigation.fault
                                          and 0 <= now - request.at < MAX_RESPONSE_AGE
                                          and self.navigation.anchor_current(request, view, now)
                                          and offered_direction(request, direction))
                        if direction and (accepted or mapped is None and keep_direction):
                            self.room_plan = request.sequence, direction
                            accepted = True
                        elif accepted and choice == 'stop':
                            self.room_plan = None
                if accepted and direction:
                    self._review_after = now + ROOM_ROUTE_REVIEW_SEC
                decision = (request.sequence, choice, accepted)
                if (not accepted and not reviewing) or (accepted and choice == 'stop'):
                    self._next_request = now + 1.

        last_at = self.navigation.last_at
        motion = self.navigation.speed, self.navigation.steer, .05 if last_at is None else max(0., now - last_at)
        direction = self.room_plan[1] if self.room_plan is not None else None
        command = self.navigation.step(view, now, direction)
        if command.reason == 'occupied_detour':
            self.occupied_detour = self.navigation.blocked_action, command.action, self.navigation.blocked_cell
            self._predict_after = now + ROOM_DETOUR_COOLDOWN
            self._detour = ((command.action, direction, view.pose, now)
                            if command.action in ('left', 'right') else None)
        if (self.room_plan is not None and command.linear > 0. and now >= self._predict_after
                and view.grid_at != self._prediction_grid):
            until = observation_deadline(view, now)
            if until is not None and until - now >= ROOM_PREDICTION_BUDGET:
                self._prediction_grid = view.grid_at
                self._predict_after = now + ROOM_PREDICTION_PERIOD
                previous = command.action
                detour = self.navigation.anticipate_obstacle(view, now, command, self.room_plan[1])
                if detour is not None:
                    command, block = detour
                    self.early_detour = previous, block
                    self._predict_after = now + ROOM_DETOUR_COOLDOWN
                    self._detour = ((command.action, self.room_plan[1], view.pose, now)
                                    if command.linear > 0. and command.action in ('left', 'right') else None)
        if command.reason not in ('path_unknown', 'path_reacquiring'):
            self._blocked_replacement = None
        replacing = (not self._closed.is_set() and self.room_plan is not None and command.reason == 'path_unknown')
        settling_pivot = replacing and self.navigation.goal.action.startswith('pivot_')
        can_settle = self.room_plan is not None and (command.reason == 'waiting_for_goal' or settling_pivot)
        settled = can_settle and self._settled(view, now)
        if not can_settle:
            self._settling = None
        if command.reason == 'clock_or_odometry_reset':
            self.room_plan = None
        elif command.reason == 'path_tracking_lost' and self.room_plan is not None:
            self._last_grid = None
        if self.room_plan is None:
            self._detour = None
        if replacing and (not settling_pivot or settled):
            source, direction = self.room_plan
            geometry = (self.navigation.goal.sequence, view.epoch, view.pose, view.grid, direction, motion[:2])
            if geometry != self._blocked_replacement:
                replacement = self.navigation.replace_unknown_path(view, now, direction, motion)
                if replacement is not None:
                    command = replacement
                    self.room_continuation = (self.navigation.sequence, self.navigation.goal.action, source)
                    self.room_replacement = True
                    self._blocked_replacement = None
                else:
                    self._blocked_replacement = geometry
        published = self.publish(command)
        if published is not None:
            command = published
            self.navigation.speed = command.linear
        if replacing:
            return command, None, decision
        offered = None
        if (not self._closed.is_set() and not self.navigation.fault and view is not None
                and (self._job is None or self.review_origin is not None and self.room_plan is not None)
                and now >= self._next_request and view.grid_at != self._last_grid):
            self._last_grid = view.grid_at
            offered = self.navigation.request(view, now)
            held = None
            if self.room_plan is not None:
                source, direction = self.room_plan
                if offered is not None:
                    held = self._held_detour(offered, direction, view, now)
                if held is None:
                    pivot = self.navigation.prepare_pivot(view, now, direction, offered, allow_wide=settled)
                    if pivot is not None:
                        self.room_continuation = (self.navigation.sequence, pivot, source)
                        offered = None
            if offered is not None and self.room_plan is not None:
                source, direction = self.room_plan
                action = held or self._room_action(offered, direction, view)
                if action and self.navigation.accept(offered, action, view, now):
                    self.room_continuation = (offered.sequence, action, source)
                    self.detour_continuation = held is not None
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

    def _target_current(self, request, action, view, now):
        if action not in request.choices or not self.navigation.anchor_current(request, view, now):
            return False
        if action.startswith('pivot_'):
            return self.navigation._pivot_clear(view, request.anchor, action)
        return (action in self.navigation.curvature and sweep_block(
            view.grid, request.anchor, self.navigation.length, self.navigation.curvature[action], view.pose) is None)

    def wait_stage(self, decision):
        if self.navigation.goal is not None:
            return 'starting_segment'
        if self._job is not None:
            return 'model_pending'
        if decision is not None and not decision[2]:
            return 'model_rejected'
        if self.navigation.completed is not None:
            return 'successor_unavailable'
        return 'no_geometry'

    def _held_detour(self, request, direction, view, now):
        """Carry a safe turn through an occlusion while keeping the model's route."""
        if self._detour is None:
            return None
        action, model_direction, start, started_at = self._detour
        if (direction != model_direction or not 0 <= now - started_at < ROOM_DETOUR_HOLD_SEC
                or math.hypot(view.pose.x - start.x, view.pose.y - start.y) >= ROOM_DETOUR_HOLD_M
                or action not in request.choices
                or sweep_block(view.grid, request.anchor, ROOM_TRACK_LOOKAHEAD,
                               self.navigation.curvature[action], view.pose, occupied_only=True)):
            self._detour = None
            return None
        return action

    def _settled(self, view, now):
        if self._settling is not None:
            at, pose, odom_at = self._settling
            motion = _relative(pose, view.pose)
            if math.hypot(motion.x, motion.y) <= .003 and abs(motion.yaw) <= .012:
                return now - at >= .25 and view.odom_at > odom_at
        self._settling = now, view.pose, view.odom_at
        return False

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
            if view is not None and self.navigation.static_room:
                missing = sweep_block(view.grid, request.anchor, ROOM_EXIT_COVERAGE_M,
                                      self.navigation.curvature[direction], view.pose)
                if missing is not None and missing[0] == 'unknown':
                    for option in ('forward', away):
                        if option in request.choices and sweep_block(
                                view.grid, request.anchor, ROOM_EXIT_COVERAGE_M,
                                self.navigation.curvature[option], view.pose) is None:
                            return option
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
