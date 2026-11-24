"""Odometry-frame local navigation. A selected segment lives until a local event."""
from __future__ import division

from collections import namedtuple
import math


Pose = namedtuple('Pose', 'x y yaw')
Grid = namedtuple('Grid', 'x y resolution width height cells')
View = namedtuple('View', 'grid_at scan_at depth_at odom_at status_at epoch pose grid ready')
Request = namedtuple('Request', 'sequence at epoch anchor choices predecessor blocks')
Command = namedtuple('Command', 'linear angular action reason')
Segment = namedtuple('Segment', 'sequence action origin turn')

LENGTH = .24
ROOM_LENGTH = .12
ROOM_OBSTACLE_LOOKAHEAD = .60
ROOM_TRACK_LOOKAHEAD = .36
FORWARD_SUCCESSOR_LEAD = .15
CURVATURE = {'forward': 0., 'left': .8, 'right': -.8}
ROOM_CURVATURE = {'forward': 0., 'left': 1.35, 'right': -1.35}
PIVOTS = {'pivot_left': .70, 'pivot_right': -.70}
ROOM_CORNER_TURNS = (.70, 1.40, 2.10, 2.45)
SPEED = .02
MAX_RESPONSE_AGE = 8.
HALF_LENGTH, HALF_WIDTH = .16, .11
STOP_DISTANCE = .01
ROOM_PATH_CLEAR_SEC = .40


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def arc_pose(pose, distance, curvature):
    angle = curvature * distance
    if abs(curvature) < 1e-8:
        return Pose(pose.x + distance * math.cos(pose.yaw),
                    pose.y + distance * math.sin(pose.yaw), pose.yaw)
    yaw = pose.yaw + angle
    return Pose(pose.x + (math.sin(yaw) - math.sin(pose.yaw)) / curvature,
                pose.y + (math.cos(pose.yaw) - math.cos(yaw)) / curvature, _wrap(yaw))


def _relative(origin, pose):
    dx, dy = pose.x - origin.x, pose.y - origin.y
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    return Pose(c * dx + s * dy, -s * dx + c * dy, _wrap(pose.yaw - origin.yaw))


def _overlap(points, base, grid, column, row):
    """Clip the swept rectangle to a cell, then test it against the known body."""
    for axis, index, origin in ((0, column, grid.x), (1, row, grid.y)):
        low = origin + index * grid.resolution
        for sign, edge in ((1., low), (-1., low + grid.resolution)):
            if not points:
                return False, False
            result = []
            last = points[-1]
            before = sign * (last[axis] - edge)
            for point in points:
                after = sign * (point[axis] - edge)
                if (before >= 0) != (after >= 0):
                    fraction = before / (before - after)
                    result.append((last[0] + fraction * (point[0] - last[0]),
                                   last[1] + fraction * (point[1] - last[1])))
                if after >= 0:
                    result.append(point)
                last, before = point, after
            points = result
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    for x, y in points:
        dx, dy = x - base.x, y - base.y
        if abs(c * dx + s * dy) > HALF_LENGTH + 1e-8 or abs(-s * dx + c * dy) > HALF_WIDTH + 1e-8:
            return True, True
    return bool(points), False


def _corners(pose, front):
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return [(pose.x + c * x - s * y, pose.y + s * x + c * y)
            for x, y in ((-HALF_LENGTH, -HALF_WIDTH), (front, -HALF_WIDTH),
                         (front, HALF_WIDTH), (-HALF_LENGTH, HALF_WIDTH))]


def _hull(points):
    ordered = sorted(set(points))

    def half(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2:
                a, b = result[-2:]
                if (b[0] - a[0]) * (point[1] - a[1]) > (b[1] - a[1]) * (point[0] - a[0]):
                    break
                result.pop()
            result.append(point)
        return result[:-1]

    return half(ordered) + half(reversed(ordered))


def _polygon_block(grid, polygon, known_body, occupied_only=False):
    x_min, x_max = min(p[0] for p in polygon), max(p[0] for p in polygon)
    y_min, y_max = min(p[1] for p in polygon), max(p[1] for p in polygon)

    def to_index(value, offset):
        return int(math.floor((value - offset) / grid.resolution))

    for row in range(to_index(y_min, grid.y), to_index(y_max, grid.y) + 1):
        for column in range(to_index(x_min, grid.x), to_index(x_max, grid.x) + 1):
            within = 0 <= row < grid.height and 0 <= column < grid.width
            cell = grid.cells[row * grid.width + column] if within else -1
            if cell == 0 or occupied_only and cell == -1:
                continue
            overlaps, new_space = _overlap(polygon, known_body, grid, column, row)
            if overlaps and (cell != -1 or new_space):
                return ('unknown' if cell == -1 else 'occupied',
                        grid.x + (column + .5) * grid.resolution, grid.y + (row + .5) * grid.resolution)
    return None


def sweep_block(grid, pose, distance, curvature, known_body=None, occupied_only=False):
    """Keep unknown outside the already occupied body out of a swept corridor."""
    if known_body is None:
        known_body = pose
    if abs(curvature) < 1e-8:
        return _polygon_block(grid, _corners(pose, HALF_LENGTH + distance), known_body, occupied_only)
    steps = max(1, int(math.ceil(distance / .025)))
    stride = distance / steps
    cx = pose.x - math.sin(pose.yaw) / curvature
    cy = pose.y + math.cos(pose.yaw) / curvature
    scale = 1. / math.cos(curvature * stride * .5)
    start = _corners(pose, HALF_LENGTH)
    for index in range(steps):
        middle = _corners(arc_pose(pose, (index + .5) * stride, curvature), HALF_LENGTH)
        end = _corners(arc_pose(pose, (index + 1) * stride, curvature), HALF_LENGTH)
        # Each corner's arc is enclosed by its endpoints and the intersection
        # of their tangents. The convex hull encloses the body between poses.
        outer = [(cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in middle]
        blocked = _polygon_block(grid, _hull(start + end + outer), known_body, occupied_only)
        if blocked is not None:
            return blocked
        start = end
    return None


def sweep_clear(grid, pose, distance, curvature, known_body=None):
    return sweep_block(grid, pose, distance, curvature, known_body) is None


def rotation_block(grid, pose, angle, known_body=None):
    body = pose if known_body is None else known_body
    steps = max(1, int(math.ceil(abs(angle) / .05)))
    stride = angle / steps
    scale = 1. / math.cos(stride / 2.)
    start = _corners(pose, HALF_LENGTH)
    for index in range(steps):
        middle = _corners(pose._replace(yaw=pose.yaw + (index + .5) * stride), HALF_LENGTH)
        end = _corners(pose._replace(yaw=pose.yaw + (index + 1) * stride), HALF_LENGTH)
        outer = [(pose.x + (x - pose.x) * scale, pose.y + (y - pose.y) * scale) for x, y in middle]
        blocked = _polygon_block(grid, _hull(start + end + outer), body)
        if blocked is not None:
            return blocked
        start = end
    return None


def observation_deadline(view, now):
    if view is None or view.ready is not True:
        return None
    sources = ((view.grid_at, .25), (view.scan_at, .30), (view.depth_at, .20),
               (view.odom_at, .15), (view.status_at, 1.))
    if all(0 <= now - stamp < limit for stamp, limit in sources):
        return min(stamp + limit for stamp, limit in sources)
    return None


def ready(view, now):
    return observation_deadline(view, now) is not None


def _remaining(goal, pose, length, curvature):
    local = _relative(goal.origin, pose)
    distance = (math.atan2(curvature * local.x, 1. - curvature * local.y) / curvature
                if curvature else local.x)
    return length - distance


class Navigation(object):
    """Single-writer task state. Feed immutable observations and finished choices."""

    def __init__(self, forward_only=False, speed=SPEED, successor_lead=None, static_room=False):
        if not 0. < speed <= SPEED:
            raise ValueError('invalid navigation speed')
        self.forward_only, self.max_speed = forward_only, speed
        self.static_room = static_room
        self.length = ROOM_LENGTH if static_room else LENGTH
        self.curvature = ROOM_CURVATURE if static_room else CURVATURE
        self.successor_lead = successor_lead or (FORWARD_SUCCESSOR_LEAD if forward_only else .09)
        self.goal = self.successor = self.pending = None
        self.sequence = 0
        self.epoch = self.last_at = self.completed = None
        self.fault = None
        self.speed = 0.
        self.steer = 0.
        self.paused = False
        self.blocked_cell = self.blocked_action = None
        self.candidate_blocks = {}
        self._blocked_corner = None
        self._recovering_unknown = False
        self._path_clear_since = None

    def stop(self, reason):
        self.goal = self.successor = self.pending = self.completed = None
        self.paused = False
        self._recovering_unknown = False
        self._path_clear_since = None
        self.speed = self.steer = 0.
        return Command(0., 0., 'stop', reason)

    def _pause(self, reason):
        self.speed = self.steer = 0.
        self.paused = True
        if self.static_room:
            self._recovering_unknown = self._recovering_unknown or reason == 'path_unknown'
            self._path_clear_since = None
        return Command(0., 0., 'stop', reason)

    def _path_reacquiring(self, view):
        if not self._recovering_unknown:
            return False
        # Rechecking the same frame at the control rate adds no evidence.
        observed_at = min(view.scan_at, view.depth_at)
        if self._path_clear_since is None:
            self._path_clear_since = observed_at
        if observed_at - self._path_clear_since < ROOM_PATH_CLEAR_SEC:
            return True
        self._recovering_unknown = False
        self._path_clear_since = None
        return False

    def request(self, view, now):
        self.candidate_blocks = {}
        if self.fault or self.paused or self.successor is not None or not ready(view, now):
            return None
        actions = ('forward',) if self.forward_only else ('forward', 'left', 'right')
        if self.goal is not None and self.goal.action in PIVOTS:
            end_yaw = _wrap(self.goal.origin.yaw + self.goal.turn)
            if abs(_wrap(end_yaw - view.pose.yaw)) > .12:
                return None
            anchor, predecessor = view.pose._replace(yaw=end_yaw), self.goal.sequence
            actions = ('forward',)
        elif self.goal is not None:
            remaining = _remaining(self.goal, view.pose, self.length, self.curvature[self.goal.action])
            if remaining > self.successor_lead:
                return None
            if self.forward_only:
                # A straight continuation follows the current measured line.
                # The predecessor's original heading may differ slightly.
                anchor = arc_pose(view.pose, max(0., remaining), 0.)
            else:
                anchor = arc_pose(self.goal.origin, self.length, self.curvature[self.goal.action])
            predecessor = self.goal.sequence
        else:
            anchor, predecessor = view.pose, None
        choices = []
        for action in actions:
            block = sweep_block(view.grid, anchor, self.length, self.curvature[action], view.pose)
            if block is None:
                choices.append(action)
            else:
                self.candidate_blocks[action] = block
        if (not choices and self.static_room and self.goal is None
                and any(block[0] == 'occupied' for block in self.candidate_blocks.values())):
            choices = [action for action in ('pivot_left', 'pivot_right') if self._pivot_clear(view, anchor, action)]
        if not choices:
            return None
        self.sequence += 1
        blocks = tuple((action,) + self.candidate_blocks[action] for action in actions
                       if action in self.candidate_blocks)
        self.pending = Request(self.sequence, now, view.epoch, anchor, tuple(choices) + ('stop',), predecessor,
                               blocks)
        return self.pending

    def prepare_pivot(self, view, now, direction, request=None, allow_wide=False):
        if (not self.static_room or self.fault or self.paused or self.successor is not None
                or not ready(view, now) or self.goal is not None and request is None
                or self.goal is not None and self.goal.action in PIVOTS
                or request is not None and request is not self.pending):
            return None
        anchor = view.pose if request is None else request.anchor
        if not sweep_block(view.grid, anchor, .42, 0., occupied_only=True):
            return None
        if request is not None and any(
                action in request.choices and not sweep_block(
                    view.grid, anchor, ROOM_OBSTACLE_LOOKAHEAD, self.curvature[action], occupied_only=True)
                for action in ('left', 'right')):
            return None
        away = 'right' if direction == 'left' else 'left'
        corner = ((view.epoch, view.pose, view.grid, direction)
                  if allow_wide and self.goal is None and request is None else None)
        if corner is not None and corner == self._blocked_corner:
            return None
        for angle in ROOM_CORNER_TURNS if corner is not None else (.70,):
            for option in (direction, away):
                action = 'pivot_' + option
                turn = math.copysign(angle, PIVOTS[action])
                if not self._pivot_clear(view, anchor, action, turn):
                    continue
                if request is None:
                    self.sequence += 1
                self.pending = self._blocked_corner = None
                pivot = Segment(self.sequence, action, anchor, turn)
                if self.goal is None:
                    self.goal = pivot
                else:
                    self.successor = pivot
                return action
        if corner is not None:
            self._blocked_corner = corner
        return None

    def replace_unknown_path(self, view, now, direction):
        """Replace a paused path only with a fresh sweep from the measured pose."""
        if (self.fault or self.forward_only or not self.paused or self.goal is None
                or self.blocked_cell is None or self.blocked_cell[0] != 'unknown' or not ready(view, now)
                or view.epoch != self.epoch):
            return None
        self.candidate_blocks = {}
        away = 'right' if direction == 'left' else 'left'
        for action in ('forward', direction, away):
            if action == self.goal.action:
                continue
            block = sweep_block(view.grid, view.pose, self.length, self.curvature[action])
            if block is None:
                break
            self.candidate_blocks[action] = block
        else:
            if not self.static_room or not sweep_block(view.grid, view.pose, .42, 0., occupied_only=True):
                return None
            for action in ('pivot_' + direction, 'pivot_' + away):
                if action != self.goal.action and self._pivot_clear(view, view.pose, action):
                    break
            else:
                return None
        self.stop('path_unknown')
        self.sequence += 1
        self.goal = Segment(self.sequence, action, view.pose, PIVOTS.get(action))
        return action

    def anticipate_obstacle(self, view, now, command, direction):
        """Start a checked detour while the current setpoint still passes its sweep."""
        if (not self.static_room or command.linear <= 0. or self.goal is None
                or self.goal.action != command.action or not ready(view, now)
                or view.epoch != self.epoch or self.successor is not None and self.successor.action in PIVOTS):
            return None
        curvature = self.curvature[self.goal.action]
        distance = ROOM_TRACK_LOOKAHEAD
        if self.successor is not None:
            distance = min(distance, max(0., _remaining(self.goal, view.pose, self.length, curvature)))
        block = sweep_block(view.grid, view.pose, distance, curvature, occupied_only=True)
        if block is None and self.successor is not None and distance < ROOM_TRACK_LOOKAHEAD:
            block = sweep_block(view.grid, self.successor.origin, ROOM_TRACK_LOOKAHEAD - distance,
                                self.curvature[self.successor.action], view.pose, occupied_only=True)
        if block is None:
            return None
        away = 'right' if direction == 'left' else 'left'
        for action in (direction, away, 'forward'):
            if action == self.goal.action:
                continue
            candidate = self.curvature[action]
            if (sweep_clear(view.grid, view.pose, self.length, candidate)
                    and not sweep_block(view.grid, view.pose, ROOM_TRACK_LOOKAHEAD, candidate, occupied_only=True)):
                self.sequence += 1
                self.goal = Segment(self.sequence, action, view.pose, None)
                self.successor = self.pending = self.completed = None
                # Reuse the tracker's steering and swept-body gate for the published command.
                return self._track(view, 0.), block
        return None

    def _pivot_clear(self, view, anchor, action, turn=None):
        angle = PIVOTS[action] if turn is None else turn
        turned = anchor._replace(yaw=_wrap(anchor.yaw + angle))
        return (sweep_clear(view.grid, turned, self.length, 0., view.pose)
                and sweep_block(view.grid, turned, .31, 0., occupied_only=True) is None
                and rotation_block(view.grid, anchor, angle, view.pose) is None)

    def anchor_current(self, request, view, now):
        if not ready(view, now) or request.epoch != view.epoch:
            return False
        active_predecessor = self.goal is not None and request.predecessor == self.goal.sequence
        anchor_limit = max(.10, self.successor_lead) if active_predecessor else .10
        return (math.hypot(view.pose.x - request.anchor.x, view.pose.y - request.anchor.y) <= anchor_limit
                and abs(_wrap(view.pose.yaw - request.anchor.yaw)) <= .20)

    def accept(self, request, choice, view, now):
        if (request is None or request is not self.pending or self.fault
                or not 0 <= now - request.at < MAX_RESPONSE_AGE
                or choice not in request.choices):
            return False
        self.pending = None
        if choice == 'stop':
            self.stop('model_stop')
            return True
        active_predecessor = self.goal is not None and request.predecessor == self.goal.sequence
        # An early response remains tied to the segment still being followed.
        if not self.anchor_current(request, view, now):
            return False
        if choice in PIVOTS:
            offset = _relative(request.anchor, view.pose)
            if (math.hypot(offset.x, offset.y) > .015 or abs(offset.yaw) > .05
                    or not self._pivot_clear(view, request.anchor, choice)):
                return False
        elif not sweep_clear(view.grid, request.anchor, self.length, self.curvature[choice], view.pose):
            return False
        segment = Segment(request.sequence, choice, request.anchor, PIVOTS.get(choice))
        if active_predecessor:
            self.successor = segment
        elif self.goal is None and request.predecessor in (None, self.completed):
            self.goal = segment
        else:
            return False
        return True

    def step(self, view, now):
        self.blocked_cell = self.blocked_action = None
        elapsed = .05 if self.last_at is None else now - self.last_at
        if elapsed < 0 or (view is not None and self.epoch is not None and view.epoch != self.epoch):
            self.fault = 'clock_or_odometry_reset'
        self.last_at = now
        if self.fault:
            return self.stop(self.fault)
        if not ready(view, now):
            return self._pause('perception_unavailable')
        self.epoch = view.epoch
        if self.goal is None:
            self.paused = False
            self._recovering_unknown = False
            self._path_clear_since = None
            return Command(0., 0., 'stop', 'waiting_for_goal')
        if self.goal.action in PIVOTS:
            return self._pivot(view, elapsed)

        remaining = _remaining(self.goal, view.pose, self.length, self.curvature[self.goal.action])
        if remaining <= STOP_DISTANCE:
            end = arc_pose(self.goal.origin, self.length, self.curvature[self.goal.action])
            if math.hypot(end.x - view.pose.x, end.y - view.pose.y) > .03:
                return self.stop('path_tracking_lost')
            self.completed = self.goal.sequence
            self.goal, self.successor = self.successor, None
            if self.goal is None:
                self.speed = self.steer = 0.
                return Command(0., 0., 'stop', 'goal_reached')
            if self.goal.action in PIVOTS:
                return self._pivot(view, elapsed)
        return self._track(view, elapsed)

    def _track(self, view, elapsed):
        curvature = self.curvature[self.goal.action]
        remaining = _remaining(self.goal, view.pose, self.length, curvature)
        on_path = arc_pose(self.goal.origin, max(0., self.length - remaining), curvature)
        error = _relative(on_path, view.pose)
        lateral_limit, yaw_limit = (.02, .05) if self.forward_only else (.04, .22)
        if abs(error.y) > lateral_limit or abs(error.yaw) > yaw_limit:
            return self.stop('path_tracking_lost')
        if (self.static_room and remaining < .08 and self.successor is not None
                and self.successor.action not in PIVOTS):
            target = arc_pose(self.successor.origin, .08 - max(0., remaining),
                              self.curvature[self.successor.action])
        else:
            target = arc_pose(self.goal.origin, self.length - max(0., remaining - .08), curvature)
        lookahead = _relative(view.pose, target)
        if lookahead.x <= 0:
            return self.stop('path_tracking_lost')
        steer = (0. if self.forward_only else
                 max(-2., min(2., 2. * lookahead.y / (lookahead.x ** 2 + lookahead.y ** 2))))
        lookahead_distance = min(.10, remaining + .03)
        checked = False
        if self.static_room:
            limit = 5. * min(.05, elapsed)
            gradual = max(self.steer - limit, min(self.steer + limit, steer))
            if gradual != steer and sweep_clear(view.grid, view.pose, lookahead_distance, gradual):
                steer = gradual
                checked = True
        if not checked:
            self.blocked_cell = sweep_block(view.grid, view.pose, lookahead_distance, steer)
        if (self.goal.action == 'forward' and steer and self.blocked_cell is not None
                and self.blocked_cell[0] == 'unknown'
                and sweep_clear(view.grid, view.pose, lookahead_distance, 0.)):
            steer = 0.
            self.blocked_cell = None
        if self.blocked_cell is not None:
            self.blocked_action = self.goal.action
            if self.blocked_cell[0] == 'occupied':
                return self.stop('path_occupied')
            return self._pause('path_unknown')
        if self._path_reacquiring(view):
            return Command(0., 0., 'stop', 'path_reacquiring')
        self.paused = False
        speed = self.max_speed if self.successor is not None else min(self.max_speed, math.sqrt(.08 * remaining))
        self.speed = min(speed, self.speed + .04 * min(.05, elapsed))
        self.steer = steer
        return Command(self.speed, self.speed * steer, self.goal.action, 'tracking')

    def _pivot(self, view, elapsed):
        self.steer = 0.
        relative = _relative(self.goal.origin, view.pose)
        if math.hypot(relative.x, relative.y) > .025:
            return self.stop('path_tracking_lost')
        angle = self.goal.turn
        sign = 1. if angle > 0. else -1.
        remaining = (angle - relative.yaw) * sign
        if remaining <= .012:
            self.completed = self.goal.sequence
            self.goal, self.successor = self.successor, None
            self.speed = 0.
            if self.goal is not None:
                return self._track(view, elapsed)
            return Command(0., 0., 'stop', 'goal_reached')
        self.blocked_cell = rotation_block(view.grid, view.pose, sign * min(.13, remaining + .025))
        if self.blocked_cell is not None:
            self.blocked_action = self.goal.action
            if self.blocked_cell[0] == 'occupied':
                return self.stop('path_occupied')
            return self._pause('path_unknown')
        if self._path_reacquiring(view):
            return Command(0., 0., 'stop', 'path_reacquiring')
        self.paused = False
        return Command(0., sign * min(.05, remaining * 1.5), self.goal.action, 'tracking')
