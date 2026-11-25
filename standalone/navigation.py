"""Odometry-frame local navigation. A selected segment lives until a local event."""
from __future__ import division

from collections import namedtuple
import math


Pose = namedtuple('Pose', 'x y yaw')
Grid = namedtuple('Grid', 'x y resolution width height cells hazard_discs')
Grid.__new__.__defaults__ = ((),)
View = namedtuple('View', 'grid_at scan_at depth_at odom_at status_at epoch pose grid ready depth_evidence')
View.__new__.__defaults__ = (None,)
Request = namedtuple('Request', 'sequence at epoch anchor choices predecessor blocks pivot_turns observation_turns short_segments')
Request.__new__.__defaults__ = ((), (), ())
Command = namedtuple('Command', 'linear angular action reason')
Segment = namedtuple('Segment', 'sequence action origin turn length')
Segment.__new__.__defaults__ = (None,)

LENGTH = .24
ROOM_LENGTH = .12
ROOM_SHORT_SEGMENTS = (.08, .05, .035)
ROOM_TRACK_SWEEP = .10
ROOM_TRACK_RESERVE = .03
ROOM_OBSTACLE_LOOKAHEAD = .60
ROOM_TRACK_LOOKAHEAD = .36
FORWARD_SUCCESSOR_LEAD = .15
CURVATURE = {'forward': 0., 'left': .8, 'right': -.8}
ROOM_CURVATURE = {'forward': 0., 'left': 1.35, 'right': -1.35}
PIVOTS = {'pivot_left': .70, 'pivot_right': -.70}
ROOM_CORNER_TURNS = (.70, 1.40, 2.10, 2.45)
ROOM_SHALLOW_TURNS = (.45, .25)
ROOM_OBSERVATION_TURN = .25
SPEED = .02
ROOM_START_SPEED = .014
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


def segment_endpoint(request, action, length, curvatures):
    if action in PIVOTS:
        turn = dict(request.pivot_turns).get(action, PIVOTS[action])
        anchor = request.anchor._replace(yaw=_wrap(request.anchor.yaw + turn))
        distance = 0. if action in request.observation_turns else length
        return arc_pose(anchor, distance, 0.), distance
    distance = dict(request.short_segments).get(action, length)
    return arc_pose(request.anchor, distance, curvatures[action]), distance


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
    if not points:
        return False, False
    bodies = (base,) if isinstance(base, Pose) else base
    for body in bodies:
        c, s = math.cos(body.yaw), math.sin(body.yaw)
        if all(abs(c * (x - body.x) + s * (y - body.y)) <= HALF_LENGTH + 1e-8
               and abs(-s * (x - body.x) + c * (y - body.y)) <= HALF_WIDTH + 1e-8 for x, y in points):
            return True, False
    return True, True


def _corners(pose, front, rear=HALF_LENGTH, half_width=HALF_WIDTH):
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return [(pose.x + c * x - s * y, pose.y + s * x + c * y)
            for x, y in ((-rear, -half_width), (front, -half_width),
                         (front, half_width), (-rear, half_width))]


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


def _disc_overlaps_polygon(polygon, x, y, radius):
    """Check a closed disc against a counterclockwise convex sweep hull."""
    inside = True
    radius_sq = radius * radius + 1e-12
    last_x, last_y = polygon[-1]
    for next_x, next_y in polygon:
        dx, dy = next_x - last_x, next_y - last_y
        ox, oy = x - last_x, y - last_y
        inside = inside and dx * oy - dy * ox >= -1e-12
        length_sq = dx * dx + dy * dy
        if length_sq:
            fraction = min(1., max(0., (ox * dx + oy * dy) / length_sq))
            ox, oy = ox - fraction * dx, oy - fraction * dy
        if ox * ox + oy * oy <= radius_sq:
            return True
        last_x, last_y = next_x, next_y
    return inside


def _polygon_block(grid, polygon, known_body, occupied_only=False):
    x_min, x_max = min(p[0] for p in polygon), max(p[0] for p in polygon)
    y_min, y_max = min(p[1] for p in polygon), max(p[1] for p in polygon)
    for x, y, radius in grid.hazard_discs:
        bound = radius + 1e-9
        if (x_min - bound <= x <= x_max + bound and y_min - bound <= y <= y_max + bound
                and _disc_overlaps_polygon(polygon, x, y, radius)):
            return 'occupied', x, y

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


def room_continuations(grid, endpoint):
    """Yield continuations supported at a proposed endpoint by the current grid."""
    for action in ('forward', 'left', 'right'):
        if sweep_clear(grid, endpoint, ROOM_LENGTH, ROOM_CURVATURE[action]):
            yield action


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
    return (goal.length or length) - distance


class Navigation(object):
    """Single-writer task state. Feed immutable observations and finished choices."""

    def __init__(self, forward_only=False, speed=SPEED, successor_lead=None, static_room=False,
                 short_room_segments=False):
        self.last_observation_turn = None
        self._pivot_body = None
        if not 0. < speed <= SPEED:
            raise ValueError('invalid navigation speed')
        self.forward_only, self.max_speed = forward_only, speed
        self.static_room = static_room
        self.short_room_segments = static_room and short_room_segments
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
        self._failed_pivots = {}
        self._recovering_unknown = False
        self._path_clear_since = None

    def stop(self, reason):
        self.goal = self.successor = self.pending = self.completed = None
        self.paused = False
        self._recovering_unknown = False
        self._path_clear_since = None
        self.speed = self.steer = 0.
        return Command(0., 0., 'stop', reason)

    def retire_stalled_pivot(self, pose):
        goal = self.goal
        self._failed_pivots[goal.action] = pose
        if self.last_observation_turn == goal.origin:
            self.last_observation_turn = None
        self._blocked_corner = None
        self.stop('yaw_no_progress')

    def _pivot_failed_near(self, action, anchor):
        previous = self._failed_pivots.get(action)
        return previous is not None and math.hypot(anchor.x - previous.x, anchor.y - previous.y) <= ROOM_LENGTH

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
                anchor = arc_pose(self.goal.origin, self.goal.length or self.length, self.curvature[self.goal.action])
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
        pivot_turns, observation_turns, short_segments = (), (), ()
        if self.static_room and self.goal is None:
            pivot_turns, observation_turns = self._room_turns(view, anchor, choices)
            if pivot_turns:
                choices = [action for action, _ in pivot_turns]
        if not choices and self.short_room_segments and predecessor is None:
            short_segments = tuple((action, distance) for action in actions for distance in
                                   [self._short_segment(view, anchor, action)] if distance is not None)
            choices = [action for action, _ in short_segments]
        if not choices:
            return None
        self.sequence += 1
        blocks = tuple((action,) + self.candidate_blocks[action] for action in actions
                       if action in self.candidate_blocks)
        self.pending = Request(self.sequence, now, view.epoch, anchor, tuple(choices) + ('stop',), predecessor,
                               blocks, pivot_turns, observation_turns, short_segments)
        return self.pending

    def _short_segment(self, view, anchor, action):
        for distance in ROOM_SHORT_SEGMENTS:
            if self._segment_clear(view, anchor, action, distance):
                return distance
        return None

    def _segment_clear(self, view, anchor, action, shortened):
        distance = self.length if shortened is None else min(ROOM_TRACK_SWEEP, shortened + ROOM_TRACK_RESERVE)
        return sweep_clear(view.grid, anchor, distance, self.curvature[action], view.pose)

    def segment_request_clear(self, request, action, view):
        return self._segment_clear(view, request.anchor, action, dict(request.short_segments).get(action))

    def _room_turns(self, view, anchor, choices):
        obstacle = sweep_block(view.grid, anchor, .42, 0., occupied_only=True)
        if obstacle is None or any(sweep_block(view.grid, anchor, .31, self.curvature[action],
                                               view.pose, occupied_only=True) is None for action in choices):
            return (), ()
        pivots = []
        for action in ('pivot_left', 'pivot_right'):
            for magnitude in (.70,) + ROOM_SHALLOW_TURNS:
                angle = math.copysign(magnitude, PIVOTS[action])
                if self._pivot_clear(view, anchor, action, angle):
                    pivots.append((action, angle))
                    break
        if pivots:
            return tuple(pivots), ()
        last = self.last_observation_turn
        if (-1 not in view.grid.cells
                or last is not None and math.hypot(anchor.x - last.x, anchor.y - last.y) < .12):
            return (), ()
        side = _relative(anchor, Pose(obstacle[1], obstacle[2], anchor.yaw)).y
        directions = ('pivot_left',) if side < -.025 else ('pivot_right',) if side > .025 else (
            'pivot_left', 'pivot_right')
        for action in directions:
            angle = math.copysign(ROOM_OBSERVATION_TURN, PIVOTS[action])
            if (not self._pivot_failed_near(action, anchor)
                    and rotation_block(view.grid, anchor, angle, view.pose) is None):
                pivots.append((action, angle))
        return tuple(pivots), tuple(action for action, _ in pivots)

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
        turns = ROOM_CORNER_TURNS if corner is not None else (.70,)
        if self.goal is None:
            turns += ROOM_SHALLOW_TURNS
        for angle in turns:
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

    def replace_unknown_path(self, view, now, direction, motion):
        """Replace a paused path only when its actual starting curvature is observed."""
        if (self.fault or self.forward_only or not self.paused or self.goal is None
                or self.blocked_cell is None or self.blocked_cell[0] != 'unknown' or not ready(view, now)
                or view.epoch != self.epoch):
            return None
        self.candidate_blocks = {}
        speed, previous_steer, elapsed = motion
        slew = 5. * min(.05, elapsed)
        away = 'right' if direction == 'left' else 'left'
        for action in ('forward', direction, away):
            if action == self.goal.action:
                continue
            curvature = self.curvature[action]
            block = sweep_block(view.grid, view.pose, self.length, curvature)
            if block is None:
                steer = max(previous_steer - slew, min(previous_steer + slew, curvature))
                block = sweep_block(view.grid, view.pose, .10, steer)
                if block is not None and speed == 0.:
                    steer, block = curvature, None  # The full nominal arc already passed from this pose.
                if block is None:
                    self._start_detour(view.pose, action)
                    self.speed = self._next_speed(self.max_speed, speed, elapsed)
                    self.steer = steer
                    return Command(self.speed, self.speed * steer, action, 'unknown_detour')
            self.candidate_blocks[action] = block
        if (not speed and self.static_room
                and sweep_block(view.grid, view.pose, .42, 0., occupied_only=True)):
            for action in ('pivot_' + direction, 'pivot_' + away):
                if action != self.goal.action and self._pivot_clear(view, view.pose, action):
                    self._start_detour(view.pose, action)
                    self.goal = self.goal._replace(turn=PIVOTS[action])
                    return self._pivot(view, elapsed)
        return None

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
        action = self._checked_detour(view, (direction, away, 'forward'))
        if action is not None:
            self._start_detour(view.pose, action)
            # Reuse the tracker's steering and swept-body gate for the published command.
            return self._track(view, 0.), block
        return None

    def _checked_detour(self, view, actions):
        for action in actions:
            if action == self.goal.action:
                continue
            candidate = self.curvature[action]
            if (sweep_clear(view.grid, view.pose, self.length, candidate)
                    and not sweep_block(view.grid, view.pose, ROOM_TRACK_LOOKAHEAD, candidate, occupied_only=True)):
                return action
        return None

    def _start_detour(self, pose, action):
        self.sequence += 1
        self.goal = Segment(self.sequence, action, pose, None)
        self.successor = self.pending = self.completed = None
        self.paused = self._recovering_unknown = False
        self._path_clear_since = None

    def _pivot_clear(self, view, anchor, action, turn=None):
        if self._pivot_failed_near(action, anchor):
            return False
        angle = PIVOTS[action] if turn is None else turn
        turned = anchor._replace(yaw=_wrap(anchor.yaw + angle))
        return (sweep_clear(view.grid, turned, self.length, 0., view.pose)
                and sweep_block(view.grid, turned, .31, 0., occupied_only=True) is None
                and rotation_block(view.grid, anchor, angle, view.pose) is None)

    def pivot_request_clear(self, request, action, view):
        angle = dict(request.pivot_turns).get(action, PIVOTS[action])
        if action in request.observation_turns:
            return (not self._pivot_failed_near(action, request.anchor)
                    and rotation_block(view.grid, request.anchor, angle, view.pose) is None)
        return self._pivot_clear(view, request.anchor, action, angle)

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
        turn = dict(request.pivot_turns).get(choice, PIVOTS.get(choice))
        if choice in PIVOTS:
            offset = _relative(request.anchor, view.pose)
            if (math.hypot(offset.x, offset.y) > .015 or abs(offset.yaw) > .05
                    or not self.pivot_request_clear(request, choice, view)):
                return False
        elif not self.segment_request_clear(request, choice, view):
            return False
        segment = Segment(request.sequence, choice, request.anchor, turn, dict(request.short_segments).get(choice))
        if active_predecessor:
            self.successor = segment
        elif self.goal is None and request.predecessor in (None, self.completed):
            self.goal = segment
            if choice in request.observation_turns:
                self.last_observation_turn = request.anchor
        else:
            return False
        return True

    def step(self, view, now, room_direction=None):
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
            end = arc_pose(self.goal.origin, self.goal.length or self.length, self.curvature[self.goal.action])
            if math.hypot(end.x - view.pose.x, end.y - view.pose.y) > .03:
                return self.stop('path_tracking_lost')
            self.completed = self.goal.sequence
            self.goal, self.successor = self.successor, None
            if self.goal is None:
                self.speed = self.steer = 0.
                return Command(0., 0., 'stop', 'goal_reached')
            if self.goal.action in PIVOTS:
                return self._pivot(view, elapsed)
        return self._track(view, elapsed, room_direction)

    def _track(self, view, elapsed, room_direction=None):
        curvature = self.curvature[self.goal.action]
        length = self.goal.length or self.length
        remaining = _remaining(self.goal, view.pose, self.length, curvature)
        on_path = arc_pose(self.goal.origin, max(0., length - remaining), curvature)
        error = _relative(on_path, view.pose)
        lateral_limit, yaw_limit = (.02, .05) if self.forward_only else (.04, .22)
        if abs(error.y) > lateral_limit or abs(error.yaw) > yaw_limit:
            return self.stop('path_tracking_lost')
        if (self.static_room and remaining < .08 and self.successor is not None
                and self.successor.action not in PIVOTS):
            target_distance = min(.08 - max(0., remaining), self.successor.length or self.length)
            target = arc_pose(self.successor.origin, target_distance,
                              self.curvature[self.successor.action])
        else:
            target = arc_pose(self.goal.origin, length - max(0., remaining - .08), curvature)
        lookahead = _relative(view.pose, target)
        if lookahead.x <= 0:
            return self.stop('path_tracking_lost')
        steer = (0. if self.forward_only else
                 max(-2., min(2., 2. * lookahead.y / (lookahead.x ** 2 + lookahead.y ** 2))))
        lookahead_distance = min(ROOM_TRACK_SWEEP, remaining + ROOM_TRACK_RESERVE)
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
                if self.static_room and room_direction in ('left', 'right') and self.speed > 0.:
                    away = 'right' if room_direction == 'left' else 'left'
                    alternate = self._checked_detour(view, ('forward', room_direction, away))
                    if alternate is not None:
                        # Command the same nominal arc that passed the fresh sweep.
                        self._start_detour(view.pose, alternate)
                        self.steer = self.curvature[alternate]
                        return Command(self.speed, self.speed * self.steer, alternate, 'occupied_detour')
                return self.stop('path_occupied')
            return self._pause('path_unknown')
        if self._path_reacquiring(view):
            return Command(0., 0., 'stop', 'path_reacquiring')
        self.paused = False
        speed = self.max_speed if self.successor is not None else min(self.max_speed, math.sqrt(.08 * remaining))
        self.speed = self._next_speed(speed, self.speed, elapsed)
        self.steer = steer
        return Command(self.speed, self.speed * steer, self.goal.action, 'tracking')

    def _next_speed(self, target, previous, elapsed):
        speed = previous + .04 * min(.05, elapsed)
        if self.static_room and previous == 0. and elapsed > 0.:
            # Lower startup setpoints rarely produce wheel motion on the LIMO.
            speed = max(speed, ROOM_START_SPEED)
        return min(target, speed)

    def _pivot(self, view, elapsed):
        self.steer = 0.
        relative = _relative(self.goal.origin, view.pose)
        if math.hypot(relative.x, relative.y) > .025:
            return self.stop('path_tracking_lost')
        if self._pivot_body is None or self._pivot_body[0] is not self.goal:
            self._pivot_body = self.goal, view.pose
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
        self.blocked_cell = rotation_block(view.grid, view.pose, sign * min(.13, remaining + .025),
                                           (view.pose, self._pivot_body[1]))
        if self.blocked_cell is not None:
            self.blocked_action = self.goal.action
            if self.blocked_cell[0] == 'occupied':
                return self.stop('path_occupied')
            return self._pause('path_unknown')
        if self._path_reacquiring(view):
            return Command(0., 0., 'stop', 'path_reacquiring')
        self.paused = False
        return Command(0., sign * min(.05, remaining * 1.5), self.goal.action, 'tracking')
