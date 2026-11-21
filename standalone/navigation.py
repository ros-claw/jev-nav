"""Odometry-frame local navigation. A selected segment lives until a local event."""
from __future__ import division

from collections import namedtuple
import math


Pose = namedtuple('Pose', 'x y yaw')
Grid = namedtuple('Grid', 'x y resolution width height cells')
View = namedtuple('View', 'grid_at scan_at depth_at odom_at status_at epoch pose grid ready')
Request = namedtuple('Request', 'sequence at epoch anchor choices predecessor')
Command = namedtuple('Command', 'linear angular action reason')
Segment = namedtuple('Segment', 'sequence action origin')

LENGTH = .24
CURVATURE = {'forward': 0., 'left': .8, 'right': -.8}
SPEED = .02
MAX_RESPONSE_AGE = 8.
HALF_LENGTH, HALF_WIDTH = .16, .11
STOP_DISTANCE = .01


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


def _polygon_block(grid, polygon, known_body):
    x_min, x_max = min(p[0] for p in polygon), max(p[0] for p in polygon)
    y_min, y_max = min(p[1] for p in polygon), max(p[1] for p in polygon)

    def to_index(value, offset):
        return int(math.floor((value - offset) / grid.resolution))

    for row in range(to_index(y_min, grid.y), to_index(y_max, grid.y) + 1):
        for column in range(to_index(x_min, grid.x), to_index(x_max, grid.x) + 1):
            within = 0 <= row < grid.height and 0 <= column < grid.width
            cell = grid.cells[row * grid.width + column] if within else -1
            if cell == 0:
                continue
            overlaps, new_space = _overlap(polygon, known_body, grid, column, row)
            if overlaps and (cell != -1 or new_space):
                return ('unknown' if cell == -1 else 'occupied',
                        grid.x + (column + .5) * grid.resolution, grid.y + (row + .5) * grid.resolution)
    return None


def sweep_block(grid, pose, distance, curvature, known_body=None):
    """Keep unknown outside the already occupied body out of a swept corridor."""
    if known_body is None:
        known_body = pose
    if abs(curvature) < 1e-8:
        return _polygon_block(grid, _corners(pose, HALF_LENGTH + distance), known_body)
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
        blocked = _polygon_block(grid, _hull(start + end + outer), known_body)
        if blocked is not None:
            return blocked
        start = end
    return None


def sweep_clear(grid, pose, distance, curvature, known_body=None):
    return sweep_block(grid, pose, distance, curvature, known_body) is None


def ready(view, now):
    if view is None or view.ready is not True:
        return False
    return all(0 <= now - stamp < limit for stamp, limit in (
        (view.grid_at, .25), (view.scan_at, .30), (view.depth_at, .20),
        (view.odom_at, .15), (view.status_at, 1.)))


def _remaining(goal, pose):
    local = _relative(goal.origin, pose)
    curvature = CURVATURE[goal.action]
    distance = (math.atan2(curvature * local.x, 1. - curvature * local.y) / curvature
                if curvature else local.x)
    return LENGTH - distance


class Navigation(object):
    """Single-writer task state. Feed immutable observations and finished choices."""

    def __init__(self, forward_only=False, speed=SPEED):
        if not 0. < speed <= SPEED:
            raise ValueError('invalid navigation speed')
        self.forward_only, self.max_speed = forward_only, speed
        self.goal = self.successor = self.pending = None
        self.sequence = 0
        self.epoch = self.last_at = self.completed = None
        self.fault = None
        self.speed = 0.
        self.paused = False
        self.blocked_cell = None
        self.candidate_blocks = {}

    def _stop(self, reason):
        self.goal = self.successor = self.pending = self.completed = None
        self.paused = False
        self.speed = 0.
        return Command(0., 0., 'stop', reason)

    def _pause(self, reason):
        self.speed = 0.
        self.paused = True
        return Command(0., 0., 'stop', reason)

    def request(self, view, now):
        self.candidate_blocks = {}
        if self.fault or self.paused or self.successor is not None or not ready(view, now):
            return None
        if self.goal is not None:
            remaining = _remaining(self.goal, view.pose)
            if remaining > .09:
                return None
            if self.forward_only:
                # A straight continuation follows the current measured line.
                # The predecessor's original heading may differ slightly.
                anchor = arc_pose(view.pose, max(0., remaining), 0.)
            else:
                anchor = arc_pose(self.goal.origin, LENGTH, CURVATURE[self.goal.action])
            predecessor = self.goal.sequence
        else:
            anchor, predecessor = view.pose, None
        choices = []
        actions = ('forward',) if self.forward_only else ('forward', 'left', 'right')
        for action in actions:
            block = sweep_block(view.grid, anchor, LENGTH, CURVATURE[action], view.pose)
            if block is None:
                choices.append(action)
            else:
                self.candidate_blocks[action] = block
        if not choices:
            return None
        self.sequence += 1
        self.pending = Request(self.sequence, now, view.epoch, anchor, tuple(choices) + ('stop',), predecessor)
        return self.pending

    def accept(self, request, choice, view, now):
        if (request is None or request is not self.pending or self.fault
                or not 0 <= now - request.at < MAX_RESPONSE_AGE
                or choice not in request.choices):
            return False
        self.pending = None
        if choice == 'stop':
            self._stop('model_stop')
            return True
        if not ready(view, now) or request.epoch != view.epoch:
            return False
        if (math.hypot(view.pose.x - request.anchor.x, view.pose.y - request.anchor.y) > .10
                or abs(_wrap(view.pose.yaw - request.anchor.yaw)) > .20):
            return False
        if not sweep_clear(view.grid, request.anchor, LENGTH, CURVATURE[choice], view.pose):
            return False
        segment = Segment(request.sequence, choice, request.anchor)
        if self.goal is not None and request.predecessor == self.goal.sequence:
            self.successor = segment
        elif self.goal is None and request.predecessor in (None, self.completed):
            self.goal = segment
        else:
            return False
        return True

    def step(self, view, now):
        self.blocked_cell = None
        elapsed = .05 if self.last_at is None else now - self.last_at
        if elapsed < 0 or (view is not None and self.epoch is not None and view.epoch != self.epoch):
            self.fault = 'clock_or_odometry_reset'
        self.last_at = now
        if self.fault:
            return self._stop(self.fault)
        if not ready(view, now):
            return self._pause('perception_unavailable')
        self.epoch = view.epoch
        if self.goal is None:
            self.paused = False
            return Command(0., 0., 'stop', 'waiting_for_goal')

        remaining = _remaining(self.goal, view.pose)
        if remaining <= STOP_DISTANCE:
            end = arc_pose(self.goal.origin, LENGTH, CURVATURE[self.goal.action])
            if math.hypot(end.x - view.pose.x, end.y - view.pose.y) > .03:
                return self._stop('path_tracking_lost')
            self.completed = self.goal.sequence
            self.goal, self.successor = self.successor, None
            if self.goal is None:
                self.speed = 0.
                return Command(0., 0., 'stop', 'goal_reached')
            remaining = _remaining(self.goal, view.pose)

        curvature = CURVATURE[self.goal.action]
        on_path = arc_pose(self.goal.origin, max(0., LENGTH - remaining), curvature)
        error = _relative(on_path, view.pose)
        lateral_limit, yaw_limit = (.02, .05) if self.forward_only else (.04, .22)
        if abs(error.y) > lateral_limit or abs(error.yaw) > yaw_limit:
            return self._stop('path_tracking_lost')
        target = arc_pose(self.goal.origin, LENGTH - max(0., remaining - .08), curvature)
        lookahead = _relative(view.pose, target)
        if lookahead.x <= 0:
            return self._stop('path_tracking_lost')
        steer = (0. if self.forward_only else
                 max(-2., min(2., 2. * lookahead.y / (lookahead.x ** 2 + lookahead.y ** 2))))
        self.blocked_cell = sweep_block(view.grid, view.pose, min(.10, remaining + .03), steer)
        if self.blocked_cell is not None:
            if self.blocked_cell[0] == 'occupied':
                return self._stop('path_occupied')
            return self._pause('path_unknown')
        self.paused = False
        speed = self.max_speed if self.successor is not None else min(self.max_speed, math.sqrt(.08 * remaining))
        self.speed = min(speed, self.speed + .04 * min(.05, elapsed))
        return Command(self.speed, self.speed * steer, self.goal.action, 'tracking')
