"""Room memory tied to mapper keyframes and reprojected into continuous odom."""
from __future__ import division

import math
from collections import namedtuple

from navigation import PIVOTS, Pose, _relative, _wrap, arc_pose


MapFix = namedtuple('MapFix', 'map_id revision odom_to_map anchors')
Hazard = namedtuple('Hazard', 'anchor x y radius low_z high_z')


def compose(origin, offset):
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    return Pose(origin.x + c * offset.x - s * offset.y, origin.y + s * offset.x + c * offset.y,
                _wrap(origin.yaw + offset.yaw))


def _finite(value):
    return not (math.isnan(value) or math.isinf(value))


class SpatialMemory(object):
    def __init__(self, map_id, state=None):
        self.map_id = map_id
        self.hazards, self.visits = {}, {}
        self.fixed = None
        self.last_visit = None
        self.changed = False
        if state is not None:
            if state.get('version') != 1 or state.get('map_id') != map_id:
                raise ValueError('spatial memory belongs to a different map or format')
            for row in state['hazards']:
                record = Hazard(*row)
                if (not all(_finite(v) for v in row) or record.anchor <= 0 or not 0. < record.radius <= .20
                        or record.low_z > record.high_z):
                    raise ValueError('invalid stored map hazard')
                self.hazards[self._hazard_key(record)] = record
            for anchor, col, row, count in state['visits']:
                if not all(type(v) is int for v in (anchor, col, row, count)) or min(anchor, count) <= 0:
                    raise ValueError('invalid stored map visit')
                self.visits[anchor, col, row] = count
        self.reset_registration()

    def reset_registration(self):
        self.old_anchors = set(k[0] for k in self.hazards) | set(k[0] for k in self.visits)
        self.registered = not self.old_anchors
        self.fixed = None
        self.last_visit = None

    def update_fix(self, fix, closure=None):
        if (fix is None or fix.map_id != self.map_id or not fix.anchors
                or not self.old_anchors.issubset(fix.anchors)):
            self.fixed = None
            return False
        self.registered = self.registered or closure in self.old_anchors
        self.fixed = fix if self.registered else None
        return self.fixed is not None

    def snapshot(self):
        return {'version': 1, 'map_id': self.map_id, 'hazards': list(self.hazards.values()),
                'visits': [key + (value,) for key, value in self.visits.items()]}

    def _anchor(self, pose):
        anchors = self.fixed.anchors
        return min(anchors, key=lambda n: (pose.x - anchors[n].x) ** 2 + (pose.y - anchors[n].y) ** 2)

    @staticmethod
    def _hazard_key(record):
        return record.anchor, int(round(record.x / .05)), int(round(record.y / .05))

    def record_depth(self, measurements):
        if self.fixed is None:
            return
        for x, y, half_cell, low, high in measurements:
            point = compose(self.fixed.odom_to_map, Pose(x, y, 0.))
            anchor = self._anchor(point)
            offset = _relative(self.fixed.anchors[anchor], point)
            # The cell circumcircle and a small additional uncertainty margin
            # enclose the recorded position through a map correction.
            record = Hazard(anchor, offset.x, offset.y, half_cell * math.sqrt(2.) + .03, low, high)
            key = self._hazard_key(record)
            if self.hazards.get(key) != record:
                self.hazards[key] = record
                self.changed = True

    def clear_depth(self, measurements):
        if self.fixed is None:
            return
        for x, y, low, high in measurements:
            point = compose(self.fixed.odom_to_map, Pose(x, y, 0.))
            for key, hazard in list(self.hazards.items()):
                if not low <= hazard.low_z <= hazard.high_z <= high:
                    continue
                original = compose(self.fixed.anchors[hazard.anchor], Pose(hazard.x, hazard.y, 0.))
                if math.hypot(original.x - point.x, original.y - point.y) <= .005:
                    del self.hazards[key]
                    self.changed = True

    def _visit_key(self, point):
        anchor = self._anchor(point)
        offset = _relative(self.fixed.anchors[anchor], point)
        return anchor, int(math.floor(offset.x / .20)), int(math.floor(offset.y / .20))

    def visit(self, pose):
        if self.fixed is None:
            return False
        key = self._visit_key(compose(self.fixed.odom_to_map, pose))
        if key == self.last_visit:
            return False
        self.last_visit = key
        self.visits[key] = self.visits.get(key, 0) + 1
        self.changed = True
        return True

    def apply(self, view):
        if view is None or not view.ready:
            return view
        if self.fixed is None:
            return view._replace(ready=False)
        grid = view.grid
        cells = None
        for record in self.hazards.values():
            anchor = self.fixed.anchors.get(record.anchor)
            if anchor is None:
                return view._replace(ready=False)
            point = _relative(self.fixed.odom_to_map, compose(anchor, Pose(record.x, record.y, 0.)))
            col0 = max(0, int(math.floor((point.x - record.radius - grid.x) / grid.resolution)))
            row0 = max(0, int(math.floor((point.y - record.radius - grid.y) / grid.resolution)))
            col1 = min(grid.width - 1, int(math.floor((point.x + record.radius - grid.x) / grid.resolution)))
            row1 = min(grid.height - 1, int(math.floor((point.y + record.radius - grid.y) / grid.resolution)))
            if col0 > col1 or row0 > row1:
                continue
            if cells is None:
                cells = list(grid.cells)
            for row in range(row0, row1 + 1):
                first, last = row * grid.width + col0, row * grid.width + col1 + 1
                cells[first:last] = [100] * (last - first)
        return view if cells is None else view._replace(grid=grid._replace(cells=tuple(cells)))

    @staticmethod
    def _frontier_cells(grid, endpoint):
        """Count unknown cells just beyond the checked endpoint as a relative gain signal."""
        if grid is None:
            return 0
        c, s = math.cos(endpoint.yaw), math.sin(endpoint.yaw)
        extent = .45
        col0 = max(0, int(math.floor((endpoint.x - extent - grid.x) / grid.resolution)))
        row0 = max(0, int(math.floor((endpoint.y - extent - grid.y) / grid.resolution)))
        col1 = min(grid.width, int(math.ceil((endpoint.x + extent - grid.x) / grid.resolution)))
        row1 = min(grid.height, int(math.ceil((endpoint.y + extent - grid.y) / grid.resolution)))
        count = 0
        for row in range(row0, row1):
            y = grid.y + (row + .5) * grid.resolution - endpoint.y
            for col in range(col0, col1):
                if grid.cells[row * grid.width + col] >= 0:
                    continue
                x = grid.x + (col + .5) * grid.resolution - endpoint.x
                along, across = c * x + s * y, c * y - s * x
                if .10 < along <= .40 and abs(across) <= .55 * along:
                    count += 1
        return count

    def targets(self, request, length, curvatures, grid=None):
        if self.fixed is None:
            return []
        result = []
        for action in request.choices:
            if action in PIVOTS:
                endpoint = request.anchor._replace(yaw=_wrap(request.anchor.yaw + PIVOTS[action]))
                endpoint = arc_pose(endpoint, length, 0.)
            elif action in curvatures:
                endpoint = arc_pose(request.anchor, length, curvatures[action])
            else:
                continue
            point = compose(self.fixed.odom_to_map, endpoint)
            key = self._visit_key(point)
            result.append({'id': '%d-%s' % (request.sequence, action), 'action': action,
                           'map_id': self.map_id, 'revision': self.fixed.revision,
                           'anchor': key[0], 'x': round(point.x, 3), 'y': round(point.y, 3),
                           'visits': self.visits.get(key, 0), 'distance_m': length,
                           'frontier_cells': self._frontier_cells(grid, endpoint)})
        return result
