"""Short-range odometry-frame memory for a supervised static room."""
from __future__ import division

import math
from collections import deque, namedtuple

from navigation import HALF_LENGTH, HALF_WIDTH, ready


DepthEvidence = namedtuple('DepthEvidence', 'sequence frame hits clear_volume')
DepthHazard = namedtuple('DepthHazard', 'frame low high first seen confirmed clear_since last_clear')


class RoomPerception(object):
    MAX_AGE_SEC = 30.
    ISOLATED_OCCUPANCY_AGE_SEC = 1.5
    OCCUPANCY_CONFIRM_SEC = .30
    OCCUPANCY_SAMPLE_GAP_SEC = .15
    DEPTH_SAMPLE_GAP_SEC = .20
    REAR_LIMIT = HALF_LENGTH + .12
    SIDE_LIMIT = HALF_WIDTH + .12
    LIDAR_X = .105

    def __init__(self):
        self.epoch = self.lattice = self.last_grid = None
        self.cells = {}
        self.depth_hazards = {}
        self.depth_changed = set()
        self.depth_clearing = set()
        self.cleared_depth = deque(maxlen=512)
        self.depth_sequence = None
        self.pruned_at = -1.
        self.oldest_age = None
        self.reused_cells = 0

    def confirmed_depth(self, only_changed=False):
        if self.lattice is not None:
            x, y, resolution = self.lattice
            keys = self.depth_changed if only_changed else self.depth_hazards
            for col, row in keys:
                record = self.depth_hazards[col, row]
                if record.confirmed:
                    yield (x + (col + .5) * resolution, y + (row + .5) * resolution,
                           .5 * resolution, record.low, record.high)
            if only_changed:
                self.depth_changed.clear()

    def _observe_depth(self, observation, at, grid, ox, oy):
        if self.depth_sequence is not None and observation.sequence <= self.depth_sequence:
            return
        self.depth_sequence = observation.sequence
        for index, (low, high) in observation.hits.items():
            row, column = divmod(index, grid.width)
            key = (ox + column, oy + row)
            old = self.depth_hazards.get(key)
            first, confirmed = at, False
            if old is not None and old.frame == observation.frame:
                if old.confirmed or 0 <= at - old.seen <= self.DEPTH_SAMPLE_GAP_SEC:
                    low, high, first = min(low, old.low), max(high, old.high), old.first
                    confirmed = old.confirmed or at - first >= self.OCCUPANCY_CONFIRM_SEC
            self.depth_hazards[key] = DepthHazard(observation.frame, low, high, first, at, confirmed, None, None)
            if confirmed and (not old.confirmed or (low, high) != (old.low, old.high)):
                self.depth_changed.add(key)
        clearing = set()
        for index, (low, high) in observation.clear_volume.items():
            row, column = divmod(index, grid.width)
            key = (ox + column, oy + row)
            old = self.depth_hazards.get(key)
            if (old is None or old.frame != observation.frame or grid.cells[index] > 0
                    or index in observation.hits or not low <= old.low <= old.high <= high):
                continue
            first = old.clear_since if old.last_clear is not None and 0 <= at - old.last_clear <= self.DEPTH_SAMPLE_GAP_SEC else at
            if at - first >= self.OCCUPANCY_CONFIRM_SEC:
                del self.depth_hazards[key]
                self.depth_changed.discard(key)
                x, y, resolution = self.lattice
                self.cleared_depth.append((x + (key[0] + .5) * resolution, y + (key[1] + .5) * resolution,
                                           low, high))
            else:
                self.depth_hazards[key] = old._replace(clear_since=first, last_clear=at)
                clearing.add(key)
        for key in self.depth_clearing - clearing:
            if key in self.depth_hazards:
                self.depth_hazards[key] = self.depth_hazards[key]._replace(clear_since=None, last_clear=None)
        self.depth_clearing = clearing

    def view(self, view, now):
        self.oldest_age = None
        self.reused_cells = 0
        if not ready(view, now):
            return view
        grid, pose = view.grid, view.pose
        if self.epoch != view.epoch or self.last_grid is not None and view.grid_at < self.last_grid:
            self.__init__()
            self.epoch = view.epoch
        if self.lattice is None:
            self.lattice = (grid.x, grid.y, grid.resolution)
        origin_x, origin_y, resolution = self.lattice
        dx, dy = (grid.x - origin_x) / resolution, (grid.y - origin_y) / resolution
        ox, oy = int(round(dx)), int(round(dy))
        if grid.resolution != resolution or abs(ox - dx) > .001 or abs(oy - dy) > .001:
            self.__init__()
            return view
        if now - self.pruned_at >= 1.:
            self.cells = {key: entry for key, entry in self.cells.items() if now - entry[1] <= self.MAX_AGE_SEC}
            self.depth_hazards = {key: record for key, record in self.depth_hazards.items()
                                  if record.confirmed or now - record.seen <= self.ISOLATED_OCCUPANCY_AGE_SEC}
            self.pruned_at = now

        # Visit only cells near the chassis. The observation timestamp is the
        # source acquisition time for free space, which comes from lidar rays.
        center_x, center_y = (pose.x - grid.x) / resolution, (pose.y - grid.y) / resolution
        extent = .40 / resolution
        columns = range(max(0, int(math.floor(center_x - extent))),
                        min(grid.width, int(math.ceil(center_x + extent))))
        rows = range(max(0, int(math.floor(center_y - extent))),
                     min(grid.height, int(math.ceil(center_y + extent))))
        observe = self.last_grid != view.grid_at
        self.last_grid = view.grid_at
        evidence = view.depth_evidence
        if evidence is not None:
            self._observe_depth(evidence, view.depth_at, grid, ox, oy)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        # Include a partially occluded cell when its rear edge is behind
        # the laser plane. Ahead of the chassis the grid must stay live.
        half_cell = .5 * resolution * (abs(c) + abs(s))
        blind_limit = min(HALF_LENGTH, self.LIDAR_X + half_cell)
        revised = None
        for row in rows:
            delta_y = (row + .5 - center_y) * resolution
            for column in columns:
                index = row * grid.width + column
                key = (ox + column, oy + row)
                value = grid.cells[index]
                remembered = self.cells.get(key)
                from_depth = evidence is not None and index in evidence.hits
                if from_depth:
                    self.cells.pop(key, None)
                    remembered = None
                if observe and value >= 0 and not from_depth:
                    acquired = view.scan_at if value == 0 else view.grid_at
                    unqualified_clear = (value == 0 and evidence is None and remembered is not None
                                         and remembered[0] > 0 and remembered[3]
                                         and now - remembered[1] <= self.MAX_AGE_SEC)
                    if not unqualified_clear and (remembered is None or acquired >= remembered[1]):
                        occupied_since, confirmed = acquired, False
                        if value > 0 and remembered is not None and remembered[0] > 0:
                            _, last_seen, previous_since, confirmed = remembered
                            if acquired - last_seen <= self.OCCUPANCY_SAMPLE_GAP_SEC:
                                occupied_since = previous_since
                            confirmed = (confirmed and acquired - last_seen <= self.MAX_AGE_SEC
                                         or acquired - occupied_since >= self.OCCUPANCY_CONFIRM_SEC)
                        remembered = self.cells[key] = (value, acquired, occupied_since, confirmed)
                hazard = self.depth_hazards.get(key)
                if hazard is not None and value <= 0:
                    depth_age = now - hazard.seen
                    if hazard.confirmed or 0 <= depth_age <= self.ISOLATED_OCCUPANCY_AGE_SEC:
                        if revised is None:
                            revised = list(grid.cells)
                        revised[index] = 100
                        self.reused_cells += 1
                        self.oldest_age = depth_age if self.oldest_age is None else max(depth_age, self.oldest_age)
                        continue
                if remembered is None or value > 0:
                    continue
                old_value, acquired, _, confirmed = remembered
                age = now - acquired
                retention = self.ISOLATED_OCCUPANCY_AGE_SEC if old_value > 0 and not confirmed else self.MAX_AGE_SEC
                # Lost evidence returns to the live grid's state. It never restores older free space.
                if not 0 <= age <= retention:
                    continue
                delta_x = (column + .5 - center_x) * resolution
                x, y = c * delta_x + s * delta_y, -s * delta_x + c * delta_y
                if old_value == 0 and (value == 0 or not -self.REAR_LIMIT <= x < blind_limit
                                       or abs(y) > self.SIDE_LIMIT):
                    continue
                if revised is None:
                    revised = list(grid.cells)
                revised[index] = old_value
                self.reused_cells += 1
                self.oldest_age = age if self.oldest_age is None else max(age, self.oldest_age)
        return view if revised is None else view._replace(grid=grid._replace(cells=tuple(revised)))
