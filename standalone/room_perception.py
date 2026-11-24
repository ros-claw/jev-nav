"""Short-range odometry-frame memory for a supervised static room."""
from __future__ import division

import math

from navigation import HALF_LENGTH, HALF_WIDTH, ready


class RoomPerception(object):
    MAX_AGE_SEC = 30.
    ISOLATED_OCCUPANCY_AGE_SEC = 1.5
    OCCUPANCY_CONFIRM_SEC = .30
    OCCUPANCY_SAMPLE_GAP_SEC = .15
    REAR_LIMIT = HALF_LENGTH + .12
    SIDE_LIMIT = HALF_WIDTH + .12
    LIDAR_X = .105

    def __init__(self):
        self.epoch = self.lattice = self.last_grid = None
        self.cells = {}
        self.pruned_at = -1.
        self.oldest_age = None
        self.reused_cells = 0

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
                if observe and value >= 0:
                    acquired = view.scan_at if value == 0 else view.grid_at
                    if remembered is None or acquired >= remembered[1]:
                        occupied_since, confirmed = acquired, False
                        if value > 0 and remembered is not None and remembered[0] > 0:
                            _, last_seen, previous_since, confirmed = remembered
                            if acquired - last_seen <= self.OCCUPANCY_SAMPLE_GAP_SEC:
                                occupied_since = previous_since
                            confirmed = (confirmed and acquired - last_seen <= self.MAX_AGE_SEC
                                         or acquired - occupied_since >= self.OCCUPANCY_CONFIRM_SEC)
                        remembered = self.cells[key] = (value, acquired, occupied_since, confirmed)
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
