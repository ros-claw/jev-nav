"""Register previously observed blind-side clearance at a measured stop."""
from __future__ import division

import json
import math
import os

from navigation import HALF_LENGTH, HALF_WIDTH, Pose, _relative, ready
from trail_recovery import recorded_anchor_matches, recorded_scan_matches


MAX_HISTORY_SEC = 3600.
MAX_PIVOT_HISTORY_SEC = 600.
SECTORS = (-65., -45., -30., -12., 8., 30., 55., 70.)


def driver_generation(pid):
    with open('/proc/sys/kernel/random/boot_id') as stream:
        boot = stream.read().strip()
    with open('/proc/%d/stat' % pid) as stream:
        start = int(stream.read().rsplit(') ', 1)[1].split()[19])
    return [boot, start, os.readlink('/proc/%d/exe' % pid)]


def scene_sectors(scan):
    buckets = [[] for _ in SECTORS[:-1]]
    for index, distance in enumerate(scan.ranges):
        if not scan.range_min <= distance < scan.range_max:
            continue
        degrees = math.degrees(scan.angle_min + index * scan.angle_increment)
        for sector, (lo, hi) in enumerate(zip(SECTORS, SECTORS[1:])):
            if lo <= degrees < hi:
                buckets[sector].append(distance)
                break
    result = []
    for bucket in buckets:
        bucket.sort()
        result.append(bucket[len(bucket) // 2] if len(bucket) >= 4 else None)
    return result


def _finite(value):
    return not (math.isnan(value) or math.isinf(value))


def inside_measured_body(pose, x, y, resolution):
    half = resolution / 2.
    for dx in (-half, half):
        for dy in (-half, half):
            offset = _relative(pose, Pose(x + dx, y + dy, 0.))
            if abs(offset.x) > HALF_LENGTH - .002 or abs(offset.y) > HALF_WIDTH - .002:
                return False
    return True


class RoomRevisit(object):
    def __init__(self, state):
        if state['version'] != 1 or state['frame'] != 'odom' or not 0 < len(state['cells']) <= 384:
            raise ValueError('invalid room revisit')
        self.pose = Pose(*state['pose'])
        self.at, self.resolution = state['recorded_stop'], state['resolution']
        self.cells = [tuple(row) for row in state['cells']]
        self.closest = state['closest_rays']
        self.sectors = state['sectors']
        pid = state['driver_pid']
        if (type(pid) is not int or pid < 2 or not .03 <= self.resolution <= .10 or self.at <= 0
                or not all(_finite(value) for value in self.pose + (self.at, self.resolution))
                or not 6 <= len(self.closest) <= 20
                or any(not (-80. <= a <= 80. and .08 <= d <= 5.) for a, d in self.closest)
                or len(self.sectors) != len(SECTORS) - 1
                or sum(v is not None for v in self.sectors) < 5
                or any(v is not None and not .08 <= v <= 12. for v in self.sectors)
                or any(not all(_finite(v) for v in (x, y, at)) or not 0 <= self.at - at < MAX_HISTORY_SEC
                       or math.hypot(x - self.pose.x, y - self.pose.y) > .44 for x, y, at in self.cells)):
            raise ValueError('invalid room revisit evidence')
        if driver_generation(pid) != state['driver_generation']:
            raise ValueError('room revisit driver generation changed')
        self.body_cells = []
        body = state.get('measured_body')
        if body is not None:
            origin, at, cells = Pose(*body['pose']), body['at'], body['cells']
            if (not all(_finite(v) for v in origin + (at,)) or not 0 <= self.at - at < MAX_PIVOT_HISTORY_SEC
                    or math.hypot(origin.x - self.pose.x, origin.y - self.pose.y) > .025
                    or len(cells) > 64
                    or any(not (_finite(x) and _finite(y) and inside_measured_body(origin, x, y, self.resolution))
                           for x, y in cells)):
                raise ValueError('recorded pivot body does not cover the cells')
            self.body_cells = [(x, y, at) for x, y in cells]

    @classmethod
    def load(cls, path):
        with open(path) as stream:
            return cls(json.load(stream))

    def registered_cells(self, view, scan, now, ros_now):
        if (not ready(view, now) or scan is None or not 0 <= ros_now - self.at < MAX_HISTORY_SEC
                or not 0 <= ros_now - scan.header.stamp.to_sec() < .30
                or scan.header.frame_id.lstrip('/') != 'laser_link'
                or abs(view.grid.resolution - self.resolution) > 1e-6
                or not recorded_anchor_matches(self.pose, view.pose)):
            return []
        seen = scene_sectors(scan)
        matched = [(old is not None and new is not None and abs(new - old) <= .05 + .025 * old)
                   for old, new in zip(self.sectors, seen)]
        if not (sum(matched) >= 5 and any(matched[:2]) and any(matched[4:])
                and recorded_scan_matches(scan, {'closest_rays': self.closest})):
            return []
        observations = [(x, y, now - (ros_now - observed)) for x, y, observed in self.cells
                        if 0 <= ros_now - observed < MAX_HISTORY_SEC]
        observations.extend((x, y, now - (ros_now - at)) for x, y, at in self.body_cells
                            if 0 <= ros_now - at < MAX_PIVOT_HISTORY_SEC)
        return observations
