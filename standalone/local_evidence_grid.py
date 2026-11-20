"""Merge explicitly observed voxel IDs without inventing free space.

This diagnostic structure does not project sensors, verify registration or
calibration, cover the volume between samples, check a swept body, or authorize
motion. Callers must supply observations in the same fixed frame and epoch.
"""
from __future__ import division

import math
import threading

try:
    INTEGER_TYPES = (int, long)  # noqa: F821
    STRING_TYPES = (str, unicode)  # noqa: F821
except NameError:
    INTEGER_TYPES = (int,)
    STRING_TYPES = (str,)


def _number(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, INTEGER_TYPES + (float,))
                and not math.isnan(value) and not math.isinf(value))
    except (OverflowError, TypeError, ValueError):
        return False


class LocalEvidenceGrid(object):  # noqa: UP004 - ROS Melodic Python 2
    def __init__(self, shape, source_ages, frame, epoch):
        if (not isinstance(shape, (tuple, list)) or len(shape) != 3
                or any(type(value) is not int or not 1 <= value <= 128 for value in shape)
                or shape[0] * shape[1] * shape[2] > 262144
                or not isinstance(source_ages, dict) or not source_ages or len(source_ages) > 8
                or any(not isinstance(name, STRING_TYPES) or not name or not _number(age)
                       or not 0 < age <= 1. for name, age in source_ages.items())
                or not isinstance(frame, STRING_TYPES) or not frame or len(frame) > 128
                or isinstance(epoch, bool) or not isinstance(epoch, INTEGER_TYPES) or epoch < 0):
            raise ValueError('invalid grid configuration')
        self.shape, self.ages, self.frame, self.epoch = tuple(shape), dict(source_ages), frame, epoch
        self._free = {name: None for name in source_ages}
        self._stamp = {name: None for name in source_ages}
        self._occupied = []
        self._last_obstacle = {}
        self._last_now, self._fault = None, False
        self._lock = threading.Lock()

    def _cells(self, values):
        if not isinstance(values, (tuple, list)) or len(values) > 8192:
            raise ValueError('invalid cell list')
        result = []
        for cell in values:
            if (not isinstance(cell, (tuple, list)) or len(cell) != 3
                    or any(type(value) is not int or not 0 <= value < self.shape[index]
                           for index, value in enumerate(cell))):
                raise ValueError('invalid cell')
            result.append(tuple(cell))
        return result

    def _check_context(self, now, frame, epoch):
        if (self._fault or not _number(now) or now < 0
                or self._last_now is not None and now < self._last_now
                or frame != self.frame or type(epoch) is not type(self.epoch) or epoch != self.epoch):
            self._fault = True
            raise ValueError('invalid context')
        self._last_now = now

    def update(self, source, acquired_at, valid_until, now, frame, epoch, free, occupied):
        with self._lock:
            try:
                self._check_context(now, frame, epoch)
                if (not isinstance(source, STRING_TYPES) or source not in self.ages
                        or not _number(acquired_at) or not _number(valid_until)
                        or not 0 <= acquired_at <= now < valid_until <= acquired_at + self.ages[source]
                        or self._stamp[source] is not None and acquired_at <= self._stamp[source]):
                    raise ValueError('invalid observation')
                free_cells, occupied_cells = set(self._cells(free)), set(self._cells(occupied))
                if free_cells & occupied_cells:
                    raise ValueError('contradictory observation')
                retained = [(expires, cells) for expires, cells in self._occupied if expires > now]
                if occupied_cells:
                    retained.append((valid_until, occupied_cells))
                if len(retained) > 128:
                    raise ValueError('occupied history overflow')
                tombstones = {cell: stamp for cell, stamp in self._last_obstacle.items()
                              if stamp + max(self.ages.values()) > now}
                for cell in occupied_cells:
                    tombstones[cell] = max(acquired_at, tombstones.get(cell, acquired_at))
                if len(tombstones) > 32768:
                    raise ValueError('obstacle history overflow')
                self._occupied = retained
                self._last_obstacle = tombstones
                self._free[source] = (acquired_at, valid_until, frozenset(free_cells))
                self._stamp[source] = acquired_at
            except (TypeError, ValueError, OverflowError):
                self._fault = True
                raise ValueError('invalid grid evidence')

    def query(self, cells, now, frame, epoch):
        with self._lock:
            try:
                self._check_context(now, frame, epoch)
                wanted = self._cells(cells)
                self._occupied = [(expires, values) for expires, values in self._occupied if expires > now]
                self._last_obstacle = {cell: stamp for cell, stamp in self._last_obstacle.items()
                                       if stamp + max(self.ages.values()) > now}
                states, free_until = [], []
                for cell in wanted:
                    if any(cell in values for _, values in self._occupied):
                        states.append('occupied')
                        continue
                    barrier = self._last_obstacle.get(cell, -1.)
                    expiries = [value[1] for value in self._free.values()
                                if value is not None and value[0] > barrier and value[1] > now
                                and cell in value[2]]
                    if expiries:
                        states.append('free')
                        free_until.append(max(expiries))
                    else:
                        states.append('unknown')
                candidate = bool(states) and all(value == 'free' for value in states)
                return {'cells': states, 'sampled_candidate': candidate,
                        'valid_until': min(free_until) if candidate else None,
                        'unobserved_space': True, 'geometric_clearance_verified': False, 'executable': False}
            except (TypeError, ValueError, OverflowError):
                self._fault = True
                raise ValueError('invalid grid query')
