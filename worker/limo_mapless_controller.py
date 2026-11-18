"""Mapless local controller for the daemon-owned LIMO worker.

All observations must be sampled by the worker. Thresholds are initial test
parameters and do not prove coverage of physical obstacles or braking distance.
"""

import math

try:
    NUMBER_TYPES = (int, long, float)  # noqa: F821
except NameError:
    NUMBER_TYPES = (int, float)

DECISION_TTL = 8.0


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, NUMBER_TYPES):
        return False
    try:
        return not math.isnan(value) and not math.isinf(value)
    except (OverflowError, ValueError):
        return False


def _approach(current, target, limit):
    return current + max(-limit, min(limit, target - current))


class MaplessController(object):  # noqa: UP004 - Python 2 ROS worker
    def __init__(self, started_at, deadline):
        if not _finite(started_at) or not _finite(deadline) or deadline <= started_at:
            raise ValueError("invalid mission times")
        self.deadline = deadline
        self.last_tick = started_at
        self.observed_at = float("-inf")
        self.choice = "stop"
        self.linear = 0.0
        self.angular = 0.0

    def set_decision(self, choice, observed_at, now):
        if (
            choice not in ("forward", "left", "right", "stop")
            or not _finite(observed_at)
            or not _finite(now)
            or not self.observed_at < observed_at <= now < observed_at + DECISION_TTL
            or now >= self.deadline
            or now < self.last_tick
        ):
            return False
        self.choice = choice
        self.observed_at = observed_at
        return True

    def tick(self, observation, now):
        if not _finite(now):
            self.linear, self.angular = 0.0, 0.0
            return self.linear, self.angular
        dt = now - self.last_tick
        if dt < 0:
            self.linear, self.angular = 0.0, 0.0
            return self.linear, self.angular
        self.last_tick = now
        if (
            dt > 0.5
            or now >= self.deadline
            or now >= self.observed_at + DECISION_TTL
            or not self._fresh(observation, now)
        ):
            self.linear, self.angular = 0.0, 0.0
            return self.linear, self.angular
        front = observation["front"]
        depth = observation["depth_min"]
        direction = self.choice
        if direction == "stop":
            self.linear, self.angular = 0.0, 0.0
            return self.linear, self.angular
        if direction == "forward":
            if front <= 0.6 or depth <= 0.4:
                self.linear, self.angular = 0.0, 0.0
                return self.linear, self.angular
            target_linear, target_angular = 0.1, 0.0
        else:
            side = observation[direction]
            if not _finite(side) or side <= 0.5:
                self.linear, self.angular = 0.0, 0.0
                return self.linear, self.angular
            target_linear = 0.07 if front > 0.6 and depth > 0.4 else 0.0
            target_angular = 0.25 if direction == "left" else -0.25
        self.linear = _approach(self.linear, target_linear, 0.3 * dt)
        self.angular = _approach(self.angular, target_angular, 0.8 * dt)
        return self.linear, self.angular

    @staticmethod
    def _fresh(observation, now):
        if not isinstance(observation, dict):
            return False
        for key, limit in (
            ("scan_at", 0.6),
            ("status_at", 1.5),
            ("odom_at", 0.8),
            ("depth_at", 1.0),
        ):
            stamp = observation.get(key)
            if not _finite(stamp) or not 0 <= now - stamp <= limit:
                return False
        for key in ("front", "depth_min", "depth_valid_ratio", "battery_voltage"):
            if not _finite(observation.get(key)):
                return False
        return (
            observation["front"] > 0
            and observation["depth_min"] > 0
            and 0.05 <= observation["depth_valid_ratio"] <= 1.0
            and type(observation.get("error_code")) is int
            and observation["error_code"] == 0
            and type(observation.get("motion_mode")) is int
            and observation["motion_mode"] == 0
            and observation["battery_voltage"] >= 10
        )
