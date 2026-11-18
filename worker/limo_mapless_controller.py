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
POSE_TRANSLATION_BUDGET = 0.12
POSE_ROTATION_BUDGET = 0.28


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, NUMBER_TYPES):
        return False
    try:
        return not math.isnan(value) and not math.isinf(value)
    except (OverflowError, ValueError):
        return False


def _approach(current, target, limit):
    return current + max(-limit, min(limit, target - current))


def _budgeted_target(current, horizon, budget, acceleration, maximum):
    """Bound a commanded target, accounting for the modeled deceleration ramp."""
    current = abs(current)
    if budget <= 0:
        return 0.0 if current == 0 else None

    def projected(target):
        if current <= target:
            return target * horizon
        ramp = (current - target) / acceleration
        if ramp >= horizon:
            return current * horizon - 0.5 * acceleration * horizon * horizon
        return target * horizon + 0.5 * (current - target) * ramp

    if projected(0.0) > budget:
        return None
    if projected(maximum) <= budget:
        return maximum
    low, high = 0.0, maximum
    for _ in range(40):
        middle = (low + high) / 2.0
        if projected(middle) <= budget:
            low = middle
        else:
            high = middle
    return low


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
        self.linear_limit = 0.1
        self.angular_limit = 0.25

    def set_decision(self, choice, observed_at, now, translation_budget=POSE_TRANSLATION_BUDGET,
                     rotation_budget=POSE_ROTATION_BUDGET, measured_linear=0.0, measured_angular=0.0):
        if (
            choice not in ("forward", "left", "right", "stop")
            or not _finite(observed_at)
            or not _finite(now)
            or not self.observed_at < observed_at <= now < observed_at + DECISION_TTL
            or now >= self.deadline
            or now < self.last_tick
            or not all(_finite(value) for value in
                       (translation_budget, rotation_budget, measured_linear, measured_angular))
            or translation_budget < 0 or rotation_budget < 0
        ):
            return False
        self.choice = choice
        self.observed_at = observed_at
        # The next decision must arrive before the existing pose window runs
        # out. Adapt to measured age; an unexpectedly longer delay still stops.
        planning_horizon = 2.0 * (now - observed_at) + 0.5
        linear_limit = _budgeted_target(max(abs(self.linear), abs(measured_linear)),
                                        planning_horizon, translation_budget, 0.3, 0.1)
        angular_limit = _budgeted_target(max(abs(self.angular), abs(measured_angular)),
                                         planning_horizon, rotation_budget, 0.8, 0.25)
        if (linear_limit is None or angular_limit is None
                or (choice in ("left", "right") and angular_limit <= 1e-9)):
            self.choice = "stop"
            self.linear, self.angular = 0.0, 0.0
            self.linear_limit, self.angular_limit = 0.0, 0.0
            return True
        self.linear_limit, self.angular_limit = linear_limit, angular_limit
        return True

    def accept_stop(self, observed_at, now):
        # A repeated older sensor stamp must not discard an explicit stop.
        if (not _finite(observed_at) or not _finite(now) or observed_at > now
                or now < self.last_tick):
            return False
        self.choice = "stop"
        self.observed_at = max(self.observed_at, observed_at)
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
        # Candidate bound using the controller's permitted speed, not a
        # measured physical braking or obstacle-motion guarantee.
        scan_age_distance = 0.2 * (now - observation["scan_at"])
        depth_age_distance = 0.2 * (now - observation["depth_at"])
        front_candidates = [observation["front"]]
        front_candidates.extend(observation[side] for side in ("left", "right")
                                if _finite(observation.get(side)))
        front = min(front_candidates) - scan_age_distance
        depth = observation["depth_min"] - depth_age_distance
        direction = self.choice
        if direction == "stop":
            self.linear, self.angular = 0.0, 0.0
            return self.linear, self.angular
        front_blocked = front <= 0.6 or depth <= 0.4
        if front_blocked:
            self.linear, self.angular = 0.0, 0.0
            return self.linear, self.angular
        if direction == "forward":
            target_linear, target_angular = self.linear_limit, 0.0
        else:
            target_linear = min(0.07, self.linear_limit, 0.28 * self.angular_limit)
            target_angular = self.angular_limit if direction == "left" else -self.angular_limit
        next_linear = _approach(self.linear, target_linear, 0.3 * dt)
        next_angular = _approach(self.angular, target_angular, 0.8 * dt)
        angular_motion = (self.angular, next_angular, observation["angular_z_radps"])
        turning = any(abs(v) > 1e-6 for v in angular_motion)
        for side in ("left", "right", "rear"):
            required = turning or (side != "rear" and next_linear > 0)
            if required and (not _finite(observation.get(side))
                             or observation[side] - scan_age_distance <= 0.5):
                self.linear, self.angular = 0.0, 0.0
                return self.linear, self.angular
        self.linear, self.angular = next_linear, next_angular
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
        for key in (
            "front",
            "depth_min",
            "depth_valid_ratio",
            "battery_voltage",
            "linear_x_mps",
            "angular_z_radps",
        ):
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
            and abs(observation["linear_x_mps"]) <= 0.2
            and abs(observation["angular_z_radps"]) <= 0.6
        )
