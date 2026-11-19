"""Experimental, slow forward-only controller for the standalone LIMO path.

The ray test is a sampling heuristic. It cannot prove that the space between
rays, the LiDAR near field, or the full 3-D vehicle volume is clear. Inputs
must share a clock and express rays in the current base frame. The caller must
recompute the fusion snapshot with a fresh, continuous pose for every step.

Depth evidence includes a 3x3 ``valid_grid``: each entry is the proportion
of valid depth pixels in one equal-sized image partition, ordered top to bottom
and left to right. The minimum is taken over all valid depth pixels. These image
statistics do not establish which physical volume the camera observes.
"""

from __future__ import division

import math

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - ROS Melodic Python 2
except NameError:
    INTEGER_TYPES = (int,)
NUMBER_TYPES = INTEGER_TYPES + (float,)


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (OverflowError, TypeError, ValueError):
        return False


def assess_front(scene, now, line_x=0.60, half_width=0.15, max_age=0.65, max_gap=0.035):
    """Sample the latest scan and retain all supplied hits, without certifying clearance."""
    result = {"candidate": False, "reason": "invalid_evidence", "geometric_clearance_verified": False}
    try:
        if (not _finite(now) or now < 0
                or not all(_finite(v) for v in (line_x, half_width, max_age, max_gap))
                or not 0.25 <= line_x <= 1.0 or not 0.05 <= half_width <= 0.5
                or not 0 < max_age <= 1.0 or not 0 < max_gap <= 0.05
                or not isinstance(scene, dict) or scene.get("status") != "ok"):
            return result
        coverage = scene.get("coverage")
        if not isinstance(coverage, dict) or coverage.get("sensor_fresh") is not True:
            return result
        scan_stamp = coverage.get("scan_stamp")
        front_samples = coverage.get("front_samples")
        latest_valid = coverage.get("latest_valid_samples")
        valid_fraction = coverage.get("latest_valid_fraction")
        if (not _finite(scan_stamp) or scan_stamp < 0
                or not 0 <= now - scan_stamp <= min(0.5, max_age)
                or isinstance(front_samples, bool) or not isinstance(front_samples, INTEGER_TYPES)
                or isinstance(latest_valid, bool) or not isinstance(latest_valid, INTEGER_TYPES)
                or not 1 <= front_samples <= 8192 or not 0 <= latest_valid <= front_samples
                or not _finite(valid_fraction) or not 0.5 <= valid_fraction <= 1.0
                or abs(valid_fraction - float(latest_valid) / front_samples) > 1e-9):
            return result
        rays = scene["rays"]
        hits = scene["occupied_hits"]
        if (not isinstance(rays, list) or not isinstance(hits, list)
                or len(rays) > 200000 or len(rays) != len(hits)):
            return result
        crossings = []
        latest_ray_count = 0
        for ray, hit in zip(rays, hits):
            if not isinstance(ray, dict) or not isinstance(hit, dict):
                return result
            start, end, stamp = ray["start_base"], ray["hit_base"], ray["stamp"]
            point, hit_stamp = hit["point_base"], hit["stamp"]
            if (not isinstance(start, (tuple, list)) or len(start) != 2
                    or not isinstance(end, (tuple, list)) or len(end) != 2
                    or not isinstance(point, (tuple, list)) or len(point) != 2
                    or not all(_finite(v) for v in
                               (start[0], start[1], end[0], end[1], point[0], point[1], stamp, hit_stamp))
                    or not 0 <= stamp <= scan_stamp or hit_stamp != stamp
                    or point[0] != end[0] or point[1] != end[1]):
                return result
            # Fusion owns evidence expiry; never discard a hit it still supplies.
            if 0.0 <= point[0] <= line_x and abs(point[1]) <= half_width:
                result["reason"] = "forward_obstacle"
                return result
            if stamp != scan_stamp:
                continue
            latest_ray_count += 1
            if not start[0] <= line_x < end[0]:
                continue
            fraction = (line_x - start[0]) / (end[0] - start[0])
            y = start[1] + fraction * (end[1] - start[1])
            if not _finite(y):
                return result
            if -half_width <= y <= half_width:
                crossings.append(y)
        if latest_ray_count != latest_valid:
            return result
        crossings.sort()
        gaps = ([crossings[0] + half_width, half_width - crossings[-1]]
                if crossings else [2 * half_width])
        gaps.extend(crossings[i] - crossings[i - 1] for i in range(1, len(crossings)))
        result["samples"] = len(crossings)
        result["max_sample_gap_m"] = max(gaps)
        if not crossings or result["max_sample_gap_m"] > max_gap:
            result["reason"] = "forward_coverage_incomplete"
            return result
        result.update(candidate=True, reason="sampled_forward_corridor")
        return result
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, IndexError):
        return result


class ForwardPolicy(object):  # noqa: UP004 - ROS Melodic Python 2
    """Bound a model's forward preference; unsupported turns remain stopped."""

    def __init__(self, started_at, deadline, speed=0.03):
        if (not all(_finite(v) for v in (started_at, deadline, speed)) or started_at < 0
                or deadline <= started_at or not 0 < speed <= 0.05):
            raise ValueError("invalid bounded forward policy")
        self.started_at = started_at
        self.deadline = deadline
        self.speed = speed
        self.last_at = started_at
        self.last_decision_stamp = None
        self.last_decision_choice = None
        self.linear = 0.0

    def step(self, scene, state, depth, decision, now):
        output = {"linear": 0.0, "angular": 0.0, "reason": "invalid_input"}
        try:
            if not _finite(now) or not 0 <= now - self.last_at <= 0.5 or now >= self.deadline:
                self.linear = 0.0
                return output
            dt = now - self.last_at
            self.last_at = now
            if (not isinstance(decision, dict) or not _finite(decision.get("stamp"))
                    or decision["stamp"] < self.started_at
                    or not 0 <= now - decision["stamp"] <= 8.0
                    or (self.last_decision_stamp is not None
                        and (decision["stamp"] < self.last_decision_stamp
                             or (decision["stamp"] == self.last_decision_stamp
                                 and self.last_decision_choice != "forward"
                                 and decision.get("choice") == "forward")))):
                output["reason"] = "no_fresh_forward_decision"
                self.linear = 0.0
                return output
            self.last_decision_stamp = decision["stamp"]
            self.last_decision_choice = decision.get("choice")
            if decision.get("choice") != "forward":
                output["reason"] = "no_fresh_forward_decision"
                self.linear = 0.0
                return output
            if (not isinstance(state, dict) or not _finite(state.get("stamp"))
                    or not 0 <= now - state["stamp"] <= 1.5
                    or type(state.get("motion_mode")) is not int or state["motion_mode"] != 0
                    or type(state.get("error_code")) is not int or state["error_code"] != 0
                    or not _finite(state.get("battery_voltage")) or state["battery_voltage"] < 10.0
                    or not _finite(state.get("linear_x_mps")) or not 0 <= state["linear_x_mps"] <= 0.12
                    or not _finite(state.get("angular_z_radps")) or abs(state["angular_z_radps"]) > 0.1):
                output["reason"] = "base_state_unavailable"
                self.linear = 0.0
                return output
            if (not isinstance(depth, dict) or not _finite(depth.get("stamp"))
                    or not 0 <= now - depth["stamp"] <= 0.8
                    or not _finite(depth.get("min_m")) or depth["min_m"] <= 0.5):
                output["reason"] = "depth_unavailable_or_close"
                self.linear = 0.0
                return output
            grid = depth.get("valid_grid")
            if (not isinstance(grid, (tuple, list)) or len(grid) != 3
                    or any(not isinstance(row, (tuple, list)) or len(row) != 3 for row in grid)
                    or any(not _finite(ratio) or not 0.5 <= ratio <= 1.0
                           for row in grid for ratio in row)):
                output["reason"] = "depth_unavailable_or_close"
                self.linear = 0.0
                return output
            evidence = assess_front(scene, now)
            output["reason"] = evidence["reason"]
            if not evidence["candidate"]:
                self.linear = 0.0
                return output
            self.linear = min(self.speed, self.linear + 0.12 * dt)
            output["linear"] = self.linear
            return output
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError, IndexError):
            self.linear = 0.0
            return output
