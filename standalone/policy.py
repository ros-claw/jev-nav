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
    import numpy as _np
except ImportError:  # scalar implementation remains available
    _np = None

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - ROS Melodic Python 2
except NameError:
    INTEGER_TYPES = (int,)
NUMBER_TYPES = INTEGER_TYPES + (float,)
_INFINITY = float("inf")


def _finite(value):
    if type(value) is float:
        return -_INFINITY < value < _INFINITY
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _largest_gap(crossings, half_width):
    if not crossings:
        return 2 * half_width
    values = sorted(crossings)
    gaps = [values[0] + half_width, half_width - values[-1]]
    gaps.extend(values[i] - values[i - 1] for i in range(1, len(values)))
    return max(gaps)


def _fast_assess_rays(rays, hits, scan_stamp, latest_valid, now, line_x, half_width,
                      max_age, max_gap, history_age_limit, history_span_limit):
    """Vectorize the immutable, shared tuple form emitted by scan fusion.

    Return None for any other input shape so the full scalar validator keeps
    its original behavior for malformed or externally constructed evidence.
    """
    if _np is None or len(rays) < 32:
        return None
    rows = []
    for ray, hit in zip(rays, hits):
        if type(ray) is not dict or type(hit) is not dict:
            return None
        try:
            start, end, stamp = ray['start_base'], ray['hit_base'], ray['stamp']
            if (type(start) is not tuple or len(start) != 2 or type(end) is not tuple or len(end) != 2
                    or hit['point_base'] is not end or hit['stamp'] is not stamp or type(stamp) is not float
                    or type(start[0]) is not float or type(start[1]) is not float
                    or type(end[0]) is not float or type(end[1]) is not float):
                return None
            rows.append((start[0], start[1], end[0], end[1], stamp))
        except (KeyError, IndexError, TypeError):
            return None
    try:
        data = _np.asarray(rows, dtype=_np.float64)
        sx, sy, hx, hy, stamps = data.T
        if (not _np.all(_np.isfinite(data)) or _np.any(stamps < 0) or _np.any(stamps > scan_stamp)):
            return None
        result = {"candidate": False, "reason": "invalid_evidence", "geometric_clearance_verified": False}
        if _np.any((hx >= -0.18) & (hx <= line_x) & (_np.abs(hy) <= max(half_width, 0.15))):
            result['reason'] = 'forward_obstacle'
            return result
        latest = stamps == scan_stamp
        if int(_np.count_nonzero(latest)) != latest_valid:
            return result
        age = now - stamps
        recent = ((age >= 0) & (age <= min(history_age_limit, max_age))
                  & (scan_stamp - stamps <= history_span_limit)
                  & (sx <= line_x) & (hx > line_x))
        indices = _np.flatnonzero(recent)
        with _np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            crossing_y = sy[indices] + ((line_x - sx[indices]) / (hx[indices] - sx[indices])) * (hy[indices] - sy[indices])
        if not _np.all(_np.isfinite(crossing_y)):
            return None
        inside = (crossing_y >= -half_width) & (crossing_y <= half_width)
        crossings = crossing_y[inside].tolist()
        latest_crossings = crossing_y[inside & latest[indices]].tolist()
        result['samples'] = len(crossings)
        result['max_sample_gap_m'] = _largest_gap(crossings, half_width)
        result['latest_max_sample_gap_m'] = _largest_gap(latest_crossings, half_width)
        if (not crossings or not latest_crossings or result['max_sample_gap_m'] > max_gap
                or result['latest_max_sample_gap_m'] > 2 * max_gap):
            result['reason'] = 'forward_coverage_incomplete'
            return result
        result.update(candidate=True, reason='sampled_forward_corridor')
        return result
    except (ArithmeticError, AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return None


def assess_front(scene, now, line_x=0.60, half_width=0.15, max_age=0.65, max_gap=0.035,
                 latest_age_limit=0.3, history_age_limit=0.45, history_span_limit=0.25):
    """Sample recent scans and retain all supplied hits, without certifying clearance.

    Rays must already be transformed into the actual current base frame. This
    function receives neither a current pose nor a pose timestamp and cannot
    verify that requirement; the caller must reproject with fresh, continuous
    odometry on each step and validate sensor/pose synchronization. Historical
    hits always block when inside the checked area; historical rays may fill
    sampling gaps only within both the built-in and caller-specified age limits.
    """
    result = {"candidate": False, "reason": "invalid_evidence", "geometric_clearance_verified": False}
    try:
        if (not _finite(now) or now < 0
                or not all(_finite(v) for v in (line_x, half_width, max_age, max_gap,
                                                latest_age_limit, history_age_limit, history_span_limit))
                or not 0.25 <= line_x <= 1.0 or not 0.05 <= half_width <= 0.5
                or not 0 < max_age <= 1.0 or not 0 < max_gap <= 0.035
                or not 0 < latest_age_limit <= 0.5 or not 0 < history_age_limit <= 0.65
                or not 0 < history_span_limit <= 0.4
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
                or not 0 <= now - scan_stamp <= min(latest_age_limit, max_age)
                or isinstance(front_samples, bool) or not isinstance(front_samples, INTEGER_TYPES)
                or isinstance(latest_valid, bool) or not isinstance(latest_valid, INTEGER_TYPES)
                or not 1 <= front_samples <= 8192 or not 0 <= latest_valid <= front_samples
                or not _finite(valid_fraction) or not 0 < valid_fraction <= 1.0
                or abs(valid_fraction - float(latest_valid) / front_samples) > 1e-9):
            return result
        rays = scene["rays"]
        hits = scene["occupied_hits"]
        if (not isinstance(rays, list) or not isinstance(hits, list)
                or len(rays) > 200000 or len(rays) != len(hits)):
            return result
        fast = _fast_assess_rays(rays, hits, scan_stamp, latest_valid, now, line_x, half_width,
                                 max_age, max_gap, history_age_limit, history_span_limit)
        if fast is not None:
            return fast
        crossings = []
        latest_crossings = []
        latest_ray_count = 0
        for ray, hit in zip(rays, hits):
            if not isinstance(ray, dict) or not isinstance(hit, dict):
                return result
            start, end, stamp = ray["start_base"], ray["hit_base"], ray["stamp"]
            point, hit_stamp = hit["point_base"], hit["stamp"]
            # Fusion snapshots share immutable tuples and timestamps between
            # ray and hit. Keep full validation for all other input shapes.
            if (type(start) is tuple and len(start) == 2 and type(end) is tuple and len(end) == 2
                    and point is end and hit_stamp is stamp and type(stamp) is float
                    and type(start[0]) is float and type(start[1]) is float
                    and type(end[0]) is float and type(end[1]) is float):
                if not (-_INFINITY < start[0] < _INFINITY and -_INFINITY < start[1] < _INFINITY
                        and -_INFINITY < end[0] < _INFINITY and -_INFINITY < end[1] < _INFINITY
                        and -_INFINITY < stamp < _INFINITY and 0 <= stamp <= scan_stamp):
                    return result
            else:
                if (not isinstance(start, (tuple, list)) or len(start) != 2
                        or not isinstance(end, (tuple, list)) or len(end) != 2
                        or not isinstance(point, (tuple, list)) or len(point) != 2
                        or not _finite(start[0]) or not _finite(start[1])
                        or not _finite(end[0]) or not _finite(end[1])
                        or not _finite(point[0]) or not _finite(point[1])
                        or not _finite(stamp) or not _finite(hit_stamp)
                        or not 0 <= stamp <= scan_stamp or hit_stamp != stamp
                        or point[0] != end[0] or point[1] != end[1]):
                    return result
            # Fusion owns evidence expiry; never discard a hit it still supplies.
            # Include the example rear footprint (-0.16 m) plus 0.02 m padding;
            # narrowing the sampled line must not shrink the obstacle check.
            if -0.18 <= point[0] <= line_x and abs(point[1]) <= max(half_width, 0.15):
                result["reason"] = "forward_obstacle"
                return result
            if stamp == scan_stamp:
                latest_ray_count += 1
            # Only a short rolling window contributes to sampled coverage;
            # older retained hits above still block, but do not fill gaps.
            if (not 0 <= now - stamp <= min(history_age_limit, max_age)
                    or scan_stamp - stamp > history_span_limit):
                continue
            if not start[0] <= line_x < end[0]:
                continue
            fraction = (line_x - start[0]) / (end[0] - start[0])
            y = start[1] + fraction * (end[1] - start[1])
            if not _finite(y):
                return result
            if -half_width <= y <= half_width:
                crossings.append(y)
                if stamp == scan_stamp:
                    latest_crossings.append(y)
        if latest_ray_count != latest_valid:
            return result
        result["samples"] = len(crossings)
        result["max_sample_gap_m"] = _largest_gap(crossings, half_width)
        result["latest_max_sample_gap_m"] = _largest_gap(latest_crossings, half_width)
        if (not crossings or not latest_crossings or result["max_sample_gap_m"] > max_gap
                or result["latest_max_sample_gap_m"] > 2 * max_gap):
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
                    or not _finite(state.get("linear_y_mps")) or abs(state["linear_y_mps"]) > 0.01
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
