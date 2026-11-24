"""Validate a short straight retreat within a previously driven static-room trail."""
from __future__ import division

import math

from navigation import Command, HALF_LENGTH, HALF_WIDTH, Pose, _corners, _hull, _relative, _wrap
from progress_monitor import PROGRESS_FAULTS

TRAIL_MARGIN_M = .04
MAX_LATERAL_M = .006
MAX_HEADING_RAD = .025
LIDAR_X = .105
RECORDED_ANCHOR_DISTANCE_M = .005
RECORDED_ANCHOR_HEADING_RAD = .015
STOPPING_HORIZON_SEC = .30
FRONT_PADDING_M = .012
RETRACE_CLEAR_SEC = .35
RETRACE_BLOCK_SEC = 2.


def _recording_rows(trace):
    trajectory = trace['trajectory']
    if trace.get('reason') in PROGRESS_FAULTS:
        raise ValueError('recorded trail lacks trustworthy physical progress')
    if trace.get('spatial_memory') or trace.get('remembered_depth_hazards'):
        raise ValueError('recovery requires a current registered hazard check for this recording')
    if (trace['mode'] != 'supervised_static_room' or trace['relay_exited'] is not True
            or trace.get('relay_fault') is not None or trajectory['clock'] != 'host_monotonic_receipt'):
        raise ValueError('curved retrace needs a completed supervised static-room recording')
    rows = trajectory['odom']
    if not rows:
        raise ValueError('missing recorded odometry')
    return rows


def recording_driver(trace):
    return trace.get('authority_diagnostics', {}).get('ros', {}).get('driver_pid_expected')


def _reverse_rows(current, prior):
    for row in reversed(current):
        yield row
    if prior:
        yield None  # There was no recorded motion across this stationary join.
        for row in reversed(prior):
            yield row


def recorded_anchor_matches(anchor, pose):
    offset = _relative(anchor, pose)
    return (math.hypot(offset.x, offset.y) <= RECORDED_ANCHOR_DISTANCE_M
            and abs(offset.yaw) <= RECORDED_ANCHOR_HEADING_RAD)


def measured_trail(trace, reference, distance, start_sequence=None, prior=None):
    """Recover pose-relative keyframes from completed physical recordings."""
    if not .04 <= distance <= .45:
        raise ValueError('invalid curved retrace distance')
    rows = _recording_rows(trace)
    if start_sequence is not None:
        matches = [index for index, row in enumerate(rows) if len(row) >= 8 and row[7] == start_sequence]
        if len(matches) != 1:
            raise ValueError('recorded start sequence must identify exactly one odometry sample')
        rows = rows[:matches[0] + 1]
    end = Pose(rows[-1][1], rows[-1][2], rows[-1][4])
    if not recorded_anchor_matches(end, Pose(reference['x'], reference['y'], reference['yaw'])):
        raise ValueError('front reference must match the recorded start pose')
    report = {'retreat_distance_m': distance, 'recorded_anchor': end}
    prior_rows = None
    if prior is not None:
        prior_rows = _recording_rows(prior)
        old, new = prior_rows[-1], rows[0]
        if (recording_driver(trace) is None or recording_driver(trace) != recording_driver(prior)
                or min(len(old), len(new)) < 8 or old[1:6] != new[1:6]
                or new[3] != 0. or new[5] != 0. or not old[0] < new[0]
                or not old[6] < new[6] or not old[7] < new[7]):
            raise ValueError('recordings must join at the same stopped pose with the same driver')
        report['recording_join_gap_sec'] = new[0] - old[0]
    points = [Pose(0., 0., 0.)]
    length, turn = 0., 0.
    previous = None
    for row in _reverse_rows(rows, prior_rows):
        if row is None:
            previous = None
            continue
        if not all(not math.isnan(value) and not math.isinf(value) for value in row[:6]):
            raise ValueError('invalid recorded odometry')
        pose = _relative(end, Pose(row[1], row[2], row[4]))
        if previous is not None:
            step = _relative(previous[1], pose)
            if (not 0 < previous[0] - row[0] <= .20 or math.hypot(step.x, step.y) > .01
                    or abs(step.yaw) > .035 or step.x > .002):
                raise ValueError('recorded trail is discontinuous or contains forward retrace motion')
        previous = row[0], pose
        delta = _relative(points[-1], pose)
        linear = math.hypot(delta.x, delta.y)
        if linear < .015 and abs(delta.yaw) < .039 and length + linear < distance:
            continue
        if linear > .018 or abs(delta.y) > .005 or abs(delta.yaw) > .055:
            raise ValueError('recorded keyframe is not reachable within its corridor')
        length += linear
        turn += abs(delta.yaw)
        if turn > 1.1:
            raise ValueError('recorded trail exceeds the recovery rotation budget')
        points.append(pose)
        if length >= distance:
            report.update(measured_trail_m=length, measured_turn_rad=turn, keyframes=points)
            return points, report
    raise ValueError('recorded trail is too short')


class TrailFollower(object):
    """Track short recorded segments in order using current wheel odometry."""
    def __init__(self, points):
        self.points = points
        self.index = 0

    def step(self, pose):
        while self.index + 1 < len(self.points):
            start, end = self.points[self.index:self.index + 2]
            dx, dy = end.x - start.x, end.y - start.y
            distance = math.hypot(dx, dy)
            turn = _wrap(end.yaw - start.yaw)
            if distance < .007:
                error = _wrap(end.yaw - pose.yaw)
                if math.hypot(end.x - pose.x, end.y - pose.y) > .012 or abs(error) > .11:
                    return Command(0., 0., 'stop', 'retrace_path_departure')
                if abs(error) > .014:
                    angular = math.copysign(min(.05, max(.027, 2.4 * abs(error))), error)
                    return Command(0., angular, 'reverse', 'recorded_pivot')
            else:
                x, y = pose.x - start.x, pose.y - start.y
                along = (x * dx + y * dy) / distance
                fraction = min(1., max(0., along / distance))
                yaw = _wrap(start.yaw + turn * fraction)
                error = _wrap(yaw - pose.yaw)
                if math.hypot(x - fraction * dx, y - fraction * dy) > .014 or abs(error) > .11:
                    return Command(0., 0., 'stop', 'retrace_path_departure')
                if along < distance - .002:
                    lateral = -(x - fraction * dx) * math.sin(yaw) + (y - fraction * dy) * math.cos(yaw)
                    linear = .018
                    angular = max(-.045, min(.045, turn * linear / distance + 2. * error + 9. * lateral))
                    return Command(-linear, angular, 'reverse', 'recorded_retrace')
            self.index += 1
        return Command(0., 0., 'stop', 'retrace_distance_reached')


class RetraceClearance(object):
    """Hold zero for a transient obstruction until fresh scans clear the same trail."""
    def __init__(self):
        self.blocked_at = self.clear_since = self.last_clear_at = None
        self.pauses = self.resumes = 0

    def observe(self, now, scan_at, blocked):
        if blocked and self.blocked_at is None:
            self.blocked_at = now
            self.pauses += 1
        if self.blocked_at is None:
            return 'clear'
        if blocked or scan_at is None:
            self.clear_since = self.last_clear_at = None
        else:
            if self.last_clear_at is None or not 0 <= scan_at - self.last_clear_at < .30:
                self.clear_since = scan_at
            self.last_clear_at = scan_at
            if scan_at - self.clear_since >= RETRACE_CLEAR_SEC:
                self.blocked_at = self.clear_since = self.last_clear_at = None
                self.resumes += 1
                return 'clear'
        return 'timed_out' if now - self.blocked_at >= RETRACE_BLOCK_SEC else 'hold'


def _stopping_footprint(linear, angular, half_length, half_width):
    def corners(x, y, yaw):
        return _corners(Pose(x, y, yaw), half_length, half_length, half_width)

    initial = corners(0., 0., 0.)
    angle = angular * STOPPING_HORIZON_SEC
    if abs(angle) < 1e-8:
        return _hull(initial + corners(linear * STOPPING_HORIZON_SEC, 0., 0.))

    radius = linear / angular
    end = corners(radius * math.sin(angle), radius * (1. - math.cos(angle)), angle)
    middle = corners(radius * math.sin(angle / 2.), radius * (1. - math.cos(angle / 2.)), angle / 2.)
    # Include the tangent intersection for every corner to enclose the whole sweep.
    scale = 1. / math.cos(angle / 2.)
    outer = [(x * scale, radius + (y - radius) * scale) for x, y in middle]
    return _hull(initial + end + outer)


def front_obstruction(scan, angular, linear=0.):
    """Check live front returns against the footprint over the stopping horizon."""
    half_length, half_width = HALF_LENGTH + FRONT_PADDING_M, HALF_WIDTH + FRONT_PADDING_M
    polygon = _stopping_footprint(linear, angular, half_length, half_width)
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    front_limit = max(x for x, _ in polygon)
    valid = 0
    for index, value in enumerate(scan.ranges):
        angle = scan.angle_min + index * scan.angle_increment
        if not (-math.pi / 2 <= angle <= math.pi / 2 and scan.range_min <= value < scan.range_max):
            continue
        valid += 1
        x, y = LIDAR_X + value * math.cos(angle), value * math.sin(angle)
        current = abs(x) <= half_length and abs(y) <= half_width
        if (x <= front_limit and all((b[0] - a[0]) * (y - a[1]) >= (b[1] - a[1]) * (x - a[0]) - 1e-10
                                     for a, b in edges)):
            return {'x_m': x, 'y_m': y, 'bearing_deg': math.degrees(angle), 'range_m': value,
                    'current_footprint': current}
    return {'valid_returns': valid} if valid < 6 else None


def straight_trail(trace, reference, distance):
    trajectory = trace['trajectory']
    if (trace['mode'] != 'supervised_static_room' or trace['relay_exited'] is not True
            or trajectory['clock'] != 'host_monotonic_receipt' or not .10 <= distance <= .35):
        raise ValueError('retrace needs a completed supervised static-room recording')
    rows = _recording_rows(trace)
    if not rows:
        raise ValueError('missing recorded odometry')
    end = Pose(rows[-1][1], rows[-1][2], rows[-1][4])
    anchor = _relative(end, Pose(reference['x'], reference['y'], reference['yaw']))
    if math.hypot(anchor.x, anchor.y) > .005 or abs(anchor.yaw) > .015:
        raise ValueError('front reference must match the end of the recorded path')
    needed = distance + TRAIL_MARGIN_M
    previous = None
    for row in reversed(rows):
        if not all(not math.isnan(value) and not math.isinf(value) for value in row[:6]):
            raise ValueError('invalid recorded odometry')
        relative = _relative(end, Pose(row[1], row[2], row[4]))
        if (relative.x > MAX_LATERAL_M or abs(relative.y) > MAX_LATERAL_M
                or abs(relative.yaw) > MAX_HEADING_RAD):
            raise ValueError('most recent trail is not straight')
        if previous is not None and (not 0 < previous[0] - row[0] <= .20
                                     or abs(previous[1] - relative.x) > .01):
            raise ValueError('recorded trail has a gap')
        if -relative.x >= needed:
            return {'measured_trail_m': -relative.x, 'retreat_distance_m': distance}
        previous = (row[0], relative.x)
    raise ValueError('recorded straight trail is too short')


def scan_matches(scan, reference):
    expected = reference['closest_rays']
    if len(expected) < 6:
        return False
    count = 0
    for angle, distance in expected:
        index = int(round((math.radians(angle) - scan.angle_min) / scan.angle_increment))
        if not 0 <= index < len(scan.ranges):
            continue
        value = scan.ranges[index]
        if scan.range_min <= value <= scan.range_max and abs(value - distance) <= .025:
            count += 1
    return count >= len(expected) * .8


def recorded_scan_matches(scan, reference):
    """Match distinct neighboring beams despite sub-degree scanner sampling jitter."""
    expected = sorted(reference['closest_rays'])
    shift = front_registration(scan, reference)
    if shift is None or abs(shift) > .008:
        return False
    matched = set()
    for degrees, distance in expected:
        angle = math.radians(degrees)
        index = int(round((angle - scan.angle_min) / scan.angle_increment))
        candidates = []
        for beam in range(max(0, index - 2), min(len(scan.ranges), index + 3)):
            value = scan.ranges[beam]
            separation = abs(angle - (scan.angle_min + beam * scan.angle_increment))
            if (beam not in matched and separation <= math.radians(.85)
                    and scan.range_min <= value < scan.range_max and abs(value - distance) <= .012):
                candidates.append((separation, beam))
        if candidates:
            matched.add(min(candidates)[1])
    return len(matched) >= len(expected) * .8


def prior_retrace_offset(report, distance):
    rows = report['trajectory']['odom']
    if (report['relay_exited'] is not True or report['retreat_distance_m'] != distance or len(rows) < 2):
        raise ValueError('prior retreat recording does not match this trail')
    if 'final_offset' in report:
        pose = Pose(*report['final_offset'])
    else:
        begin, end = rows[0], rows[-1]
        pose = _relative(Pose(begin[1], begin[2], begin[4]), Pose(end[1], end[2], end[4]))
    if (not -distance < pose.x < 0. or abs(pose.y) > MAX_LATERAL_M or abs(pose.yaw) > .04
            or abs(pose.x + report['displacement_m']) > .006):
        raise ValueError('prior retreat departed from the measured trail')
    return pose


def predicted_front(reference, offset):
    rays = []
    for degrees, distance in reference['closest_rays']:
        angle = math.radians(degrees)
        point = _relative(offset, Pose(LIDAR_X + distance * math.cos(angle), distance * math.sin(angle), 0.))
        x = point.x - LIDAR_X
        rays.append((math.degrees(math.atan2(point.y, x)), math.hypot(x, point.y)))
    return dict(reference, closest_rays=rays)


def front_registration(scan, reference):
    """Estimate the small longitudinal discrepancy in a partial retreat."""
    expected = reference['closest_rays']
    if len(expected) < 6:
        return None
    lower = min(row[0] for row in expected) - .8
    upper = max(row[0] for row in expected) + .8
    shifts = []
    for index, distance in enumerate(scan.ranges):
        angle = scan.angle_min + index * scan.angle_increment
        degrees = math.degrees(angle)
        if not lower <= degrees <= upper or not scan.range_min <= distance <= scan.range_max:
            continue
        match = min(expected, key=lambda item: abs(item[0] - degrees))
        residual = (distance - match[1]) * math.cos(angle)
        if abs(residual) <= .055:
            shifts.append(residual)
    if len(shifts) < 4:
        return None
    shifts.sort()
    median = shifts[len(shifts) // 2]
    if abs(median) > .042 or sum(abs(value - median) <= .018 for value in shifts) < 4:
        return None
    return median


def continued_offset(origin, step):
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    return Pose(origin.x + c * step.x - s * step.y, origin.y + s * step.x + c * step.y,
                origin.yaw + step.yaw)
