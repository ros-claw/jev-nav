"""Validate a short straight retreat within a previously driven static-room trail."""
from __future__ import division

import math

from navigation import Pose, _relative

TRAIL_MARGIN_M = .04
MAX_LATERAL_M = .006
MAX_HEADING_RAD = .025
LIDAR_X = .105


def straight_trail(trace, reference, distance):
    trajectory = trace['trajectory']
    if (trace['mode'] != 'supervised_static_room' or trace['relay_exited'] is not True
            or trajectory['clock'] != 'host_monotonic_receipt' or not .10 <= distance <= .35):
        raise ValueError('retrace needs a completed supervised static-room recording')
    rows = trajectory['odom']
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
