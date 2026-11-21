"""Identify isolated returns in the extra lateral margin of a forward scan."""

from __future__ import division

import math

from processed_evidence import _number


def isolated_margin_bins(scan, previous, extrinsics):
    """Return bins to mark unknown, using two scans at nearly the same pose.

    The caller verifies the source times, consecutive sequence and odometry.
    The center corridor and persistent close hits retain all their evidence.
    """
    layout = ('angle_min', 'angle_increment', 'range_min', 'range_max')
    try:
        angle_min, increment, minimum, maximum = (getattr(scan, key) for key in layout)
        if (previous is None or len(scan.ranges) != len(previous.ranges)
                or not 1 <= len(scan.ranges) <= 8192
                or not all(_number(v) for v in (angle_min, increment, minimum, maximum))
                or not .002 <= abs(increment) <= .02 or not 0 < minimum < maximum <= 100
                or len(extrinsics) != 6 or not all(_number(v) for v in extrinsics)
                or any(getattr(scan, key) != getattr(previous, key) for key in layout)):
            return ()
        radius = int(.055 / abs(increment))
        rejected = []
        for index, distance in enumerate(scan.ranges):
            if not _number(distance) or not minimum <= distance < maximum or distance > .30:
                continue
            angle = angle_min + index * increment + extrinsics[5]
            x = extrinsics[0] + distance * math.cos(angle)
            y = extrinsics[1] + distance * math.sin(angle)
            if not (-math.pi / 2 <= angle <= math.pi / 2 and 0 <= x <= .25 and .163 < abs(y) <= .17):
                continue
            lo, hi = max(0, index - radius), min(len(scan.ranges), index + radius + 1)
            nearby = [(i, scan.ranges[i]) for i in range(lo, hi) if i != index]
            before = [(i, previous.ranges[i]) for i in range(lo, hi)]
            nearby = [(i, value) for i, value in nearby if _number(value) and minimum <= value < maximum]
            before = [(i, value) for i, value in before if _number(value) and minimum <= value < maximum]
            if any(value < distance + .35 for _, value in nearby + before):
                continue
            if any(abs(i - j) <= 2 and abs(value - prior) <= .25
                   for i, value in nearby for j, prior in before):
                rejected.append(index)
        return tuple(rejected)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return ()
