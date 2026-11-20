"""Experimental synthetic geometry; never authorizes or commands motion.

The grid must represent whole-cell free/occupied knowledge, not laser ray samples.
Unknown current footprint cells are not treated as free; they are merely excluded
from the *new* region occupied by a forward translation. This 2D model does not
address vertical obstacles, braking, sensor calibration, or real-world clearance.
"""

import math
import sys

try:
    NUMBER_TYPES = (int, long, float)  # noqa: F821 - Python 2 ROS worker
    INTEGER_TYPES = (int, long)  # noqa: F821
except NameError:
    NUMBER_TYPES = (int, float)
    INTEGER_TYPES = (int,)


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, NUMBER_TYPES):
        return False
    try:
        return not math.isnan(value) and not math.isinf(value)
    except (OverflowError, ValueError):
        return False


def _rectangle(pose, minimum_x, maximum_x, half_width):
    x, y, yaw = pose
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [
        (x + px * cosine - py * sine, y + px * sine + py * cosine)
        for px, py in (
            (minimum_x, -half_width),
            (maximum_x, -half_width),
            (maximum_x, half_width),
            (minimum_x, half_width),
        )
    ]


def _overlaps_cell(polygon, left, bottom, size, yaw, tolerance):
    """Conservative separating-axis test; shared edges and corners overlap."""
    center_x, center_y = left + size / 2.0, bottom + size / 2.0
    cosine, sine = math.cos(yaw), math.sin(yaw)
    for ax, ay in ((1.0, 0.0), (0.0, 1.0), (cosine, sine), (-sine, cosine)):
        values = [(point[0] - center_x) * ax + (point[1] - center_y) * ay for point in polygon]
        radius = size / 2.0 * (abs(ax) + abs(ay))
        if max(values) < -radius - tolerance or min(values) > radius + tolerance:
            return False
    return True


def check_forward_sweep(grid, pose, footprint, distance, now, max_age=2.0):
    """Evaluate a synthetic rectangular forward sweep using whole-cell evidence.

    footprint is (rear extent, front extent, half width), in metres. The grid is
    expressed in the same world frame as pose. A result is never executable.
    """
    result = {"status": "invalid", "executable": False}
    try:
        res, origin, cells, stamp = grid["resolution"], grid["origin"], grid["cells"], grid["stamp"]
        if len(origin) != 2 or len(pose) != 3 or len(footprint) != 3:
            return result
        values = list(origin) + list(pose) + list(footprint) + [res, stamp, distance, now, max_age]
        if not all(_finite(value) for value in values):
            return result
        if (
            not 0.01 <= res <= 1.0
            or not 0 < distance <= 0.25
            or not 0 < max_age <= 10
            or any(not 0 < value <= 1 for value in footprint)
        ):
            return result
        if not 0 <= now - stamp <= max_age:
            result["status"] = "stale"
            return result
        if (
            not isinstance(cells, list)
            or not cells
            or not isinstance(cells[0], list)
            or not cells[0]
        ):
            return result
        height, width = len(cells), len(cells[0])
        if height > 400 or width > 400:
            return result
        if any(
            not isinstance(row, list)
            or len(row) != width
            or any(
                isinstance(cell, bool)
                or not isinstance(cell, INTEGER_TYPES)
                or not -1 <= cell <= 100
                for cell in row
            )
            for row in cells
        ):
            return result

        coord_scale = max(
            1.0,
            abs(pose[0]),
            abs(pose[1]),
            abs(origin[0]),
            abs(origin[1]),
            abs(origin[0] + width * res),
            abs(origin[1] + height * res),
        )
        tolerance = max(1e-9, res * 1e-8, coord_scale * sys.float_info.epsilon * 16)
        if tolerance >= res * 0.01:
            return result
        rear, front, half_width = footprint
        body = _rectangle(pose, -rear, front, half_width)
        new_space = _rectangle(pose, front, front + distance, half_width)
        points = body + new_space
        ox, oy = origin
        min_col = int(math.floor((min(p[0] for p in points) - ox - tolerance) / res))
        max_col = int(math.floor((max(p[0] for p in points) - ox + tolerance) / res))
        min_row = int(math.floor((min(p[1] for p in points) - oy - tolerance) / res))
        max_row = int(math.floor((max(p[1] for p in points) - oy + tolerance) / res))
        unknown = False
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                left, bottom = ox + col * res, oy + row * res
                inside = 0 <= row < height and 0 <= col < width
                value = cells[row][col] if inside else -1
                hits_new = _overlaps_cell(new_space, left, bottom, res, pose[2], tolerance)
                if value > 0 and (
                    hits_new or _overlaps_cell(body, left, bottom, res, pose[2], tolerance)
                ):
                    result["status"] = "blocked"
                    return result
                if hits_new and value == -1:
                    unknown = True
        result["status"] = "unknown" if unknown else "clear_in_simulation"
        return result
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return result
