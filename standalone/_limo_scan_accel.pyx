# cython: language_level=2
from libc.math cimport atan2, cos, fabs, hypot, sin, isnan, isinf
from libc.stdint cimport uint16_t
cimport numpy as cnp

cnp.import_array()

cdef inline bint finite(double x) nogil:
    return not isnan(x) and not isinf(x)


cdef struct _DepthPoint:
    double x, y, z, optical, distance
    Py_ssize_t column, row
    bint available


cdef void _keep_nearest(_DepthPoint *point, double x, double y, double z, double optical,
                        double distance, Py_ssize_t column, Py_ssize_t row) nogil:
    if not point.available or distance < point.distance:
        point.x, point.y, point.z = x, y, z
        point.optical, point.distance = optical, distance
        point.column, point.row = column, row
        point.available = True


cdef object _depth_point(_DepthPoint point):
    if not point.available:
        return None
    return {'x_m': point.x, 'y_m': point.y, 'z_m': point.z, 'optical_depth_m': point.optical,
            'planar_distance_m': point.distance, 'pixel_u': point.column, 'pixel_v': point.row}


def fast_depth_corridor(cnp.ndarray image, cnp.ndarray transform, double fx, double fy, double cx, double cy,
                        double xmin, double xmax, double half, double floor, double tolerance,
                        double zmin, double zmax, double dmin, double dmax, Py_ssize_t stride):
    """Sample the existing 16UC1 corridor once, with no Python calls per pixel."""
    cdef uint16_t *pixels
    cdef double *matrix
    cdef Py_ssize_t height, width
    cdef Py_ssize_t row, column
    cdef Py_ssize_t total = 0, valid = 0, ground_count = 0, obstacle_count = 0, outside = 0
    cdef double optical, ox, oy, x, y, z, height_above_floor, distance
    cdef _DepthPoint nearest_ground, nearest_obstacle
    cdef bint malformed = False
    if (image.ndim != 2 or transform.ndim != 2 or transform.shape[0] != 4 or transform.shape[1] != 4
            or image.dtype.num != cnp.NPY_UINT16 or not image.dtype.isnative
            or transform.dtype.num != cnp.NPY_DOUBLE or not transform.dtype.isnative
            or not image.flags.c_contiguous or not image.flags.aligned
            or not transform.flags.c_contiguous or not transform.flags.aligned
            or not 1 <= stride <= 1024):
        return None
    pixels = <uint16_t *>cnp.PyArray_DATA(image)
    matrix = <double *>cnp.PyArray_DATA(transform)
    height, width = image.shape[0], image.shape[1]
    if not 1 <= height <= 8192 or not 1 <= width <= 8192:
        return None
    nearest_ground.available = nearest_obstacle.available = False
    with nogil:
        for row from 0 <= row < height by stride:
            for column from 0 <= column < width by stride:
                total += 1
                optical = pixels[row * width + column] * .001
                if not dmin <= optical < dmax:
                    continue
                valid += 1
                ox = (column - cx) * optical / fx
                oy = (row - cy) * optical / fy
                x = matrix[0]*ox + matrix[1]*oy + matrix[2]*optical + matrix[3]
                y = matrix[4]*ox + matrix[5]*oy + matrix[6]*optical + matrix[7]
                z = matrix[8]*ox + matrix[9]*oy + matrix[10]*optical + matrix[11]
                height_above_floor = z - floor
                distance = hypot(x, y)
                if not (finite(x) and finite(y) and finite(z) and finite(height_above_floor) and finite(distance)):
                    malformed = True
                    break
                if not xmin <= x <= xmax or fabs(y) > half:
                    outside += 1
                elif fabs(height_above_floor) <= tolerance:
                    ground_count += 1
                    _keep_nearest(&nearest_ground, x, y, z, optical, distance, column, row)
                elif zmin <= height_above_floor <= zmax:
                    obstacle_count += 1
                    _keep_nearest(&nearest_obstacle, x, y, z, optical, distance, column, row)
            if malformed:
                break
    if malformed:
        return None
    return {'sample_count': total, 'valid_count': valid, 'invalid_count': total-valid, 'sample_stride': stride,
            'ground_count': ground_count, 'obstacle_count': obstacle_count, 'outside_corridor_count': outside,
            'unclassified_corridor_count': valid-outside-ground_count-obstacle_count,
            'nearest_ground': _depth_point(nearest_ground), 'nearest_obstacle': _depth_point(nearest_obstacle),
            'status': ('no_valid_depth' if not valid else 'observed' if valid > outside else 'no_corridor_samples')}


def fast_transform_records(records, pose):
    """Transform bounded fusion records, or use the checked Python path."""
    cdef double pose_x, pose_y, cosine, sine, sx, sy, hx, hy, ax, ay, bx, by
    cdef Py_ssize_t total = 0
    cdef list rays = []
    cdef list hits = []
    try:
        if (type(records) is not list or type(pose) is not tuple or len(pose) != 3
                or any(type(v) is not float for v in pose)):
            return None
        pose_x, pose_y = pose[0], pose[1]
        cosine, sine = cos(pose[2]), sin(pose[2])
        for record in records:
            if type(record) is not dict or type(record['rays']) is not list:
                return None
            total += len(record['rays'])
            if total > 1536:
                return None
        for record in records:
            for ray in record['rays']:
                if type(ray) is not dict:
                    return None
                start, end, stamp = ray['start_world'], ray['hit_world'], ray['stamp']
                if (type(start) is not tuple or len(start) != 2 or type(end) is not tuple or len(end) != 2
                        or type(stamp) is not float
                        or any(type(v) is not float for v in start + end)):
                    return None
                sx, sy = start[0] - pose_x, start[1] - pose_y
                hx, hy = end[0] - pose_x, end[1] - pose_y
                ax, ay = cosine * sx + sine * sy, -sine * sx + cosine * sy
                bx, by = cosine * hx + sine * hy, -sine * hx + cosine * hy
                if not finite(ax) or not finite(ay) or not finite(bx) or not finite(by):
                    return None
                hit = (float(bx), float(by))
                rays.append({'start_base': (float(ax), float(ay)), 'hit_base': hit, 'stamp': stamp})
                hits.append({'point_base': hit, 'stamp': stamp})
        return rays, hits
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None


def fast_newest_raw(ranges, angle_min_obj, increment_obj, lower_obj, upper_obj,
                    extrinsics, scan_pose, current_pose, half_pi_obj):
    """Project current raw rays, or return None for the checked scalar path."""
    cdef double angle_min, increment, lower, upper, half_pi
    cdef double theta, c, s, cc, cs, dx, dy, tx, ty, ex, ey, yaw
    cdef double start_x, start_y, value, angle, x, y, bx, by, fraction, crossing
    cdef Py_ssize_t index
    cdef list crossings = []
    try:
        if (type(ranges) not in (list, tuple) or type(extrinsics) is not tuple or len(extrinsics) != 6
                or type(scan_pose) is not tuple or len(scan_pose) != 3
                or type(current_pose) is not tuple or len(current_pose) != 3
                or any(type(v) is not float for v in
                       (angle_min_obj, increment_obj, lower_obj, upper_obj, half_pi_obj)
                       + extrinsics + scan_pose + current_pose)):
            return None
        angle_min, increment, lower, upper, half_pi = (angle_min_obj, increment_obj,
                                                        lower_obj, upper_obj, half_pi_obj)
        ex, ey, yaw = extrinsics[0], extrinsics[1], extrinsics[5]
        theta = scan_pose[2] - current_pose[2]
        c, s = cos(theta), sin(theta)
        cc, cs = cos(current_pose[2]), sin(current_pose[2])
        dx, dy = scan_pose[0] - current_pose[0], scan_pose[1] - current_pose[1]
        tx, ty = cc * dx + cs * dy, -cs * dx + cc * dy
        start_x, start_y = tx + c * ex - s * ey, ty + s * ex + c * ey
        for index in range(len(ranges)):
            item = ranges[index]
            if type(item) is not float and type(item) is not int:
                return None
            value = item
            if not finite(value) or not lower <= value < upper:
                continue
            angle = angle_min + index * increment + yaw
            if not finite(angle):
                return None
            angle = atan2(sin(angle), cos(angle))
            if fabs(fabs(angle) - half_pi) <= 1e-12:
                return None
            if angle < -half_pi or angle > half_pi:
                continue
            bx, by = ex + value * cos(angle), ey + value * sin(angle)
            x, y = tx + c * bx - s * by, ty + s * bx + c * by
            if not finite(x) or not finite(y):
                return None
            if (fabs(x + .20) <= 1e-12 or fabs(x - .65) <= 1e-12
                    or fabs(fabs(y) - .17) <= 1e-12 or fabs(start_x - .60) <= 1e-12
                    or fabs(x - .60) <= 1e-12):
                return None
            if -.20 <= x <= .65 and fabs(y) <= .17:
                return True, []
            if start_x <= .60 < x:
                fraction = (.60 - start_x) / (x - start_x)
                crossing = start_y + fraction * (y - start_y)
                if not finite(crossing) or fabs(fabs(crossing) - .15) <= 1e-12:
                    return None
                if -.15 <= crossing <= .15:
                    crossings.append(float(crossing))
        return False, crossings
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None


def fast_scan_points(ranges, double angle_min, double increment, double range_min,
                     double range_max, origin, double sensor_yaw, stamp, double half_pi):
    cdef double angle, value, yaw, cosine, sine, sx, sy, hx, hy, ox, oy
    cdef Py_ssize_t index, front = 0, valid = 0, nan = 0, zero = 0, other = 0
    cdef list rays = []
    cdef set indices = set()
    try:
        ox, oy = origin
        for index in range(len(ranges)):
            angle = angle_min + index * increment
            if fabs(fabs(angle) - half_pi) <= 1e-12:
                return None
            if angle < -half_pi or angle > half_pi:
                continue
            front += 1
            item = ranges[index]
            if item is None:
                other += 1
                continue
            if type(item) is not float and type(item) is not int:
                return None
            value = item
            if isnan(value):
                nan += 1
                continue
            if value == 0:
                zero += 1
                continue
            if not finite(value) or not 0 < value < range_max or value < range_min:
                other += 1
                continue
            yaw = sensor_yaw + angle
            cosine, sine = cos(yaw), sin(yaw)
            sx, sy = ox + range_min * cosine, oy + range_min * sine
            hx, hy = ox + value * cosine, oy + value * sine
            if not finite(sx) or not finite(sy) or not finite(hx) or not finite(hy):
                return None
            valid += 1
            indices.add(index)
            rays.append({'start_world': (float(sx), float(sy)),
                         'hit_world': (float(hx), float(hy)), 'stamp': stamp, 'index': index})
        return rays, indices, front, valid, nan, zero, other
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None


def fast_raw_veto(ranges, double angle_min, double increment, double range_min, double range_max,
                  extrinsics, scan_pose, pose, double duration, double speed, double half_pi):
    """Return a checked raw-ray summary, or None for the Python fallback."""
    cdef double value, angle, bearing, bx, by, wx, wy, dx, dy, x, y
    cdef double scan_c, scan_s, base_c, base_s, margin, side_margin
    cdef double laser_x, laser_y, laser_yaw, scan_x, scan_y, pose_x, pose_y
    cdef Py_ssize_t index, front_count = 0, valid_count = 0
    cdef bint measured, blocked = False
    try:
        laser_x, laser_y, laser_yaw = extrinsics
        scan_x, scan_y = scan_pose[0], scan_pose[1]
        pose_x, pose_y = pose[0], pose[1]
        scan_c, scan_s = cos(scan_pose[2]), sin(scan_pose[2])
        base_c, base_s = cos(pose[2]), sin(pose[2])
        margin = speed * duration
        side_margin = margin + .6 * .02 * duration
        for index in range(len(ranges)):
            item = ranges[index]
            measured = False
            if item is not None:
                if type(item) is not float and type(item) is not int:
                    return None
                value = item
                measured = finite(value) and value > 0 and range_min <= value < range_max
            angle = angle_min + index * increment
            if fabs(fabs(angle) - half_pi) <= 1e-12:
                return None
            if angle < -half_pi or angle > half_pi:
                continue
            front_count += 1
            if not measured:
                continue
            valid_count += 1
            bearing = angle + laser_yaw
            bx = laser_x + value * cos(bearing)
            by = laser_y + value * sin(bearing)
            wx = scan_x + scan_c * bx - scan_s * by
            wy = scan_y + scan_s * bx + scan_c * by
            dx, dy = wx - pose_x, wy - pose_y
            x, y = base_c * dx + base_s * dy, -base_s * dx + base_c * dy
            if not finite(x) or not finite(y):
                return None
            if (fabs(x + margin) <= 1e-12 or fabs(x - (.6 + margin)) <= 1e-12
                    or fabs(fabs(y) - (.15 + side_margin)) <= 1e-12):
                return None
            if -margin <= x <= .6 + margin and fabs(y) <= .15 + side_margin:
                blocked = True
        return bool(blocked), front_count, valid_count
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None


cdef double largest_gap(list values, double half_width):
    cdef Py_ssize_t index
    cdef double gap, current
    if not values:
        return 2 * half_width
    values.sort()
    gap = max(values[0] + half_width, half_width - values[-1])
    for index in range(1, len(values)):
        current = values[index] - values[index - 1]
        if current > gap:
            gap = current
    return gap


def fast_front_expiry(rays, scan_at_obj, snapshot_at_obj):
    """Find expiry for a previously validated scene, or use scalar fallback."""
    cdef double stamp, sx, sy, hx, hy, y, gap, deadline, expires, scan_at, snapshot_at
    cdef Py_ssize_t index
    cdef list crossings = []
    cdef list latest = []
    cdef list remaining
    cdef set stamps = set()
    try:
        if (type(rays) is not list or not 32 <= len(rays) <= 2048
                or type(scan_at_obj) is not float or type(snapshot_at_obj) is not float):
            return None
        scan_at, snapshot_at = scan_at_obj, snapshot_at_obj
        for index in range(len(rays)):
            ray = rays[index]
            if type(ray) is not dict:
                return None
            stamp_obj, start, end = ray['stamp'], ray['start_base'], ray['hit_base']
            if (type(stamp_obj) is not float or type(start) is not tuple or len(start) != 2
                    or type(end) is not tuple or len(end) != 2
                    or type(start[0]) is not float or type(start[1]) is not float
                    or type(end[0]) is not float or type(end[1]) is not float):
                return None
            stamp, sx, sy, hx, hy = stamp_obj, start[0], start[1], end[0], end[1]
            if not finite(stamp) or not 0 <= snapshot_at - stamp <= .65 or not 0 <= scan_at - stamp <= .4:
                continue
            if fabs(sx - .60) <= 1e-12 or fabs(hx - .60) <= 1e-12:
                return None
            if sx <= .60 < hx:
                y = sy + (.60 - sx) / (hx - sx) * (hy - sy)
                if not finite(y) or fabs(fabs(y) - .15) <= 1e-12:
                    return None
                if -.15 <= y <= .15:
                    crossings.append((float(y), stamp_obj))
                    stamps.add(stamp_obj)
                    if len(stamps) > 8:
                        return None
                    if stamp == scan_at:
                        latest.append(float(y))
        if not crossings or not latest:
            return None
        gap = largest_gap([item[0] for item in crossings], .15)
        if fabs(gap - (.035 + 1e-12)) <= 1e-11 or gap > .035 + 1e-12:
            return None
        gap = largest_gap(latest, .15)
        if fabs(gap - (.07 + 1e-12)) <= 1e-11 or gap > .07 + 1e-12:
            return None
        deadline = scan_at + .5
        for stamp_obj in sorted(stamps):
            stamp = stamp_obj
            expires = stamp + .65
            if expires >= deadline:
                break
            remaining = [item[0] for item in crossings if item[1] > stamp_obj]
            gap = largest_gap(remaining, .15)
            if fabs(gap - (.035 - 1e-12)) <= 1e-11:
                return None
            if gap > .035 - 1e-12:
                return float(expires)
        return float(deadline)
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None


def fast_assess_rays(rays, hits, double scan_stamp, long latest_valid, double now,
                     double line_x, double half_width, double max_age, double max_gap,
                     double history_age_limit, double history_span_limit):
    cdef double sx, sy, hx, hy, stamp, age, y, all_gap, latest_gap, obstacle_half
    cdef Py_ssize_t index, count, latest_count = 0
    cdef bint blocked = False
    cdef list crossings = []
    cdef list latest_crossings = []
    try:
        if len(rays) < 32 or len(rays) != len(hits):
            return None
        count = len(rays)
        obstacle_half = max(half_width, .15)
        for index in range(count):
            ray, hit = rays[index], hits[index]
            if type(ray) is not dict or type(hit) is not dict:
                return None
            start, end, stamp_obj = ray['start_base'], ray['hit_base'], ray['stamp']
            if (type(start) is not tuple or len(start) != 2 or type(end) is not tuple or len(end) != 2
                    or hit['point_base'] is not end or hit['stamp'] is not stamp_obj
                    or type(stamp_obj) is not float
                    or type(start[0]) is not float or type(start[1]) is not float
                    or type(end[0]) is not float or type(end[1]) is not float):
                return None
            sx, sy, hx, hy, stamp = start[0], start[1], end[0], end[1], stamp_obj
            if (not finite(sx) or not finite(sy) or not finite(hx) or not finite(hy)
                    or not finite(stamp) or stamp < 0 or stamp > scan_stamp):
                return None
            if (fabs(hx + .18) <= 1e-12 or fabs(hx - line_x) <= 1e-12
                    or fabs(fabs(hy) - obstacle_half) <= 1e-12 or fabs(sx - line_x) <= 1e-12):
                return None
            if -.18 <= hx <= line_x and -obstacle_half <= hy <= obstacle_half:
                blocked = True
            if stamp == scan_stamp:
                latest_count += 1
            age = now - stamp
            if (0 <= age <= min(history_age_limit, max_age)
                    and scan_stamp - stamp <= history_span_limit and sx <= line_x < hx):
                y = sy + ((line_x - sx) / (hx - sx)) * (hy - sy)
                if not finite(y) or fabs(fabs(y) - half_width) <= 1e-12:
                    return None
                if -half_width <= y <= half_width:
                    crossings.append(float(y))
                    if stamp == scan_stamp:
                        latest_crossings.append(float(y))
        result = {'candidate': False, 'reason': 'invalid_evidence', 'geometric_clearance_verified': False}
        if blocked:
            result['reason'] = 'forward_obstacle'
            return result
        if latest_count != latest_valid:
            return result
        all_gap = largest_gap(crossings, half_width)
        latest_gap = largest_gap(latest_crossings, half_width)
        if fabs(all_gap - max_gap) <= 1e-12 or fabs(latest_gap - 2 * max_gap) <= 1e-12:
            return None
        result['samples'] = len(crossings)
        result['max_sample_gap_m'] = all_gap
        result['latest_max_sample_gap_m'] = latest_gap
        if not crossings or not latest_crossings or all_gap > max_gap or latest_gap > 2 * max_gap:
            result['reason'] = 'forward_coverage_incomplete'
        else:
            result.update(candidate=True, reason='sampled_forward_corridor')
        return result
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None


cdef double _crossing_gap(list values):
    cdef double gap, current
    cdef Py_ssize_t index
    if not values:
        return .3
    values.sort()
    gap = max(values[0] + .15, .15 - values[-1])
    for index in range(1, len(values)):
        current = values[index] - values[index - 1]
        if current > gap:
            gap = current
    return gap


def fast_newest_history(rays, hits, crossings, double now, double scan_at, scene_pose, current_pose,
                        future_at=None):
    cdef double theta, c, s, cc, cs, dx, dy, tx, ty
    cdef double sx, sy, hx, hy, stamp, ax, ay, bx, by, age, y, gap, future = 0.
    cdef Py_ssize_t index, count
    cdef bint blocked = False
    cdef bint with_future = future_at is not None
    cdef list values, later = []
    try:
        if len(rays) < 32 or len(rays) != len(hits):
            return None
        theta = scene_pose[2] - current_pose[2]
        c, s = cos(theta), sin(theta)
        cc, cs = cos(current_pose[2]), sin(current_pose[2])
        dx, dy = scene_pose[0] - current_pose[0], scene_pose[1] - current_pose[1]
        tx, ty = cc * dx + cs * dy, -cs * dx + cc * dy
        values = list(crossings)
        if with_future:
            future = future_at
            if not finite(future) or not now <= future <= scan_at + .3:
                return None
            later = list(crossings)
        for item in values:
            if type(item) is not float or not finite(item):
                return None
        count = len(rays)
        for index in range(count):
            ray, hit = rays[index], hits[index]
            if type(ray) is not dict or type(hit) is not dict:
                return None
            start, end, stamp_obj = ray['start_base'], ray['hit_base'], ray['stamp']
            if (type(start) is not tuple or len(start) != 2 or type(end) is not tuple or len(end) != 2
                    or hit['point_base'] is not end or hit['stamp'] is not stamp_obj
                    or type(stamp_obj) is not float
                    or type(start[0]) is not float or type(start[1]) is not float
                    or type(end[0]) is not float or type(end[1]) is not float):
                return None
            sx, sy, hx, hy, stamp = start[0], start[1], end[0], end[1], stamp_obj
            if not finite(sx) or not finite(sy) or not finite(hx) or not finite(hy) or not finite(stamp):
                return None
            if stamp < 0 or stamp > scan_at:
                return None
            ax, ay = tx + c * sx - s * sy, ty + s * sx + c * sy
            bx, by = tx + c * hx - s * hy, ty + s * hx + c * hy
            if not finite(ax) or not finite(ay) or not finite(bx) or not finite(by):
                return None
            # Near a geometric threshold, use the original NumPy/scalar path
            # rather than relying on platform-specific floating-point ties.
            if (fabs(bx + .20) <= 1e-12 or fabs(bx - .65) <= 1e-12
                    or fabs(fabs(by) - .17) <= 1e-12 or fabs(ax - .60) <= 1e-12
                    or fabs(bx - .60) <= 1e-12):
                return None
            if -0.20 <= bx <= 0.65 and -0.17 <= by <= 0.17:
                blocked = True
            age = now - stamp
            if 0 <= age <= 0.65 and scan_at - stamp <= 0.4 and ax <= 0.60 < bx:
                y = ay + ((0.60 - ax) / (bx - ax)) * (by - ay)
                if not finite(y):
                    return None
                if fabs(fabs(y) - .15) <= 1e-12:
                    return None
                if -0.15 <= y <= 0.15:
                    values.append(float(y))
                    if with_future and future - stamp <= .65:
                        later.append(float(y))
        if blocked:
            if with_future:
                return False, False, True
            return False
        if not values:
            return None
        gap = _crossing_gap(values)
        if fabs(gap - .035) <= 1e-12:
            return None
        if with_future:
            age = _crossing_gap(later)
            if fabs(age - .035) <= 1e-12:
                return None
            return gap <= .035, age <= .035, False
        return gap <= .035
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
