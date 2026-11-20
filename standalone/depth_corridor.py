"""Project depth samples into base coordinates for diagnostics only.

Depth is optical-axis Z, not Euclidean range. The input must be a rectified
depth image matching the supplied intrinsics and a calibrated base-from-optical
rigid transform. Base coordinates use x forward, y left, and z up. No
calibration, clock, frame, floor-planarity, swept-volume, occlusion, or
between-pixel coverage is verified here. No result grants permission to move.
"""

from __future__ import division

import math

import numpy as np

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - ROS Melodic Python 2
except NameError:
    INTEGER_TYPES = (int,)
NUMBER_TYPES = INTEGER_TYPES + (float,)


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _count(mask):
    return int(np.count_nonzero(mask))


def _nearest(mask, x, y, z, optical, pixel_u, pixel_v):
    indices = np.flatnonzero(mask)
    if not indices.size:
        return None
    planar = np.hypot(x[indices], y[indices])
    index = int(indices[int(np.argmin(planar))])
    return {"x_m": float(x[index]), "y_m": float(y[index]), "z_m": float(z[index]),
            "optical_depth_m": float(optical[index]),
            "planar_distance_m": float(np.hypot(x[index], y[index])),
            "pixel_u": int(pixel_u[index]), "pixel_v": int(pixel_v[index])}


def diagnose_depth_corridor(depth_array, encoding, intrinsics, base_from_optical, config):
    """Classify sampled depth endpoints inside a configurable forward corridor.

    Intrinsics contain width, height, fx, fy, cx, cy, and rectified=True.
    base_from_optical is a proper homogeneous 4x4 rigid transform. Config needs
    forward_min_m, forward_max_m, half_width_m, ground_z_m, ground_tolerance_m,
    obstacle_min_height_m, obstacle_max_height_m, min_depth_m, max_depth_m and
    sample_stride. Heights are relative to ground_z_m; obstacle height must be
    greater than the ground tolerance. The depth maximum is treated as unknown.

    The returned counts describe sampled endpoints only. Invalid depth, areas
    outside the image, and space between pixels are not free-space evidence.
    """
    result = {"status": "invalid_input", "executable": False,
              "geometric_clearance_verified": False, "unobserved_space": True}
    try:
        if (not isinstance(depth_array, np.ndarray) or depth_array.ndim != 2
                or not isinstance(intrinsics, dict) or not isinstance(config, dict)):
            return result
        if encoding == "16UC1":
            if depth_array.dtype.kind != "u" or depth_array.dtype.itemsize != 2:
                return result
        elif encoding == "32FC1":
            if depth_array.dtype.kind != "f" or depth_array.dtype.itemsize != 4:
                return result
        else:
            return result
        height, width = depth_array.shape
        iw, ih = intrinsics.get("width"), intrinsics.get("height")
        if (isinstance(iw, bool) or not isinstance(iw, INTEGER_TYPES)
                or isinstance(ih, bool) or not isinstance(ih, INTEGER_TYPES)
                or not 1 <= width <= 8192 or not 1 <= height <= 8192
                or iw != width or ih != height or intrinsics.get("rectified") is not True):
            return result
        fx, fy, cx, cy = [intrinsics[name] for name in ("fx", "fy", "cx", "cy")]
        if not all(_finite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            return result
        names = ("forward_min_m", "forward_max_m", "half_width_m", "ground_z_m",
                 "ground_tolerance_m", "obstacle_min_height_m", "obstacle_max_height_m",
                 "min_depth_m", "max_depth_m")
        if not all(_finite(config[name]) for name in names):
            return result
        xmin, xmax, half = (config["forward_min_m"], config["forward_max_m"],
                            config["half_width_m"])
        floor, tolerance = config["ground_z_m"], config["ground_tolerance_m"]
        zmin, zmax = config["obstacle_min_height_m"], config["obstacle_max_height_m"]
        dmin, dmax, stride = config["min_depth_m"], config["max_depth_m"], config["sample_stride"]
        if (not 0 <= xmin < xmax or half <= 0 or tolerance < 0
                or not tolerance < zmin <= zmax or not 0 < dmin < dmax
                or isinstance(stride, bool) or not isinstance(stride, INTEGER_TYPES)
                or not 1 <= stride <= 1024):
            return result
        matrix = np.asarray(base_from_optical)
        if matrix.shape != (4, 4) or matrix.dtype.kind not in ("f", "i", "u"):
            return result
        matrix = matrix.astype(np.float64)
        if not np.all(np.isfinite(matrix)):
            return result
        rotation = matrix[:3, :3]
        if (not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-9)
                or not np.allclose(np.dot(rotation.T, rotation), np.eye(3), rtol=0.0, atol=1e-6)
                or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6):
            return result
        sampled = depth_array[::stride, ::stride]
        if not 1 <= sampled.size <= 2000000:
            return result
        values = sampled.astype(np.float64)
        if encoding == "16UC1":
            values *= 0.001
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(values) & (values >= dmin) & (values < dmax)
        sample_count = int(values.size)
        valid_count = _count(valid)
        result.update({"sample_count": sample_count, "valid_count": valid_count,
                       "invalid_count": sample_count - valid_count,
                       "sample_stride": stride, "ground_count": 0, "obstacle_count": 0,
                       "unclassified_corridor_count": 0, "outside_corridor_count": 0,
                       "nearest_ground": None, "nearest_obstacle": None})
        if not valid_count:
            result["status"] = "no_valid_depth"
            return result
        rows, columns = np.nonzero(valid)
        pixel_u, pixel_v = columns * stride, rows * stride
        optical = values[valid]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ox = (pixel_u.astype(np.float64) - cx) * optical / fx
            oy = (pixel_v.astype(np.float64) - cy) * optical / fy
            x = rotation[0, 0] * ox + rotation[0, 1] * oy + rotation[0, 2] * optical + matrix[0, 3]
            y = rotation[1, 0] * ox + rotation[1, 1] * oy + rotation[1, 2] * optical + matrix[1, 3]
            z = rotation[2, 0] * ox + rotation[2, 1] * oy + rotation[2, 2] * optical + matrix[2, 3]
            height_above_floor = z - floor
            planar = np.hypot(x, y)
        if not all(np.all(np.isfinite(component)) for component in (x, y, z, height_above_floor, planar)):
            return {"status": "invalid_input", "executable": False,
                    "geometric_clearance_verified": False, "unobserved_space": True}
        corridor = (x >= xmin) & (x <= xmax) & (np.abs(y) <= half)
        ground = corridor & (np.abs(height_above_floor) <= tolerance)
        obstacle = corridor & (height_above_floor >= zmin) & (height_above_floor <= zmax)
        unclassified = corridor & ~ground & ~obstacle
        result.update({"ground_count": _count(ground), "obstacle_count": _count(obstacle),
                       "unclassified_corridor_count": _count(unclassified),
                       "outside_corridor_count": _count(~corridor),
                       "nearest_ground": _nearest(ground, x, y, z, optical, pixel_u, pixel_v),
                       "nearest_obstacle": _nearest(obstacle, x, y, z, optical, pixel_u, pixel_v),
                       "status": "observed" if np.any(corridor) else "no_corridor_samples"})
        return result
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError):
        return {"status": "invalid_input", "executable": False,
                "geometric_clearance_verified": False, "unobserved_space": True}
