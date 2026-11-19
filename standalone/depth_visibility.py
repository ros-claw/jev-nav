"""Classify explicit base-frame sample centers against one depth image.

Diagnostics only: a sampled center does not establish coverage of any voxel,
area between rays, swept volume, near-field blind spot, or permission to move.
The caller supplies a correctly registered, rectified image, matching camera
intrinsics and a calibrated base-from-optical rigid transform; image/TF age,
synchronization, distortion, sensor noise, and motion are not checked here.

Optical coordinates use x right, y down and z forward; depth is optical-axis z,
not Euclidean range. A nearest-pixel lookup is an approximation and cannot
confirm visibility at every point in the pixel footprint. ``sampled_free``
describes only a supplied center whose corresponding valid depth is beyond it
by the configured margin. No output is executable or proves 3-D clearance.
"""

from __future__ import division  # noqa: UP010 - ROS Melodic Python 2

import math

import numpy as np

try:
    INTEGER_TYPES = (int, long)  # Python 2 ROS worker
except NameError:
    INTEGER_TYPES = (int,)
NUMBER_TYPES = INTEGER_TYPES + (float,)


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _empty(status):
    return {"status": status, "executable": False, "geometric_clearance_verified": False,
            "unobserved_space": True, "voxel_count": 0, "voxels": [],
            "counts": {"unknown": 0, "sampled_free": 0, "measured": 0, "occluded": 0}}


def diagnose_depth_visibility(depth_array, encoding, intrinsics, base_from_optical,
                              centers_base, config):
    """Classify explicit 3-D centers, returning diagnostic evidence only.

    ``depth_array`` must be a 2-D numpy array: uint16 millimeters for 16UC1 or
    float32 meters for 32FC1 (byte order and strided views are handled by numpy).
    Intrinsics provide width, height, fx, fy, cx, cy, rectified=True. The 4x4
    ``base_from_optical`` maps optical coordinates into base coordinates; each
    center is (x forward, y left, z up) in that same base frame and in meters.

    Config requires min_depth_m, max_depth_m, near_margin_m,
    surface_tolerance_m, and free_margin_m. A target or observation at or below
    min_depth_m + near_margin_m, or at or beyond max_depth_m, is unknown. A hit
    within surface_tolerance_m of a target is measured; a target further behind
    is occluded. A target in front is sampled_free only if the depth difference
    is strictly greater than free_margin_m; the intervening band is unknown.

    Output contains ordered ``voxels`` with center_base_m, pixel_uv,
    target_depth_m, observed_depth_m, status and reason, plus per-status counts.
    ``observed_depth_m`` is a finite raw reading in meters or None, and can be
    outside the permitted range. Invalid input returns status=invalid_input and
    no voxel results. Status=ok means the input was accepted, even if every
    center is unknown. No timestamp or sensor/transform freshness is implied.
    """
    invalid = _empty("invalid_input")
    try:
        if (not isinstance(depth_array, np.ndarray) or depth_array.ndim != 2
                or not isinstance(intrinsics, dict) or not isinstance(config, dict)):
            return invalid
        if encoding == "16UC1":
            if depth_array.dtype.kind != "u" or depth_array.dtype.itemsize != 2:
                return invalid
            depth_scale = 0.001
        elif encoding == "32FC1":
            if depth_array.dtype.kind != "f" or depth_array.dtype.itemsize != 4:
                return invalid
            depth_scale = 1.0
        else:
            return invalid
        height, width = depth_array.shape
        iw, ih = intrinsics.get("width"), intrinsics.get("height")
        if (isinstance(iw, bool) or not isinstance(iw, INTEGER_TYPES)
                or isinstance(ih, bool) or not isinstance(ih, INTEGER_TYPES)
                or not 1 <= width <= 8192 or not 1 <= height <= 8192
                or iw != width or ih != height or intrinsics.get("rectified") is not True):
            return invalid
        fx, fy, cx, cy = [intrinsics[name] for name in ("fx", "fy", "cx", "cy")]
        if not all(_finite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            return invalid
        names = ("min_depth_m", "max_depth_m", "near_margin_m", "surface_tolerance_m",
                 "free_margin_m")
        if not all(_finite(config[name]) for name in names):
            return invalid
        minimum, maximum, near, surface, free = [config[name] for name in names]
        cutoff = minimum + near
        if (not _finite(cutoff) or not 0 < minimum < cutoff < maximum
                or not 0 < surface < free):
            return invalid
        matrix = np.asarray(base_from_optical)
        if matrix.shape != (4, 4) or matrix.dtype.kind not in ("f", "i", "u"):
            return invalid
        matrix = matrix.astype(np.float64)
        if not np.all(np.isfinite(matrix)):
            return invalid
        rotation = matrix[:3, :3]
        with np.errstate(over="ignore", invalid="ignore"):
            if (not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-9)
                    or not np.allclose(np.dot(rotation.T, rotation), np.eye(3), rtol=0.0, atol=1e-6)
                    or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6):
                return invalid
        centers = np.asarray(centers_base)
        if centers.ndim == 1 and centers.size == 0:
            centers = centers.reshape((0, 3))
        if (centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] > 100000
                or centers.dtype.kind not in ("f", "i", "u")):
            return invalid
        centers = centers.astype(np.float64)
        if not np.all(np.isfinite(centers)):
            return invalid
        if not centers.shape[0]:
            return _empty("no_centers")
        # p_base = R * p_optical + t, hence p_optical = R.T * (p_base - t).
        with np.errstate(over="ignore", invalid="ignore"):
            optical = np.dot(centers - matrix[:3, 3], rotation)
        result = _empty("ok")
        result["voxel_count"] = int(centers.shape[0])
        for index in range(centers.shape[0]):
            center = tuple(float(value) for value in centers[index])
            ox, oy, oz = [float(value) for value in optical[index]]
            voxel = {"center_base_m": center, "status": "unknown", "reason": None,
                     "pixel_uv": None, "target_depth_m": oz if _finite(oz) else None,
                     "observed_depth_m": None}
            if center[0] <= 0:
                voxel["reason"] = "not_forward_base"
            elif not all(_finite(value) for value in (ox, oy, oz)):
                voxel["reason"] = "invalid_projection"
            elif oz <= 0:
                voxel["reason"] = "behind_camera"
            elif oz <= cutoff:
                voxel["reason"] = "target_near_limit"
            elif oz >= maximum:
                voxel["reason"] = "target_out_of_range"
            else:
                u, v = fx * ox / oz + cx, fy * oy / oz + cy
                if not _finite(u) or not _finite(v):
                    voxel["reason"] = "invalid_projection"
                elif not (0 <= u <= width - 1 and 0 <= v <= height - 1):
                    voxel["reason"] = "out_of_frame"
                else:
                    column, row = int(math.floor(u + 0.5)), int(math.floor(v + 0.5))  # noqa: RUF046
                    voxel["pixel_uv"] = (column, row)
                    depth = float(depth_array[row, column]) * depth_scale
                    if not _finite(depth):
                        voxel["reason"] = "invalid_depth"
                    else:
                        voxel["observed_depth_m"] = depth
                        if depth <= 0:
                            voxel["reason"] = "invalid_depth"
                        elif depth <= cutoff:
                            voxel["reason"] = "depth_near_limit"
                        elif depth >= maximum:
                            voxel["reason"] = "depth_out_of_range"
                        else:
                            delta = depth - oz
                            if abs(delta) <= surface:
                                voxel["status"], voxel["reason"] = "measured", "near_surface"
                            elif delta < -surface:
                                voxel["status"], voxel["reason"] = "occluded", "nearer_surface"
                            elif delta > free:
                                voxel["status"], voxel["reason"] = "sampled_free", "depth_beyond"
                            else:
                                voxel["reason"] = "surface_margin"
            result["voxels"].append(voxel)
            result["counts"][voxel["status"]] += 1
        return result
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError):
        return invalid
