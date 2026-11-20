"""Offline, diagnostic-only depth and lidar ray intersections with a section.

Each sample is a zero-width center ray, never an interval, voxel, swept-volume
proof, motion authorization, or a substitute for collision/unknown checks.
Different heights remain distinct. Finite samples cannot establish what lies
between rays; consequently ``safe_merge_available`` is always False.

The caller must supply a rectified image and intrinsics for that exact optical
frame, an accurately calibrated rigid base-reference-from-optical transform,
and lidar ray endpoints already registered in the same base reference pose.
The transform must include the camera extrinsic AND any ego-motion between the
image capture and reference times. Timestamps and frame labels are checked,
but calibration, odometry accuracy, rolling-shutter timing, clock epochs,
sensor noise, object motion, pixel/beam footprint and image immutability cannot
be verified here. Inputs must remain unchanged during the call.

Optical depth is optical-axis Z (meters or uint16 millimeters), not Euclidean
range. Occluded, invalid, near-limit, out-of-image and unrequested regions
remain unknown. Only explicitly requested pixels and latest-frame lidar rays
are inspected; no historical obstacle hit may be discarded by a caller.
"""
from __future__ import division

import math
import numbers
import re

import numpy as np

_FRAME = re.compile(r"\A[A-Za-z][A-Za-z0-9_/]*\Z")
_MAX_ITEMS = 100000


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, numbers.Real)
                and not math.isnan(value) and not math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _frame(value):
    return (isinstance(value, (str, type(u""))) and 0 < len(value) <= 255  # noqa: UP003, UP025 - Python 2 unicode
            and _FRAME.match(value) is not None and "//" not in value and not value.endswith("/"))


def _empty(status, reason):
    return {"status": status, "reason": reason, "candidate": False, "executable": False,
            "geometric_clearance_verified": False, "continuous_coverage_verified": False,
            "unobserved_space": True, "safe_merge_available": False,
            "merge_reason": "between_rays_and_3d_unobserved", "calibration_verified": False,
            "registration_verified": False, "depth_samples": [], "lidar_samples": [],
            "measured_hits": [], "blocking_hits": []}


def _matrix(value):
    matrix = np.asarray(value)
    if matrix.shape != (4, 4) or matrix.dtype.kind not in ("i", "u", "f"):
        raise ValueError("invalid transform")
    matrix = matrix.astype(np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("invalid transform")
    rotation = matrix[:3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        if (not np.allclose(matrix[3], [0., 0., 0., 1.], rtol=0.0, atol=1e-9)
                or not np.allclose(np.dot(rotation.T, rotation), np.eye(3), rtol=0.0, atol=1e-6)
                or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6):
            raise ValueError("invalid transform")
    return matrix


def _vector(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3 or not all(_finite(v) for v in value):
        return None
    return tuple(float(v) for v in value)


def _crossing(origin, direction, section):
    if direction[0] <= 0 or origin[0] >= section["x_m"]:
        return None
    factor = (section["x_m"] - origin[0]) / direction[0]
    if not _finite(factor) or factor <= 0:
        return None
    point = tuple(origin[i] + factor * direction[i] for i in range(3))
    if not all(_finite(v) for v in point):
        return None
    if not (-section["half_width_m"] <= point[1] <= section["half_width_m"]
            and section["z_min_m"] <= point[2] <= section["z_max_m"]):
        return None
    return factor, point


def _record_hit(result, source, identifier, point, stamp, section):
    hit = {"source": source, "id": identifier, "point_base_m": point, "stamp": stamp}
    result["measured_hits"].append(hit)
    if (section["blocking_min_x_m"] <= point[0] <= section["x_m"]
            and abs(point[1]) <= section["half_width_m"]
            and section["z_min_m"] <= point[2] <= section["z_max_m"]):
        result["blocking_hits"].append(hit.copy())


def _classify(delta, surface, free):
    if abs(delta) <= surface:
        return "measured", "near_surface"
    if delta < -surface:
        return "occluded", "nearer_surface"
    if delta > free:
        return "sampled_free", "measured_beyond_crossing"
    return "unknown", "surface_margin"


def diagnose_depth_ray_coverage(depth, intrinsics, transform, lidar, section, timing):
    """Return separate 3-D point observations, retaining all unobserved space.

    depth: array (2-D uint16 mm or float32 m), encoding, frame_id, stamp.
    intrinsics: width/height/fx/fy/cx/cy, rectified=True, frame_id, stamp.
    transform: homogeneous matrix mapping optical-at-depth-stamp to base at
    reference_stamp, source_frame, target_frame, source_stamp, reference_stamp.
    lidar: frame_id, stamp, reference_stamp, and rays with 3-D start_base,
    hit_base and stamp. Each ray is a valid-return candidate, not a no-return.
    Both transforms into the shared reference must already include ego-motion.
    section: expected frame labels, x_m, half_width_m, z_min_m, z_max_m,
    blocking_min_x_m, explicit unique (u,v) pixels, depth min/max/near and
    surface/free margins, lidar min/max range and lidar_free_margin_m.
    timing: now and reference_stamp. Sensor ages are capped at 0.25 s,
    depth/lidar skew at 0.10 s, camera-info age at 1 s and skew at 0.20 s,
    and reference-pose age at 0.05 s. Future and mismatched times are rejected.

    ``measured_hits`` includes valid measured endpoints, even outside the
    section; ``blocking_hits`` is their subset inside the supplied rectangular
    region and has no floor classification. Samples are center rays, not bins.
    A diagnostic result is never permission to merge gaps or move the robot.
    """
    invalid = _empty("invalid_input", "invalid_input")
    try:
        if not all(isinstance(value, dict) for value in (depth, intrinsics, transform, lidar, section, timing)):
            return invalid
        image = depth["array"]
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            return invalid
        if depth["encoding"] == "16UC1" and image.dtype.kind == "u" and image.dtype.itemsize == 2:
            scale = 0.001
        elif depth["encoding"] == "32FC1" and image.dtype.kind == "f" and image.dtype.itemsize == 4:
            scale = 1.0
        else:
            return invalid
        height, width = image.shape
        iw, ih = intrinsics["width"], intrinsics["height"]
        if (isinstance(iw, bool) or not isinstance(iw, numbers.Integral)
                or isinstance(ih, bool) or not isinstance(ih, numbers.Integral)
                or not 1 <= width <= 8192 or not 1 <= height <= 8192
                or (iw, ih) != (width, height) or intrinsics["rectified"] is not True):
            return invalid
        fx, fy, cx, cy = [intrinsics[name] for name in ("fx", "fy", "cx", "cy")]
        if not all(_finite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            return invalid
        base, camera = section["expected_base_frame"], section["expected_camera_frame"]
        if (not _frame(base) or not _frame(camera) or base == camera
                or depth["frame_id"] != camera or intrinsics["frame_id"] != camera
                or transform["source_frame"] != camera or transform["target_frame"] != base
                or lidar["frame_id"] != base):
            return _empty("invalid_input", "frame_mismatch")
        matrix = _matrix(transform["matrix"])
        names = ("x_m", "half_width_m", "z_min_m", "z_max_m", "blocking_min_x_m",
                 "min_depth_m", "max_depth_m", "near_margin_m", "surface_tolerance_m", "free_margin_m",
                 "lidar_min_range_m", "lidar_max_range_m", "lidar_free_margin_m")
        if not all(_finite(section[name]) for name in names):
            return invalid
        x, half = section["x_m"], section["half_width_m"]
        zmin, zmax, xmin = section["z_min_m"], section["z_max_m"], section["blocking_min_x_m"]
        dmin, dmax, near = section["min_depth_m"], section["max_depth_m"], section["near_margin_m"]
        surface, free = section["surface_tolerance_m"], section["free_margin_m"]
        lmin, lmax, lfree = section["lidar_min_range_m"], section["lidar_max_range_m"], section["lidar_free_margin_m"]
        cutoff = dmin + near
        if (not _finite(cutoff) or not 0 < x <= 5 or not 0 < half <= 2 or not -2 <= zmin < zmax <= 3
                or not -1 <= xmin <= 0 or not 0 < dmin < cutoff < dmax <= 20
                or not 0 < surface < free <= 2 or not 0 < lmin < lmax <= 100 or not 0 < lfree <= 2):
            return invalid
        now, ref = timing["now"], timing["reference_stamp"]
        image_stamp, scan_stamp, info_stamp = depth["stamp"], lidar["stamp"], intrinsics["stamp"]
        values = (now, ref, image_stamp, scan_stamp, info_stamp,
                  transform["source_stamp"], transform["reference_stamp"], lidar["reference_stamp"])
        if not all(_finite(v) and v > 0 for v in values):
            return invalid
        if (transform["source_stamp"] != image_stamp or transform["reference_stamp"] != ref
                or lidar["reference_stamp"] != ref):
            return _empty("invalid_input", "transform_time_mismatch")
        if (not 0 <= now - ref <= 0.05 or not 0 <= ref - image_stamp <= 0.25
                or not 0 <= ref - scan_stamp <= 0.25 or not 0 <= now - image_stamp <= 0.25
                or not 0 <= now - scan_stamp <= 0.25 or abs(image_stamp - scan_stamp) > 0.10
                or not 0 <= now - info_stamp <= 1.0 or abs(info_stamp - image_stamp) > 0.20):
            return _empty("stale_or_unsynchronized", "sensor_or_reference_time")
        pixels, rays = section["pixels"], lidar["rays"]
        if (not isinstance(pixels, (list, tuple)) or len(pixels) > _MAX_ITEMS
                or not isinstance(rays, (list, tuple)) or len(rays) > _MAX_ITEMS):
            return invalid
        checked_pixels, seen = [], set()
        for pixel in pixels:
            if (not isinstance(pixel, (tuple, list)) or len(pixel) != 2
                    or any(isinstance(v, bool) or not isinstance(v, numbers.Integral) for v in pixel)):
                return invalid
            u, v = int(pixel[0]), int(pixel[1])
            if not 0 <= u < width or not 0 <= v < height or (u, v) in seen:
                return invalid
            checked_pixels.append((u, v))
            seen.add((u, v))
        for ray in rays:
            if not isinstance(ray, dict) or not _finite(ray.get("stamp")) or ray["stamp"] != scan_stamp:
                return invalid
        result = _empty("diagnostic", "discrete_samples_only")
        result.update({"base_frame": base, "reference_stamp": ref, "depth_stamp": image_stamp,
                       "lidar_stamp": scan_stamp, "section_x_m": float(x)})
        origin = tuple(float(v) for v in matrix[:3, 3])
        for u, v in checked_pixels:
            sample = {"pixel_uv": (u, v), "stamp": image_stamp, "status": "unknown", "reason": None,
                      "origin_base_m": origin, "crossing_base_m": None, "endpoint_base_m": None,
                      "target_optical_depth_m": None, "observed_optical_depth_m": None}
            raw = float(image[v, u]) * scale
            optical_direction = ((u - cx) / fx, (v - cy) / fy, 1.0)
            direction = tuple(float(np.dot(matrix[i, :3], optical_direction)) for i in range(3))
            if not all(_finite(item) for item in direction):
                return invalid
            if _finite(raw):
                sample["observed_optical_depth_m"] = raw
            valid_depth = _finite(raw) and cutoff < raw < dmax
            if valid_depth:
                endpoint = tuple(origin[i] + raw * direction[i] for i in range(3))
                if not all(_finite(item) for item in endpoint):
                    return invalid
                sample["endpoint_base_m"] = endpoint
                _record_hit(result, "depth", (u, v), endpoint, image_stamp, section)
            crossing = _crossing(origin, direction, section)
            if crossing is None:
                sample["reason"] = "outside_section_or_backward"
            else:
                target, point = crossing
                sample["crossing_base_m"] = point
                sample["target_optical_depth_m"] = target
                if not cutoff < target < dmax:
                    sample["reason"] = "target_outside_depth_working_range"
                elif not valid_depth:
                    sample["reason"] = "invalid_or_outside_depth_working_range"
                else:
                    sample["status"], sample["reason"] = _classify(raw - target, surface, free)
            result["depth_samples"].append(sample)
        for index, ray in enumerate(rays):
            sample = {"ray_index": index, "stamp": scan_stamp, "status": "unknown", "reason": None,
                      "origin_base_m": None, "crossing_base_m": None, "endpoint_base_m": None}
            start, hit = _vector(ray.get("start_base")), _vector(ray.get("hit_base"))
            if start is None or hit is None:
                sample["reason"] = "invalid_ray"
            else:
                delta = tuple(hit[i] - start[i] for i in range(3))
                distance = math.sqrt(sum(v * v for v in delta))
                if not _finite(distance) or not lmin <= distance < lmax:
                    sample["reason"] = "invalid_or_outside_lidar_range"
                else:
                    sample["origin_base_m"], sample["endpoint_base_m"] = start, hit
                    _record_hit(result, "lidar", index, hit, scan_stamp, section)
                    crossing = _crossing(start, delta, section)
                    if crossing is None:
                        sample["reason"] = "outside_section_or_backward"
                    else:
                        factor, point = crossing
                        sample["crossing_base_m"] = point
                        sample["status"], sample["reason"] = _classify((1.0 - factor) * distance, surface, lfree)
            result["lidar_samples"].append(sample)
        return result
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return invalid
