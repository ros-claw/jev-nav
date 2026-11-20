"""Offline camera-model diagnostic from caller-supplied synthetic or recorded data.

Points must already be in the camera optical frame. Pixel values are assumed
to be optical-axis depth in metres. This module does not verify those premises.
It has no robot or ROS transport and cannot certify geometry or issue motion.
"""

import math
import re


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _median(values):
    values = sorted(values)
    middle = len(values) // 2
    return (
        values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    )


def compare_models(points, depth, models):
    """Compare residuals without selecting a winner or declaring calibration valid."""
    if not isinstance(points, (list, tuple)) or len(points) > 5000:
        raise ValueError("points must be a bounded sequence")
    if (
        not isinstance(depth, (list, tuple))
        or not depth
        or len(depth) > 1024
        or not isinstance(depth[0], (list, tuple))
        or not depth[0]
        or len(depth[0]) > 2048
        or len(depth) * len(depth[0]) > 1048576
    ):
        raise ValueError("invalid depth image dimensions")
    height, width = len(depth), len(depth[0])
    if any(not isinstance(row, (list, tuple)) or len(row) != width for row in depth):
        raise ValueError("depth rows must be rectangular")
    if not isinstance(models, dict) or not 1 <= len(models) <= 8:
        raise ValueError("models must be a bounded non-empty mapping")
    for name, model in models.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
            or not isinstance(model, dict)
            or set(model) != {"fx", "fy", "cx", "cy"}
            or any(not _finite(value) for value in model.values())
            or model["fx"] <= 0
            or model["fy"] <= 0
        ):
            raise ValueError("invalid candidate intrinsics")
    summaries = {}
    for name, model in models.items():
        residuals, expected_depths, measured_depths, signed_errors, nearest = (
            [],
            [],
            [],
            [],
            {},
        )
        for point in points:
            if (
                not isinstance(point, (list, tuple))
                or len(point) != 3
                or not all(_finite(value) for value in point)
            ):
                continue
            x, y, z = point
            if not 0.05 <= z <= 20:
                continue
            u, v = model["fx"] * x / z + model["cx"], model["fy"] * y / z + model["cy"]
            if (
                not _finite(u)
                or not _finite(v)
                or not (-0.5 <= u < width - 0.5 and -0.5 <= v < height - 0.5)
            ):
                continue
            col, row = math.floor(u + 0.5), math.floor(v + 0.5)
            pixel = (row, col)
            if pixel not in nearest or z < nearest[pixel]:
                nearest[pixel] = z
        for (row, col), z in nearest.items():
            measured = depth[row][col]
            if not _finite(measured) or not 0.05 <= measured <= 20:
                continue
            residuals.append(abs(measured - z))
            expected_depths.append(z)
            measured_depths.append(measured)
            signed_errors.append(measured - z)
        residuals.sort()
        count = len(residuals)
        if count >= 5:
            median = _median(residuals)
            p90 = residuals[math.ceil(0.9 * count) - 1]
            expected, measured, signed = map(
                _median, (expected_depths, measured_depths, signed_errors)
            )
        else:
            median = p90 = expected = measured = signed = None
        bands = {}
        for label, low, high in (
            ("0.05-2m", 0.05, 2.0),
            ("2-4m", 2.0, 4.0),
            ("4-6m", 4.0, 6.0),
            ("6-20m", 6.0, 20.000001),
        ):
            errors = [
                observed - predicted
                for predicted, observed in zip(expected_depths, measured_depths)
                if low <= predicted < high
            ]
            bands[label] = {
                "matched_samples": len(errors),
                "median_abs_error_m": _median([abs(value) for value in errors])
                if len(errors) >= 5
                else None,
                "median_signed_error_m": _median(errors) if len(errors) >= 5 else None,
            }
        summaries[name] = {
            "status": "scored" if count >= 5 else "insufficient_samples",
            "matched_samples": count,
            "median_abs_error_m": median,
            "p90_abs_error_m": p90,
            "median_expected_depth_m": expected,
            "median_measured_depth_m": measured,
            "median_signed_error_m": signed,
            "distance_bands": bands,
        }
    return {
        "candidates": summaries,
        "calibration_verified": False,
        "executable": False,
        "assumptions": (
            "caller-supplied optical-frame points; depth is optical Z in metres; "
            "undistorted pinhole candidates; nearest input point per pixel"
        ),
    }
