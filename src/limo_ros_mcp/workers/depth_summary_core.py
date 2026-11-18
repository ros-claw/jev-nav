"""Bounded optical depth statistics; compatible with ROS Melodic Python 2.

Image sectors are pixel columns, not robot-frame clearances. No collision or
floor safety claim can be derived from this summary.
"""

import math
import struct


def summarize(data, width, height, step, encoding, is_bigendian):
    formats = {"16UC1": ("H", 2, 0.001), "32FC1": ("f", 4, 1.0)}
    if encoding not in formats:
        raise ValueError("unsupported depth encoding")
    fmt, pixel_bytes, scale = formats[encoding]
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or not isinstance(step, int)
        or width < 3
        or height < 1
        or width * height > 1920 * 1080
        or step < width * pixel_bytes
        or step * height > 16 * 1024 * 1024
        or len(data) != step * height
    ):
        raise ValueError("invalid depth dimensions or data length")
    unpack = struct.Struct((">" if is_bigendian else "<") + fmt).unpack_from
    names = ("left", "center", "right")
    counts = [0, 0, 0]
    totals = [0, 0, 0]
    minimum = [None, None, None]
    band_counts = [[0] * 3 for _ in range(3)]
    band_totals = [[0] * 3 for _ in range(3)]
    band_minimum = [[None] * 3 for _ in range(3)]
    for row in range(height):
        row_offset = row * step
        band = min(2, 3 * row // height)
        for col in range(width):
            sector = min(2, 3 * col // width)
            totals[sector] += 1
            band_totals[band][sector] += 1
            value = float(unpack(data, row_offset + col * pixel_bytes)[0]) * scale
            if math.isnan(value) or math.isinf(value) or value <= 0 or value > 20:
                continue
            counts[sector] += 1
            band_counts[band][sector] += 1
            if minimum[sector] is None or value < minimum[sector]:
                minimum[sector] = value
            if band_minimum[band][sector] is None or value < band_minimum[band][sector]:
                band_minimum[band][sector] = value
    return {
        "sectors": {
            name: {
                "minimum_m": minimum[i],
                "valid_count": counts[i],
                "total_count": totals[i],
                "valid_ratio": float(counts[i]) / totals[i],
            }
            for i, name in enumerate(names)
        },
        "bands": {
            band_name: {
                name: {
                    "minimum_m": band_minimum[b][i],
                    "valid_count": band_counts[b][i],
                    "total_count": band_totals[b][i],
                    "valid_ratio": (
                        float(band_counts[b][i]) / band_totals[b][i] if band_totals[b][i] else None
                    ),
                }
                for i, name in enumerate(names)
            }
            for b, band_name in enumerate(("top", "middle", "bottom"))
        },
        "units": "optical_depth_m",
        "safety_clearance_verified": False,
    }
