"""Read-only, non-authoritative LiDAR/depth model comparison via LIMO MCP."""

import asyncio
import base64
import json
import math
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path

from calibration_compare import compare_models

ROOT = Path(__file__).resolve().parent
CAMERA_FRAMES = ("camera_color_optical_frame", "camera_depth_optical_frame")


def _number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("invalid finite numeric field")
    return float(value)


def _matrix(value):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        raise ValueError("invalid transform shape")
    matrix = [[_number(value) for value in row] for row in value]
    if any(abs(matrix[3][i] - v) > 1e-6 for i, v in enumerate((0, 0, 0, 1))):
        raise ValueError("invalid homogeneous transform")
    for i in range(3):
        for j in range(3):
            dot = sum(matrix[k][i] * matrix[k][j] for k in range(3))
            if abs(dot - (1 if i == j else 0)) > 1e-5:
                raise ValueError("invalid transform rotation")
    m = matrix
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    if abs(det - 1) > 1e-5:
        raise ValueError("reflected transform")
    return matrix


def _depth_rows(depth):
    width, height, step = (depth[key] for key in ("width", "height", "step"))
    if any(type(value) is not int for value in (width, height, step)):
        raise ValueError("invalid depth dimensions")
    try:
        fmt, bpp, scale = {"16UC1": ("H", 2, 0.001), "32FC1": ("f", 4, 1.0)}[
            depth["encoding"]
        ]
        if (
            not 1 <= width <= 1280
            or not 1 <= height <= 720
            or step < width * bpp
            or step * height > 2 * 1024 * 1024
            or type(depth["is_bigendian"]) is not bool
        ):
            raise ValueError("invalid depth buffer bounds")
        encoded = depth["data_base64"]
        if not isinstance(encoded, str) or len(encoded) > 2800000:
            raise ValueError("encoded depth exceeds bound")
        data = base64.b64decode(encoded, validate=True)
        if len(data) != step * height:
            raise ValueError("invalid depth buffer size")
        unpack = struct.Struct(
            (">" if depth["is_bigendian"] else "<") + fmt
        ).unpack_from
        return [
            [unpack(data, row * step + col * bpp)[0] * scale for col in range(width)]
            for row in range(height)
        ]
    except (KeyError, TypeError, ValueError, struct.error) as exc:
        raise ValueError("invalid depth buffer") from exc


def _depth_distribution(rows):
    values = sorted(
        value
        for row in rows
        for value in row
        if math.isfinite(value) and 0.05 <= value <= 20
    )
    count = len(values)
    counts = Counter(round(value * 100) for value in values)
    return {
        "valid_pixels": count,
        "total_pixels": sum(len(row) for row in rows),
        "median_m": (
            values[count // 2]
            if count % 2
            else (values[count // 2 - 1] + values[count // 2]) / 2
        )
        if count
        else None,
        "p90_m": values[math.ceil(0.9 * count) - 1] if count else None,
        "most_common_1cm": [
            {"depth_m": key / 100.0, "count": frequency}
            for key, frequency in counts.most_common(3)
        ],
    }


def analyze(bundle, sdk, *, now):
    """Compute diagnostic residuals; all frames and models remain unverified."""
    try:
        if (
            not isinstance(bundle, dict)
            or bundle.get("ok") is not True
            or bundle.get("schema_version") != "limo.calibration_diagnostic.v1"
            or bundle.get("command_dispatched") is not False
            or bundle.get("diagnostic_only") is not True
            or bundle.get("calibration_verified") is not False
            or bundle.get("transform_verified") is not False
            or bundle.get("usable_for_real_execution") is not False
            or bundle.get("deskewed") is not False
        ):
            raise ValueError("invalid diagnostic metadata")
        scan, depth, info, transform = (
            bundle[key] for key in ("scan", "depth", "camera_info", "transform")
        )
        if not all(isinstance(item, dict) for item in (scan, depth, info, transform)):
            raise ValueError("invalid sensor objects")
        stamp = [
            _number(scan["stamp"]),
            _number(depth["stamp"]),
            _number(info["stamp"]),
        ]
        current = _number(now)
        if any(value <= 0 or value > current or current - value > 5 for value in stamp):
            raise ValueError("sensor timestamps are stale or in the future")
        if max(stamp) - min(stamp) > 0.1:
            raise ValueError("sensor timestamps are not aligned")
        scan_frame = scan["frame_id"]
        if (
            not isinstance(scan_frame, str)
            or not scan_frame
            or len(scan_frame) > 128
            or depth["frame_id"] not in CAMERA_FRAMES
            or transform["source"] != scan_frame
            or transform["target"] != depth["frame_id"]
            or abs(_number(transform["stamp"]) - stamp[0]) > 1e-6
        ):
            raise ValueError("transform frame or timestamp mismatch")
        if (
            not isinstance(info["frame_id"], str)
            or len(info["frame_id"]) > 128
            or info["frame_matches_image"]
            is not (info["frame_id"] == depth["frame_id"])
            or info["dimensions_match_image"]
            is not (
                (info["width"], info["height"]) == (depth["width"], depth["height"])
            )
        ):
            raise ValueError("CameraInfo metadata is inconsistent")
        matrix = _matrix(transform["matrix"])
        ranges = scan["ranges"]
        if not isinstance(ranges, list) or not 1 <= len(ranges) <= 10000:
            raise ValueError("invalid scan sample count")
        angle, increment = _number(scan["angle_min"]), _number(scan["angle_increment"])
        minimum, maximum = _number(scan["range_min"]), _number(scan["range_max"])
        time_increment, scan_time = (
            _number(scan["time_increment"]),
            _number(scan["scan_time"]),
        )
        if (
            not 0 < minimum < maximum <= 100
            or increment == 0
            or not 0 <= time_increment <= 1
            or not 0 <= scan_time <= 2
            or time_increment * (len(ranges) - 1) > 2
        ):
            raise ValueError("invalid scan metadata")
        stride = max(1, math.ceil(len(ranges) / 5000))
        points = []
        for index in range(0, len(ranges), stride):
            try:
                distance = _number(ranges[index])
            except ValueError:
                continue
            if not minimum <= distance <= maximum:
                continue
            theta = angle + index * increment
            x, y = distance * math.cos(theta), distance * math.sin(theta)
            points.append(tuple(row[0] * x + row[1] * y + row[3] for row in matrix[:3]))
        values = info["matrices"]["K"]
        if not isinstance(values, list) or len(values) != 9:
            raise ValueError("invalid CameraInfo intrinsics")
        models = {
            "camera_info": {
                "fx": values[0],
                "fy": values[4],
                "cx": values[2],
                "cy": values[5],
            }
        }
        if (
            not isinstance(sdk, dict)
            or sdk.get("ok") is not True
            or sdk.get("command_dispatched") is not False
            or sdk.get("calibration_verified") is not False
            or sdk.get("usable_for_real_execution") is not False
        ):
            raise ValueError("SDK parameter observation unavailable")
        for name, key in (("sdk_left", "l_intr_p"), ("sdk_right", "r_intr_p")):
            values = sdk["parameters"][key]
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError("invalid SDK intrinsics")
            models[name] = dict(zip(("fx", "fy", "cx", "cy"), values))
        rows = _depth_rows(depth)
        comparison = compare_models(points, rows, models)
        return {
            "status": "diagnostic_only",
            "executable": False,
            "calibration_verified": False,
            "transform_verified": False,
            "scan_deskewed": False,
            "robot_stationarity_verified": False,
            "image_frame": depth["frame_id"],
            "camera_info_frame": info["frame_id"],
            "camera_info_frame_matches_image": info["frame_matches_image"],
            "camera_info_dimensions_match_image": info["dimensions_match_image"],
            "scan_stamp": stamp[0],
            "depth_stamp": stamp[1],
            "camera_info_stamp": stamp[2],
            "sensor_skew_sec": max(stamp) - min(stamp),
            "scan_duration_sec": time_increment * (len(ranges) - 1),
            "laser_valid_points": len(points),
            "depth_distribution": _depth_distribution(rows),
            "candidates": comparison["candidates"],
            "assumptions": comparison["assumptions"],
            "limitations": (
                "SDK parameters and TF are unverified; frame mismatch, distortion, motion, "
                "occlusion, parallax and a single scan can invalidate residuals"
            ),
        }
    except (KeyError, TypeError, OverflowError, struct.error) as exc:
        raise ValueError("invalid diagnostic bundle") from exc


async def capture():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = os.environ.copy()
    env.setdefault("ROSCLAW_PROJECT_ROOT", "/home/agilex/rosclaw/limo/rosclaw")
    env["PYTHONPATH"] = (
        str(ROOT / "worktrees/limo-mcp/src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "limo_ros_mcp.server", "--profile", "inspection"],
        env=env,
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        sdk = await asyncio.wait_for(
            session.call_tool("limo_get_dabai_camera_parameters", {}), timeout=10
        )
        bundle = await asyncio.wait_for(
            session.call_tool(
                "limo_capture_calibration_diagnostic", {"timeout_sec": 8}
            ),
            timeout=15,
        )
        if sdk.isError or bundle.isError:
            raise ValueError("MCP diagnostic call failed")
        return bundle.structuredContent, sdk.structuredContent


def main():
    try:
        bundle, sdk = asyncio.run(capture())
        if not isinstance(bundle, dict) or bundle.get("ok") is not True:
            raise ValueError(
                bundle.get("error", "diagnostic capture unavailable")
                if isinstance(bundle, dict)
                else "diagnostic capture unavailable"
            )
        result = analyze(bundle, sdk, now=time.time())
    except Exception as exc:  # noqa: BLE001 - diagnostic CLI reports unavailable on any transport failure
        result = {
            "status": "unavailable",
            "reason": str(exc)[:300],
            "executable": False,
            "calibration_verified": False,
        }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "diagnostic_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
