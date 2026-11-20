"""Repeated read-only calibration diagnostics; never a calibration certificate."""

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time

from calibration_live_diagnostic import ROOT, analyze

MODELS = ("camera_info", "sdk_left", "sdk_right")
BANDS = ("0.05-2m", "2-4m", "4-6m", "6-20m")


def _finite(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def summarize(results):
    """Summarize correlated frames without selecting a model or claiming accuracy."""
    if not isinstance(results, list) or not 1 <= len(results) <= 5:
        raise ValueError("series must contain one to five frames")
    frames = []
    last_stamp = None
    for result in results:
        if not isinstance(result, dict) or result.get("executable") is not False:
            raise ValueError("invalid frame authority")
        if result.get("status") == "unavailable":
            continue
        if (
            result.get("status") != "diagnostic_only"
            or result.get("calibration_verified") is not False
            or result.get("transform_verified") is not False
            or result.get("scan_deskewed") is not False
            or result.get("robot_stationarity_verified") is not False
        ):
            raise ValueError("invalid diagnostic frame")
        stamp = result.get("scan_stamp")
        if not _finite(stamp) or (last_stamp is not None and stamp <= last_stamp):
            raise ValueError("duplicate or reordered scan")
        last_stamp = stamp
        candidates = result.get("candidates")
        depth = result.get("depth_distribution")
        if (
            not isinstance(candidates, dict)
            or set(candidates) != set(MODELS)
            or not isinstance(depth, dict)
        ):
            raise ValueError("invalid frame statistics")
        valid, total = depth.get("valid_pixels"), depth.get("total_pixels")
        if (
            type(valid) is not int
            or type(total) is not int
            or not 0 <= valid <= total
            or total <= 0
        ):
            raise ValueError("invalid depth coverage")
        for model in MODELS:
            item = candidates[model]
            if (
                not isinstance(item, dict)
                or type(item.get("matched_samples")) is not int
                or item["matched_samples"] < 0
                or item.get("status") not in ("scored", "insufficient_samples")
                or not isinstance(item.get("distance_bands"), dict)
                or set(item["distance_bands"]) != set(BANDS)
            ):
                raise ValueError("invalid candidate statistics")
            if item["status"] == "scored" and (
                item["matched_samples"] < 5
                or not _finite(item.get("median_abs_error_m"))
                or item["median_abs_error_m"] < 0
                or not _finite(item.get("median_signed_error_m"))
            ):
                raise ValueError("invalid scored candidate")
            for band in BANDS:
                count = item["distance_bands"][band].get("matched_samples")
                if type(count) is not int or count < 0:
                    raise ValueError("invalid distance band")
        frames.append(result)
    models = {}
    for name in MODELS:
        scored = [
            frame["candidates"][name]
            for frame in frames
            if frame["candidates"][name]["status"] == "scored"
        ]
        models[name] = {
            "scored_frames": len(scored),
            "median_of_frame_abs_errors_m": statistics.median(
                x["median_abs_error_m"] for x in scored
            )
            if scored
            else None,
            "median_of_frame_signed_errors_m": statistics.median(
                x["median_signed_error_m"] for x in scored
            )
            if scored
            else None,
            "matched_samples_by_band": {
                band: sum(
                    frame["candidates"][name]["distance_bands"][band]["matched_samples"]
                    for frame in frames
                )
                for band in BANDS
            },
        }
    ratios = [
        frame["depth_distribution"]["valid_pixels"]
        / frame["depth_distribution"]["total_pixels"]
        for frame in frames
    ]
    return {
        "status": "diagnostic_only",
        "temporal_status": "summarized" if len(frames) >= 3 else "insufficient_frames",
        "requested_frames": len(results),
        "successful_frames": len(frames),
        "failed_frames": len(results) - len(frames),
        "executable": False,
        "calibration_verified": False,
        "independent_accuracy_verified": False,
        "robot_stationarity_verified": False,
        "transform_verified": False,
        "median_valid_depth_ratio": statistics.median(ratios) if ratios else None,
        "models": models,
        "limitations": (
            "frames are correlated; motion, timestamp skew, unverified TF, camera frame mismatch, "
            "distortion, occlusion and limited distance coverage remain unresolved"
        ),
    }


async def capture_series(count):
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
    results = []
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        sdk = await asyncio.wait_for(
            session.call_tool("limo_get_dabai_camera_parameters", {}), timeout=10
        )
        if (
            sdk.isError
            or not isinstance(sdk.structuredContent, dict)
            or sdk.structuredContent.get("ok") is not True
        ):
            raise ValueError("SDK parameter observation unavailable")
        for _ in range(count):
            try:
                bundle = await asyncio.wait_for(
                    session.call_tool(
                        "limo_capture_calibration_diagnostic", {"timeout_sec": 8}
                    ),
                    timeout=15,
                )
                value = bundle.structuredContent
                if (
                    bundle.isError
                    or not isinstance(value, dict)
                    or value.get("ok") is not True
                ):
                    raise ValueError("diagnostic capture unavailable")
                results.append(analyze(value, sdk.structuredContent, now=time.time()))
            except (ValueError, asyncio.TimeoutError) as exc:
                results.append(
                    {
                        "status": "unavailable",
                        "executable": False,
                        "reason": str(exc)[:200],
                    }
                )
    return summarize(results)


def main():
    parser = argparse.ArgumentParser(
        description="Read-only correlated calibration diagnostics"
    )
    parser.add_argument("--frames", type=int, choices=range(1, 6), default=5)
    args = parser.parse_args()
    try:
        result = asyncio.run(capture_series(args.frames))
    except Exception as exc:  # noqa: BLE001 - diagnostic CLI reports transport failures as unavailable
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
