"""Read-only planar LiDAR ray sampling; never a clearance or motion decision."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def analyze(bundle, *, now):
    if (
        not isinstance(bundle, dict)
        or bundle.get("ok") is not True
        or bundle.get("schema_version") != "limo.planar_scan.v1"
        or bundle.get("command_dispatched") is not False
        or bundle.get("diagnostic_only") is not True
        or bundle.get("calibration_verified", False) is not False
        or bundle.get("transform_verified") is not False
        or bundle.get("usable_for_real_execution") is not False
        or bundle.get("deskewed") is not False
    ):
        raise ValueError("invalid diagnostic metadata")
    try:
        scan = bundle["scan"]
        transform = scan.get("transform") if isinstance(scan, dict) else None
        if (
            not isinstance(scan, dict)
            or not isinstance(transform, dict)
            or transform.get("source") != scan.get("frame_id")
            or transform.get("target") != "base_link"
            or transform.get("stamp") != scan.get("stamp")
            or transform.get("static") is not False
        ):
            raise ValueError("laser-to-base transform unavailable or mismatched")
        scan = dict(scan)
        scan["transform"] = dict(transform)
        sys.path.insert(0, str(ROOT / "worktrees/rosclaw/src"))
        from rosclaw.robot_pack.limo_planar_evidence import build_planar_evidence

        config = {
            "resolution": 0.1,
            "width": 80,
            "height": 80,
            "origin_x": -4.0,
            "origin_y": -4.0,
            "max_age": 2.0,
            "max_ray_distance": 8.0,
        }
        evidence = build_planar_evidence(scan, config, now=now)
        return {
            "status": "diagnostic_only",
            "meaning": evidence["meaning"],
            "scan_stamp": evidence["stamp"],
            "frame_id": evidence["frame_id"],
            "plane_height_m": evidence["plane_height_m"],
            "valid_rays_processed": evidence["valid_rays_processed"],
            "ray_sampled_cell_count": len(evidence["ray_sampled_cells"]),
            "hit_cell_count": len(evidence["hit_cells"]),
            "transform_verified": False,
            "safety_clearance_verified": False,
            "scan_deskewed": False,
            "executable": False,
            "limitations": "sampled planar rays do not establish cell clearance or 3D obstacle coverage",
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid planar diagnostic bundle") from exc


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
        result = await asyncio.wait_for(
            session.call_tool("limo_capture_planar_scan", {"timeout_sec": 8}),
            timeout=15,
        )
        if result.isError or not isinstance(result.structuredContent, dict):
            raise ValueError("MCP diagnostic unavailable")
        return result.structuredContent


def main():
    try:
        bundle = asyncio.run(capture())
        if not isinstance(bundle, dict) or bundle.get("ok") is not True:
            raise ValueError(
                bundle.get("error", "planar scan unavailable")
                if isinstance(bundle, dict)
                else "planar scan unavailable"
            )
        result = analyze(bundle, now=time.time())
    except Exception as exc:  # noqa: BLE001 - read-only CLI reports unavailable on transport errors
        result = {
            "status": "unavailable",
            "reason": str(exc)[:300],
            "executable": False,
            "safety_clearance_verified": False,
        }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "diagnostic_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
