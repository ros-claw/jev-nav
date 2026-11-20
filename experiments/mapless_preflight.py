"""Read-only report of reported conditions; never an execution authority.

Inputs are not authenticated and conditions are not an exhaustive safety policy.
Only rosclawd can independently assess its trusted evidence and authorize work.
"""

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path


def _dict(value):
    return value if isinstance(value, dict) else {}


def _fresh(stamp, now):
    try:
        return (
            type(stamp) in (int, float)
            and type(now) in (int, float)
            and math.isfinite(stamp)
            and math.isfinite(now)
            and 0 <= now - stamp <= 2.0
        )
    except OverflowError:
        return False


def inspect_preflight(payload, *, now):
    payload = _dict(payload)
    runtime = _dict(payload.get("runtime"))
    obs = _dict(payload.get("observations"))
    physical = _dict(payload.get("physical_evidence"))
    pack = _dict(runtime.get("robot_pack"))
    ledger = _dict(runtime.get("ledger"))
    recovery = _dict(runtime.get("recovery"))
    interaction = _dict(runtime.get("interaction_plane"))
    executors = runtime.get("registered_executors")
    conditions = {
        "daemon_unavailable": runtime.get("running") is True,
        "privilege_boundary_unverified": runtime.get("privilege_separated") is True,
        "daemon_not_armed": runtime.get("supervision_state") == "ARMED",
        "estop_state_unavailable_or_latched": runtime.get("emergency_stop_latched")
        is False,
        "mapless_real_executor_unavailable": isinstance(executors, list)
        and "limo.explore:REAL" in executors,
        "signed_pack_unavailable": pack.get("loaded") is True
        and pack.get("signature_status") == "valid",
        "ledger_unavailable": ledger.get("integrity_verified") is True
        and ledger.get("write_failed") is False,
        "recovery_state_unavailable_or_required": recovery.get("required") is False,
        "operator_confirmation_unavailable": interaction.get("real_confirmation_ready")
        is True,
        "scan_not_fresh": _fresh(obs.get("scan_stamp"), now),
        "odometry_not_fresh": _fresh(obs.get("odom_stamp"), now),
        "base_status_not_fresh": _fresh(obs.get("status_stamp"), now),
        "base_health_unverified": obs.get("base_healthy") is True,
        "transform_unverified": obs.get("transform_verified") is True,
        "geometry_coverage_unverified": obs.get("geometry_coverage_verified") is True,
        "independent_stop_unverified": physical.get("independent_stop_verified")
        is True,
        "command_loss_stop_unverified": physical.get("command_loss_stop_verified")
        is True,
        "braking_profile_unverified": physical.get("braking_profile_verified") is True,
    }
    return {
        "unmet_reported_conditions": [key for key, ok in conditions.items() if not ok],
        "evidence_authenticated": False,
        "real_execution_authorized": False,
        "command_dispatched": False,
        "scope": "diagnostic only; conditions are neither authenticated nor exhaustive",
    }


async def _capture():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = os.environ.copy()
    env.setdefault("ROSCLAW_PROJECT_ROOT", "/home/agilex/rosclaw/limo/rosclaw")
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent / "worktrees/limo-mcp/src")
        + os.pathsep
        + env.get("PYTHONPATH", "")
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

        async def get(name, args):
            try:
                result = await asyncio.wait_for(
                    session.call_tool(name, args), timeout=15
                )
                return result.structuredContent if not result.isError else None
            except Exception:  # noqa: BLE001 - missing data is explicitly reported
                return None

        runtime, base, scan = await asyncio.gather(
            get("limo_get_runtime_status", {}),
            get("limo_get_base_state", {"timeout_sec": 5}),
            get("limo_capture_planar_scan", {"timeout_sec": 8}),
        )
    base = _dict(base)
    summaries = _dict(base.get("summaries")) if base.get("ok") is True else {}
    status = _dict(summaries.get("status"))
    odom = _dict(summaries.get("odometry"))
    scan = _dict(scan)
    if scan.get("ok") is not True or scan.get("diagnostic_only") is not True:
        scan = {}
    return {
        "runtime": runtime,
        "observations": {
            "scan_stamp": _dict(scan.get("scan")).get("stamp"),
            "odom_stamp": _dict(odom.get("header")).get("stamp_sec"),
            "status_stamp": _dict(status.get("header")).get("stamp_sec"),
            "base_healthy": status.get("healthy") is True,
            # No sensor or geometry diagnostic in this project certifies these.
            "transform_verified": False,
            "geometry_coverage_verified": False,
        },
        # Button presence alone does not establish its effect or stop distance.
        "physical_evidence": {},
    }


def main():
    try:
        payload = asyncio.run(_capture())
    except Exception:  # noqa: BLE001 - fail closed if MCP is unavailable
        payload = {}
    print(
        json.dumps(
            inspect_preflight(payload, now=time.time()),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
