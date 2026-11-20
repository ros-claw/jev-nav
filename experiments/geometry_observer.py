#!/usr/bin/env python3
"""Read-only local geometry diagnostic; never sends motion commands."""
import asyncio
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def project_bundle(bundle, config, *, now):
    if (not isinstance(bundle, dict) or bundle.get('ok') is not True
            or bundle.get('command_dispatched') is not False
            or bundle.get('calibration_verified') is not False
            or bundle.get('usable_for_real_execution') is not False):
        raise ValueError('observation unavailable or metadata invalid')
    try:
        depth = dict(bundle['depth'])
        encoded = depth.pop('data_base64')
        if not isinstance(encoded, str) or len(encoded) > 2800000:
            raise ValueError('depth image exceeds bound')
        depth['data'] = base64.b64decode(encoded, validate=True)
        source = ROOT / 'worktrees/rosclaw/src/rosclaw/robot_pack/limo_local_geometry.py'
        spec = importlib.util.spec_from_file_location('limo_local_geometry', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_evidence_grid(bundle['scan'], depth, config, now=now)
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError('invalid local observation bundle') from exc


async def capture_bundle():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    env = os.environ.copy()
    env.setdefault('ROSCLAW_PROJECT_ROOT', '/home/agilex/rosclaw/limo/rosclaw')
    env['PYTHONPATH'] = str(ROOT / 'worktrees/limo-mcp/src') + os.pathsep + env.get('PYTHONPATH', '')
    params = StdioServerParameters(command=sys.executable,
        args=['-m', 'limo_ros_mcp.server', '--profile', 'inspection'], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await asyncio.wait_for(session.call_tool(
                'limo_capture_local_observation', {'timeout_sec': 8}), timeout=15)
            if result.isError or not isinstance(result.structuredContent, dict):
                raise ValueError('MCP observation call failed')
            return result.structuredContent


def main():
    config = {'resolution': .1, 'width': 80, 'height': 80, 'origin_x': -4.0, 'origin_y': -4.0,
              'min_z': .03, 'max_z': .6, 'max_age': 2.0, 'max_skew': .1}
    try:
        bundle = asyncio.run(capture_bundle())
        if not bundle.get('ok'):
            raise ValueError(bundle.get('error', 'observation unavailable'))
        grid = project_bundle(bundle, config, now=time.time())
        occupied = sum(cell == 100 for row in grid['cells'] for cell in row)
        summary = {'status': 'ok', 'frame_id': grid['frame_id'], 'stats': grid['stats'],
                   'occupied_cells': occupied, 'unknown_cells': config['width'] * config['height'] - occupied,
                   'source_stamps': grid['source_stamps'], 'executable': False,
                   'calibration_verified': False, 'safety_clearance_verified': False}
    except Exception as exc:
        summary = {'status': 'unavailable', 'reason': str(exc)[:300], 'executable': False}
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0 if summary['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
