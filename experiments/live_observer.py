#!/usr/bin/env python3
"""One read-only LIMO MCP observation and advisory visual decision."""
import argparse
import asyncio
import base64
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

import decision_observer


class ObservationError(Exception):
    pass


def forward_view(bundle):
    """Describe a fixed, unverified planar scan for the advisory model."""
    from calibration_live_diagnostic import _matrix

    if (not isinstance(bundle, dict) or bundle.get('ok') is not True
            or bundle.get('schema_version') != 'limo.planar_scan.v1'
            or bundle.get('command_dispatched') is not False
            or bundle.get('diagnostic_only') is not True
            or bundle.get('transform_verified') is not False
            or bundle.get('usable_for_real_execution') is not False
            or bundle.get('deskewed') is not False):
        raise ValueError('unavailable planar diagnostic')
    scan = bundle['scan']
    transform = scan['transform']
    if (transform['source'] != scan['frame_id'] or transform['target'] != 'base_link'
            or transform['stamp'] != scan['stamp']):
        raise ValueError('scan transform mismatch')
    matrix = _matrix(transform['matrix'])
    if abs(matrix[2][0]) > .1 or abs(matrix[2][1]) > .1:
        raise ValueError('laser plane is tilted')
    source = Path(__file__).resolve().parent / 'worktrees/limo-mcp/worker/limo_mapless_worker.py'
    spec = importlib.util.spec_from_file_location('limo_mapless_observer', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    view = module.summarize_forward_view(
        scan['ranges'], scan['angle_min'], scan['angle_increment'], scan['range_min'],
        scan['range_max'], matrix[0][3], matrix[1][3], math.atan2(matrix[1][0], matrix[0][0]))
    return view, scan['stamp']


def payload(result):
    if getattr(result, 'isError', False):
        raise ObservationError('MCP tool returned an error')
    structured = getattr(result, 'structuredContent', None)
    if isinstance(structured, dict):
        return structured
    for item in getattr(result, 'content', []):
        if getattr(item, 'type', None) == 'text':
            try:
                return json.loads(item.text)
            except (ValueError, TypeError):
                continue
    raise ObservationError('MCP tool returned no structured data')


async def collect_synchronized(session):
    """Collect one fixed-topic scene; ROS header timing is not exposure synchronization."""
    try:
        result = await asyncio.wait_for(
            session.call_tool('limo_capture_synchronized_scene', {'timeout_sec': 8}), timeout=15)
        metadata = payload(result)
        if (metadata.get('ok') is not True
                or metadata.get('schema_version') != 'limo.synchronized_scene.v1'
                or metadata.get('sync') != 'ros_header_approximate'
                or metadata.get('diagnostic_only') is not True
                or metadata.get('command_dispatched') is not False
                or metadata.get('usable_for_real_execution') is not False
                or not isinstance(metadata.get('snapshot'), dict)):
            raise ObservationError('synchronized_scene_unavailable')
        images = [item for item in result.content if getattr(item, 'type', None) == 'image']
        if len(images) != 1 or getattr(images[0], 'mimeType', None) != 'image/png':
            raise ObservationError('camera_unavailable')
        png = base64.b64decode(images[0].data, validate=True)
        if (not png.startswith(decision_observer.PNG_HEADER)
                or len(png) > decision_observer.MAX_IMAGE_BYTES
                or metadata.get('png_sha256') != 'sha256:' + hashlib.sha256(png).hexdigest()):
            raise ObservationError('invalid_camera_image')
        source = Path(__file__).resolve().parent / 'worktrees/limo-mcp/src/limo_ros_mcp/workers/scene_observation_core.py'
        spec = importlib.util.spec_from_file_location('limo_scene_validation', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.validate_png(png)
        module.validate_snapshot(metadata['snapshot'], now=time.time())
        if metadata.get('scene_sha256') != module.scene_digest(png, metadata['snapshot']):
            raise ObservationError('mismatched_scene')
        snapshot = dict(metadata['snapshot'])
        snapshot['round_id'] = uuid.uuid4().hex
        return png, snapshot
    except ObservationError:
        raise
    except Exception as exc:
        raise ObservationError('synchronized_scene_unavailable') from exc


async def collect(session):
    """Collect MCP-only evidence; raw depth distance is deliberately unavailable."""
    names = ('limo_capture_camera_frame', 'limo_get_laser_summary',
             'limo_get_base_state', 'limo_get_depth_summary', 'limo_capture_planar_scan')
    args = ({'stream': 'color', 'max_dimension': 640, 'timeout_sec': 8},
            {'timeout_sec': 6}, {'timeout_sec': 6}, {'timeout_sec': 6}, {'timeout_sec': 6})
    try:
        results = await asyncio.wait_for(asyncio.gather(
            *(session.call_tool(name, arg) for name, arg in zip(names, args)),
            return_exceptions=True), timeout=35)
        required = []
        for item, reason in zip(results[:3], ('camera_unavailable', 'lidar_unavailable', 'base_unavailable')):
            try:
                if isinstance(item, BaseException):
                    raise ObservationError(reason)
                data = payload(item)
                if data.get('ok') is not True:
                    raise ObservationError(reason)
                required.append(data)
            except ObservationError as exc:
                raise ObservationError(reason) from exc
        frame_data, laser_data, base_data = required
        try:
            camera_data = payload(results[3]) if not isinstance(results[3], BaseException) else None
        except ObservationError:
            camera_data = None
        images = [c for c in results[0].content if getattr(c, 'type', None) == 'image']
        if len(images) != 1:
            raise ObservationError('camera image unavailable')
        image = base64.b64decode(images[0].data, validate=True)
        if not image.startswith(decision_observer.PNG_HEADER) or len(image) > decision_observer.MAX_IMAGE_BYTES:
            raise ObservationError('invalid camera image')
        laser = laser_data['summary']
        sectors = laser['sectors']
        status = base_data['summaries']['status']
        odom = base_data['summaries']['odometry']
        depth = camera_data if camera_data and camera_data.get('ok') is True else None
        planar = None
        try:
            if not isinstance(results[4], BaseException):
                planar = forward_view(payload(results[4]))
        except (KeyError, TypeError, ValueError, OSError, AttributeError):
            pass
        snapshot = {
            'round_id': uuid.uuid4().hex,
            'image_stamp': frame_data['stamp_sec'],
            'laser_stamp': laser['header']['stamp_sec'],
            'status_stamp': status['header']['stamp_sec'],
            'odom_stamp': odom['header']['stamp_sec'],
            'laser': {'front_min_m': sectors['front_min_m'], 'left_min_m': sectors['left_min_m'],
                      'right_min_m': sectors['right_min_m'], 'rear_min_m': sectors.get('rear_min_m'),
                      'usable_ratio': laser['usable_ratio']},
            'base': {'error_code': status['error_code'], 'motion_mode': status['motion_mode'],
                     'battery_voltage': status['battery_voltage'],
                     'linear_x_mps': odom['velocity']['linear_x_mps'],
                     'angular_z_radps': odom['velocity']['angular_z_radps']},
            'depth': {'available': bool(depth), 'distance_measured': bool(depth),
                      'stamp': depth.get('stamp_sec') if depth else None,
                      'units': depth.get('units') if depth else None,
                      'sectors': depth.get('sectors') if depth else None,
                      'bands': depth.get('bands') if depth else None},
        }
        if planar is not None:
            snapshot['laser']['forward_view'], snapshot['laser']['forward_view_stamp'] = planar
        return image, snapshot
    except ObservationError:
        raise
    except Exception as exc:
        raise ObservationError('observation_incomplete') from exc


async def collect_with_retry(session):
    """Retry once by recapturing all sensors; never combine partial rounds."""
    for attempt in range(2):
        try:
            return await collect(session)
        except ObservationError:
            if attempt == 1:
                raise


async def run_once():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    env = os.environ.copy()
    env.setdefault('ROSCLAW_PROJECT_ROOT', '/home/agilex/rosclaw/limo/rosclaw')
    source = Path(__file__).resolve().parent / 'worktrees' / 'limo-mcp' / 'src'
    if source.is_dir():
        env['PYTHONPATH'] = str(source) + os.pathsep + env.get('PYTHONPATH', '')
    params = StdioServerParameters(command=sys.executable,
                                   args=['-m', 'limo_ros_mcp.server', '--profile', 'inspection'],
                                   env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for attempt in range(2):
                try:
                    return await collect_synchronized(session)
                except ObservationError as exc:
                    failure = exc
                    if attempt == 1:
                        break
    raise failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', default='.env')
    args = parser.parse_args()
    try:
        image, snapshot = asyncio.run(run_once())
        result = decision_observer.choose_with_evidence(image, snapshot, decision_observer.read_key(args.env))
        result['evidence'] = snapshot
    except ObservationError as exc:
        result = {'choice': 'stop', 'executable': False, 'status': 'observation_unavailable',
                  'reason_code': str(exc)}
    except Exception:
        result = {'choice': 'stop', 'executable': False, 'status': 'observation_unavailable'}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
