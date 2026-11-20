#!/usr/bin/env python3
"""Advisory-only Decisions image experiment. This program cannot move LIMO."""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import re
import time
import urllib.request
from pathlib import Path

PNG_HEADER = b'\x89PNG\r\n\x1a\n'
MAX_IMAGE_BYTES = 4000000
CHOICES = ('forward', 'left', 'right', 'stop')


def read_key(path):
    keys = re.findall(r'\bsk-[A-Za-z0-9_-]{16,}', Path(path).read_text())
    if len(keys) != 1:
        raise ValueError('Expected exactly one API key in the configured file')
    return keys[0]


def make_request(png):
    if not isinstance(png, bytes) or not png.startswith(PNG_HEADER) or len(png) > MAX_IMAGE_BYTES:
        raise ValueError('Invalid or oversized PNG')
    return {
        'model': 'gpt-6-luna',
        'input': [{'role': 'user', 'content': [
            {'type': 'input_text', 'text': 'Robot camera image. This is a stationary advisory experiment; no movement will be executed.'},
            {'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')},
        ]}],
        'questions': [{'type': 'choice', 'name': 'next_action',
                       'instructions': 'Which direction is most promising for exploring? Choose stop if uncertain. Image alone does not establish safety.',
                       'choices': [{'value': 'forward', 'description': 'Explore ahead'},
                                   {'value': 'left', 'description': 'Explore to the left'},
                                   {'value': 'right', 'description': 'Explore to the right'},
                                   {'value': 'stop', 'description': 'Wait for more information'}]}],
    }


def http_transport(payload, key, timeout):
    req = urllib.request.Request('https://api.openai.com/v1/decisions',
                                 data=json.dumps(payload).encode('utf-8'),
                                 headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read(1000000))


def choose(png, key, transport=http_transport, timeout=10.0, clock=time.monotonic):
    result = {'choice': 'stop', 'executable': False, 'status': 'unavailable'}
    try:
        if not key or not 0 < timeout <= 30:
            return result
        payload = make_request(png)
        start = clock()
        response = transport(payload, key, timeout)
        elapsed = clock() - start
        result['elapsed_seconds'] = round(elapsed, 3)
        if elapsed < 0 or elapsed > timeout:
            result['status'] = 'expired'
            return result
        answers = response['answers']
        if len(answers) != 1 or answers[0].get('type') != 'choice' or answers[0].get('name') != 'next_action':
            return result
        choice = answers[0].get('choice')
        if choice not in CHOICES:
            return result
        result.update(choice=choice, status='ok')
        return result
    except Exception:
        return result


def choose_with_evidence(png, snapshot, key, transport=http_transport, timeout=8.0,
                         max_age=8.0, clock=time.time, allow_moving=False):
    """Choose an advisory direction tied to fresh observations, never an action."""
    result = {'choice': 'stop', 'executable': False, 'status': 'unavailable',
              'round_id': snapshot.get('round_id') if isinstance(snapshot, dict) else None}
    try:
        if not isinstance(snapshot, dict):
            return result
        serialized_snapshot = json.dumps(snapshot, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode('utf-8')
        snapshot = json.loads(serialized_snapshot)
        now = clock()
        if 'clock_domain' in snapshot:
            source = Path(__file__).resolve().parent / 'worktrees/limo-mcp/src/limo_ros_mcp/workers/scene_observation_core.py'
            spec = importlib.util.spec_from_file_location('limo_scene_validation', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.validate_snapshot(snapshot, now=now, max_age=max_age)
        result['round_id'] = snapshot.get('round_id')
        if not key or not isinstance(result['round_id'], str) or not result['round_id']:
            return result
        if not (0 < timeout <= 30 and 0 < max_age <= 30 and math.isfinite(now)):
            return result
        stamps = [snapshot[field] for field in ('image_stamp', 'laser_stamp', 'status_stamp', 'odom_stamp')]
        laser_evidence = snapshot.get('laser')
        if isinstance(laser_evidence, dict) and 'forward_view' in laser_evidence:
            if not isinstance(laser_evidence['forward_view'], dict):
                return result
            stamps.append(laser_evidence.get('forward_view_stamp'))
        depth = snapshot.get('depth', {'available': False, 'distance_measured': False})
        if depth.get('distance_measured'):
            if not depth.get('available') or depth.get('units') != 'optical_depth_m':
                return result
            for name in ('left', 'center', 'right'):
                sector = depth['sectors'][name]
                ratio, minimum = sector['valid_ratio'], sector['minimum_m']
                if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                        or not math.isfinite(ratio) or not 0 <= ratio <= 1):
                    return result
                if minimum is not None and (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
                                            or not math.isfinite(minimum) or minimum <= 0):
                    return result
                if (ratio == 0) != (minimum is None):
                    return result
            stamps.append(depth['stamp'])
        for stamp in stamps:
            if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                return result
            if stamp > now + .25 or now - stamp > max_age:
                return result
        result['observation_hash'] = 'sha256:' + hashlib.sha256(
            hashlib.sha256(png).digest() + serialized_snapshot).hexdigest()
        result['expires_wall_time'] = min(stamps) + max_age
        laser = snapshot['laser']
        base = snapshot['base']
        observed_sectors = 0
        for field in ('front_min_m', 'left_min_m', 'right_min_m'):
            value = laser[field]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                return result
            observed_sectors += 1
        if not observed_sectors:
            return result
        if (isinstance(laser['usable_ratio'], bool) or not isinstance(laser['usable_ratio'], (int, float))
                or not math.isfinite(laser['usable_ratio'])):
            return result
        if not 0 < laser['usable_ratio'] <= 1:
            return result
        for field in ('error_code', 'motion_mode', 'battery_voltage', 'linear_x_mps', 'angular_z_radps'):
            value = base[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return result
        if base['error_code'] != 0 or base['motion_mode'] != 0 or base['battery_voltage'] < 10:
            return result
        if type(allow_moving) is not bool:
            return result
        if not allow_moving and (abs(base['linear_x_mps']) > .01 or abs(base['angular_z_radps']) > .03):
            return result
        payload = make_request(png)
        payload['input'][0]['content'][0]['text'] = (
            'A differential-drive robot is exploring an indoor environment. '
            'The image is its forward camera view. Rank the next exploration direction '
            'using the observed scene and sensor evidence; motion safety is checked separately.'
        )
        evidence = {'round_id': result['round_id'], 'laser': laser, 'base': base,
                    'depth': depth,
                    'image_stamp': snapshot['image_stamp'], 'laser_stamp': snapshot['laser_stamp'],
                    'status_stamp': snapshot['status_stamp'], 'odom_stamp': snapshot['odom_stamp']}
        if 'sensor_frames' in snapshot:
            evidence['sensor_frames'] = snapshot['sensor_frames']
        payload['input'][0]['content'].append({'type': 'input_json', 'data': evidence})
        payload['questions'][0]['instructions'] = (
            'Choose the most promising direction for exploring this indoor scene. '
            'Use the image and measured laser sectors to distinguish open and obstructed directions. '
            'Prefer progress into visible open space when supported by the evidence. '
            'An obstacle in one direction does not automatically rule out other directions; '
            'choose stop if no direction is supported or a hazard affects every option. '
            'Depth values are minimum optical-axis depths by image column, not ground clearance; '
            'consider valid-pixel ratios, with missing measurements unknown. '
            'Laser sector minima do not establish swept-path safety; null laser sectors and missing depth distances are unknown. '
            'Laser directions use an unverified transform; RGB and depth orientation alignment is unverified. '
            'Forward-view sectors group rays from the laser origin, while nearest_bearing_from_base_rad '
            'describes each nearest hit relative to the robot base. max_unobserved_angle_rad includes sector edges; '
            'max_invalid_return_run_rad describes only consecutive invalid rays within the scan. '
            'no rear rays means the rear is unknown, '
            'and an unknown rear alone does not make the visible front obstructed. '
            'This output is advisory and is independently checked before any motion.'
        )
        response = transport(payload, key, timeout)
        finished = clock()
        result['elapsed_seconds'] = round(finished - now, 3)
        if finished < now or finished - now > timeout or any(finished - stamp > max_age for stamp in stamps):
            result['status'] = 'expired'
            return result
        answers = response['answers']
        if len(answers) != 1 or answers[0].get('type') != 'choice' or answers[0].get('name') != 'next_action':
            return result
        choice = answers[0].get('choice')
        if choice not in CHOICES:
            return result
        result.update(choice=choice, status='ok')
        probabilities = answers[0].get('probabilities')
        if isinstance(probabilities, list):
            result['probabilities'] = [{'value': p['value'], 'probability': p['probability']} for p in probabilities
                                       if isinstance(p, dict) and p.get('value') in CHOICES
                                       and isinstance(p.get('probability'), (int, float))
                                       and not isinstance(p['probability'], bool)
                                       and math.isfinite(p['probability']) and 0 <= p['probability'] <= 1]
            directional = [p for p in result['probabilities'] if p['value'] != 'stop']
            if len(directional) == 3 and {p['value'] for p in directional} == {'forward', 'left', 'right'}:
                result['direction_ranking'] = [p['value'] for p in sorted(
                    directional, key=lambda item: item['probability'], reverse=True)]
        return result
    except Exception:
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', help='PNG image path')
    parser.add_argument('--env', default='.env', help='API key file')
    parser.add_argument('--timeout', type=float, default=10.0)
    args = parser.parse_args()
    try:
        key = read_key(args.env)
        with open(args.image, 'rb') as image:
            png = image.read(MAX_IMAGE_BYTES + 1)
        result = choose(png, key, timeout=args.timeout)
    except (OSError, ValueError):
        result = {'choice': 'stop', 'executable': False, 'status': 'unavailable'}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
