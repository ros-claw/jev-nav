"""Map explicit diagnostic depth sample centers to cell IDs, not free volumes.

The caller must validate acquisition time, epoch, pose registration, sensor
calibration and the cell geometry before using these samples. Between samples,
behind surfaces and outside the field of view remain unobserved. This module
does not authorize movement.
"""
from __future__ import division

import math

try:
    INTEGER_TYPES = (int, long)  # noqa: F821
except NameError:
    INTEGER_TYPES = (int,)


def _center(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(isinstance(x, bool) or not isinstance(x, INTEGER_TYPES + (float,))
                   or math.isnan(x) or math.isinf(x) for x in value)):
        raise ValueError('invalid sample center')
    return tuple(float(x) for x in value)


def depth_cells(visibility, cells):
    """Preserve only classifications for the exact explicitly requested centers."""
    if (not isinstance(visibility, dict) or visibility.get('status') != 'ok'
            or visibility.get('executable') is not False
            or visibility.get('geometric_clearance_verified') is not False
            or visibility.get('unobserved_space') is not True
            or not isinstance(cells, (tuple, list)) or len(cells) > 8192
            or not isinstance(visibility.get('voxels'), list) or len(cells) != len(visibility['voxels'])):
        raise ValueError('invalid depth visibility')
    output = {'free': [], 'occupied': [], 'unknown': [], 'executable': False,
              'unobserved_space': True, 'geometric_clearance_verified': False}
    seen = set()
    for pair, voxel in zip(cells, visibility['voxels']):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2 or not isinstance(voxel, dict):
            raise ValueError('invalid cell mapping')
        cell, center = pair
        if (not isinstance(cell, (tuple, list)) or len(cell) != 3
                or any(isinstance(x, bool) or not isinstance(x, INTEGER_TYPES) or x < 0 for x in cell)):
            raise ValueError('invalid cell')
        cell = tuple(cell)
        if cell in seen or _center(center) != _center(voxel.get('center_base_m')):
            raise ValueError('duplicate or mismatched center')
        seen.add(cell)
        status = voxel.get('status')
        if status == 'sampled_free':
            output['free'].append(cell)
        elif status == 'measured':
            output['occupied'].append(cell)
        elif status in ('unknown', 'occluded'):
            output['unknown'].append(cell)
        else:
            raise ValueError('invalid sample status')
    return output
