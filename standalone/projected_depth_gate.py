"""Supervised trial gate for sampled, floor-classified depth observations.

This recognizes measured ground separately from measured obstacle points.
Missing depth, the camera near field, occlusion and areas between samples stay
unknown. The decision never claims a geometric clearance certificate.
"""

from __future__ import division

import math

from depth_corridor import diagnose_depth_corridor

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - ROS Melodic Python 2
except NameError:
    INTEGER_TYPES = (int,)
NUMBER_TYPES = INTEGER_TYPES + (float,)


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def project_depth_frame(array, encoding, intrinsics, transform, stamp, config):
    """Classify one depth frame with explicitly assumed, unverified calibration."""
    if not _finite(stamp):
        return None
    result = diagnose_depth_corridor(array, encoding, intrinsics, transform, config)
    result.update({'stamp': stamp, 'calibration_verified': False})
    return result


class ProjectedDepthGate(object):  # noqa: UP004 - ROS Melodic Python 2
    def assess(self, sample, now):
        result = {'candidate': False, 'reason': 'depth_unavailable', 'depth_at': None,
                  'geometric_clearance_verified': False, 'unobserved_space': True,
                  'calibration_verified': False}
        try:
            depth_at = sample['depth_at']
            if not _finite(depth_at):
                return result
            result['depth_at'] = depth_at
            if not _finite(now) or not 0 <= now - depth_at <= 0.5:
                return result
            depth = sample['depth']
            corridor = depth['corridor']
            if (depth['stamp'] != depth_at or corridor['stamp'] != depth_at
                    or corridor['status'] != 'observed'
                    or corridor['executable'] is not False
                    or corridor['geometric_clearance_verified'] is not False
                    or corridor['calibration_verified'] is not False
                    or corridor['unobserved_space'] is not True):
                return result
            names = ('sample_count', 'valid_count', 'invalid_count', 'sample_stride',
                     'ground_count', 'obstacle_count', 'unclassified_corridor_count',
                     'outside_corridor_count')
            counts = [corridor[name] for name in names]
            if (any(isinstance(value, bool) or not isinstance(value, INTEGER_TYPES) or value < 0
                    for value in counts)
                    or not 1 <= corridor['sample_stride'] <= 8
                    or corridor['valid_count'] + corridor['invalid_count'] != corridor['sample_count']
                    or sum(corridor[name] for name in names[4:]) != corridor['valid_count']):
                return result
            if corridor['obstacle_count']:
                result['reason'] = 'depth_obstacle'
                return result
            if corridor['unclassified_corridor_count']:
                result['reason'] = 'depth_unclassified'
                return result
            if corridor['valid_count'] < 100 or corridor['ground_count'] < 30:
                return result
            result.update({'candidate': True, 'reason': 'sampled_depth_corridor',
                           'valid_count': corridor['valid_count'],
                           'ground_count': corridor['ground_count']})
            return result
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return result
