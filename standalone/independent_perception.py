"""Independent frame assessments for diagnostics; never motion authorization.

The caller owns ordering, TF/intrinsic validation, odometry, backlog and
calibration. Missing coverage remains unknown. No ROS or device I/O occurs here.
"""
from __future__ import division

import math

from explore import RollingScanFusion
from forward_trial import _raw_scan
from live_trial_adapter import _validated_front_expiry
from policy import assess_front
from projected_depth_gate import ProjectedDepthGate, project_depth_frame


def _result(clear, reason, deadline=None, **extra):
    extra.update({'clear': clear, 'reason': reason, 'valid_until': deadline,
                  'geometric_clearance_verified': False, 'executable': False})
    return extra


class ScanProcessor(object):  # noqa: UP004 - ROS Melodic
    def __init__(self, sensor_pose):
        self.sensor_pose = tuple(sensor_pose)
        self.fusion = RollingScanFusion(self.sensor_pose)
        self.last_stamp = None

    def process(self, raw, scan_pose, scan_pose_at, current_pose, current_pose_at, now):
        try:
            # Own this scan's range values across validation and fusion;
            # subsequent caller mutation is a different observation.
            if (type(raw) is not dict or type(raw.get('ranges')) not in (list, tuple)
                    or not 1 <= len(raw['ranges']) <= 8192):
                return _result(None, 'invalid_scan')
            raw = dict(raw)
            raw['ranges'] = tuple(raw['ranges'])
            stamp = raw['stamp']
            if (self.last_stamp is not None and stamp <= self.last_stamp
                    or raw['frame_id'].lstrip('/') != 'laser_link'):
                return _result(None, 'invalid_scan')
            self.last_stamp = stamp
            distance = math.hypot(scan_pose[0] - current_pose[0], scan_pose[1] - current_pose[1])
            heading = math.atan2(math.sin(scan_pose[2] - current_pose[2]),
                                 math.cos(scan_pose[2] - current_pose[2]))
            if any(math.isnan(value) or math.isinf(value) for value in (distance, heading)):
                return _result(None, 'pose_invalid')
            if distance > .01 or abs(heading) > .02:
                return _result(None, 'pose_changed')
            # Check an independently measured near hit before fusion. A full
            # or unavailable history must never conceal a current obstacle.
            provisional = {'raw_scan': raw, 'laser_extrinsics': self.sensor_pose, 'scan_at': stamp,
                           'scene': {'coverage': {'scan_pose': scan_pose}}, 'pose': current_pose}
            immediate, counts = _raw_scan(provisional, need_canonical=False, return_counts=True)
            if immediate is not None:
                return _result(False if immediate == 'front_blocked' else None, immediate)
            packet = {key: raw[key] for key in ('stamp', 'ranges', 'angle_min', 'angle_increment',
                                                'range_min', 'range_max')}
            if not self.fusion.add_scan(packet, scan_pose, scan_pose_at, now):
                return _result(None, 'fusion_add')
            scene = self.fusion.snapshot(current_pose, current_pose_at, now)
            if scene.get('status') != 'ok':
                status = scene.get('status')
                known = ('invalid_time', 'clock_regression', 'pose_stale', 'no_scan', 'pose_invalid',
                         'capacity_exceeded', 'invalid_scan', 'sensor_stale', 'no_valid_front_rays')
                return _result(None, 'fusion_' + status if status in known else 'fusion_unavailable')
            sample = {'raw_scan': raw, 'laser_extrinsics': self.sensor_pose, 'scan_at': stamp,
                      'scene': scene, 'pose': current_pose, 'snapshot_at': now}
            coverage = scene.get('coverage', {})
            if counts != (coverage.get('front_samples'), coverage.get('latest_valid_samples')):
                return _result(None, 'scan_count_mismatch', raw_front_count=counts[0],
                               raw_valid_count=counts[1], fused_front_count=coverage.get('front_samples'),
                               fused_valid_count=coverage.get('latest_valid_samples'))
            front = assess_front(scene, now, latest_age_limit=.5, history_age_limit=.65, history_span_limit=.4)
            reason = front.get('reason', 'invalid_evidence')
            if not front.get('candidate'):
                return _result(False if reason == 'forward_obstacle' else None, reason)
            coverage = _validated_front_expiry(sample)
            if coverage is None:
                return _result(None, 'coverage_expiry')
            deadline = min(stamp + .3, coverage)
            if deadline <= now:
                return _result(None, 'scan_expired')
            diagnostics = scene['coverage']
            unknown = sum(diagnostics[name] for name in ('latest_nan_bins', 'latest_zero_bins',
                                                          'latest_other_bins'))
            return _result(True if unknown == 0 else None, 'sampled_forward_corridor', deadline,
                           ray_count=len(scene['rays']), retained_scan_count=len(self.fusion._scans),
                           unknown_count=unknown, sampled_candidate=True, unobserved_space=True)
        except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            return _result(None, 'invalid_scan')


class DepthProcessor(object):  # noqa: UP004 - ROS Melodic
    def __init__(self, config):
        self.config = dict(config)
        self.gate = ProjectedDepthGate()
        self.last_stamp = None

    def process(self, image, encoding, intrinsics, transform, stamp, now):
        try:
            if self.last_stamp is not None and stamp <= self.last_stamp:
                return _result(None, 'invalid_depth')
            self.last_stamp = stamp
            if not 0 <= now - stamp < .15:
                return _result(None, 'depth_expired')
            corridor = project_depth_frame(image, encoding, intrinsics, transform, stamp, self.config)
            sample = {'depth_at': stamp, 'depth': {'stamp': stamp, 'corridor': corridor}}
            assessment = self.gate.assess(sample, now)
            reason = assessment.get('reason', 'depth_unavailable')
            if assessment.get('candidate') is True:
                invalid = corridor.get('invalid_count')
                if not isinstance(invalid, int) or invalid < 0:
                    return _result(None, 'invalid_depth')
                fully_sampled = invalid == 0 and corridor.get('sample_stride') == 1
                return _result(True if fully_sampled else None, reason, stamp + .15,
                               invalid_count=invalid, sampled_candidate=True, unobserved_space=True)
            return _result(False if reason == 'depth_obstacle' else None, reason)
        except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return _result(None, 'invalid_depth')
