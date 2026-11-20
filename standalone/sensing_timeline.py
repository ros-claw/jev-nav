#!/usr/bin/env python2
"""Read-only ROS sensing timing recorder; never publishes velocity or permits motion.

This records an assessment of the captured frame only. It does not check newer
frames or current vehicle motion and is not an executable perception verdict.
"""
from __future__ import division, print_function

import argparse
import collections
import json
import re
import sys
import time

from nav_timeline import Timeline
from relay import _monotonic

_CODE = re.compile(r'^[A-Za-z0-9_.:]{1,128}$')


def _code(value, default):
    return value if isinstance(value, str) and _CODE.match(value) else default


def measure_once(source, assess, clock, timeline, diagnostics=None):
    started = clock.now()
    fields = {'perception_start': started, 'clock_domain': 'monotonic'}
    try:
        sample = source.snapshot()
        if not isinstance(sample, dict):
            reason = _code(getattr(source, 'last_issue', None), 'sensor_unavailable')
        else:
            for name in ('scan_at', 'depth_at', 'odom_at', 'status_at', 'color_at', 'snapshot_at', 'epoch'):
                if name in sample:
                    fields[name] = sample[name]
            scene = sample.get('scene', {})
            fields['ray_count'] = len(scene.get('rays', []))
            fields['front_count'] = scene.get('coverage', {}).get('front_samples', 0)
            reason, assessment, detail = assess(sample, clock.now())
            if reason is None and isinstance(assessment, dict) and assessment.get('candidate') is True:
                reason = 'ready'
                fields['valid_until'] = assessment.get('valid_until')
            else:
                reason = _code(reason, 'assessment_failed')
                if isinstance(detail, dict):
                    fields['veto_reason'] = _code(detail.get('check'), 'none')
        if diagnostics is not None:
            fields.update(diagnostics())
    except Exception as exc:  # do not retain exception text or raw sensor payload
        reason = 'exception'
        fields['error_class'] = _code(type(exc).__name__, 'Exception')
    ended = clock.now()
    fields.update({'perception_end': ended, 'processing_sec': ended - started, 'reason': reason})
    recorded = timeline.record('perception', ended, fields)
    fields['timeline_recorded'] = recorded
    return fields


class Clock(object):
    def now(self):
        return _monotonic()


class NativeMeter(object):
    """Observe the fusion accelerator call without changing its return value."""
    def __init__(self, native):
        self.native = native
        self.used, self.reason = False, 'not_called'

    def __getattr__(self, name):
        return getattr(self.native, name)

    def fast_transform_records(self, records, pose):
        self.used, self.reason = False, 'native_declined'
        result = self.native.fast_transform_records(records, pose)
        if isinstance(result, tuple) and len(result) == 2:
            self.used, self.reason = True, 'none'
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeline', required=True, help='new private JSONL file')
    parser.add_argument('--seconds', type=float, default=30.0)
    parser.add_argument('--ground-z', type=float, default=-0.1)
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 300 or not -.5 <= args.ground_z <= 0:
        parser.error('invalid duration or ground-height assumption')
    import rospy
    import explore
    import limo_scan_fusion
    from live_trial_adapter import CvJpegEncoder, ObservationAdapter, _assess_sample
    from projected_depth_gate import ProjectedDepthGate

    rospy.init_node('jev_read_only_sensing_timeline', anonymous=True, disable_signals=True)
    if rospy.get_param('/use_sim_time', False):
        raise ValueError('wall clock is required')
    config = {'forward_min_m': .18, 'forward_max_m': 1., 'half_width_m': .15,
              'ground_z_m': args.ground_z, 'ground_tolerance_m': .025,
              'obstacle_min_height_m': .03, 'obstacle_max_height_m': .4,
              'min_depth_m': .3, 'max_depth_m': 3., 'sample_stride': 4}
    observation = explore.RosObservation(depth_projection_config=config)
    source = ObservationAdapter(observation, CvJpegEncoder(observation.bridge),
                                take_scene_ownership=True, projection_only_after_model=True,
                                recheck_snapshot=False)
    source.enable_safety_snapshots()
    gate, clock = ProjectedDepthGate(), Clock()
    native = limo_scan_fusion._native_scan
    meter = NativeMeter(native) if native is not None else None
    if meter is not None:
        limo_scan_fusion._native_scan = meter
    def diagnostics():
        scans = list(observation.fusion._scans) if observation.fusion is not None else []
        return {'retained_scan_count': len(scans),
                'retained_front_ray_count': sum(len(item['rays']) for item in scans),
                'accel_used': meter.used if meter is not None else False,
                'fallback_reason': meter.reason if meter is not None else 'native_unavailable'}
    timeline = Timeline(args.timeline, 'sensing-' + str(int(time.time())))
    counts = collections.Counter()
    started = clock.now()
    try:
        while clock.now() - started < args.seconds and not rospy.is_shutdown():
            if meter is not None:
                meter.used, meter.reason = False, 'not_called'
            item = measure_once(source, lambda sample, now: _assess_sample(sample, now, gate, False),
                                clock, timeline, diagnostics)
            counts[item['reason']] += 1
            if not item['timeline_recorded']:
                break
    finally:
        complete = timeline.close(5.0)
        rospy.signal_shutdown('read-only sensing recorder finished')
    print(json.dumps({'mode': 'read_only_sensing', 'motion_requested': False,
                      'counts': dict(counts), 'timeline_complete': complete,
                      'dropped': timeline.dropped, 'timeline': args.timeline}, sort_keys=True))
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
