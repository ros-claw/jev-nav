#!/usr/bin/env python2
"""Exercise live sensing and Decisions against a non-publishing shadow loop."""

from __future__ import division, print_function

import argparse
import json
import math
import os
import threading
import time


def timing_statistics(intervals):
    values = sorted(value for value in intervals
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    and not math.isnan(value) and not math.isinf(value) and value > 0)
    if not values:
        return {'samples': 0, 'median_sec': None, 'p95_sec': None, 'maximum_sec': None}
    count = len(values)
    middle = count // 2
    median = values[middle] if count % 2 else (values[middle - 1] + values[middle]) / 2
    return {'samples': count, 'median_sec': round(median, 6),
            'p95_sec': round(values[int(math.ceil(count * 0.95)) - 1], 6),
            'maximum_sec': round(values[-1], 6)}


def main():
    parser = argparse.ArgumentParser(description='Live shadow only; never publishes velocity')
    parser.add_argument('--seconds', type=float, default=15.0)
    args = parser.parse_args()
    if math.isnan(args.seconds) or math.isinf(args.seconds) or not 2 <= args.seconds <= 60:
        parser.error('--seconds must be between 2 and 60')
    import rospy
    from explore import WORKER, RosObservation, run_decision_process
    from relay import _monotonic
    from shadow_session import ShadowSession, make_decisions_request

    rospy.init_node('limo_standalone_shadow', anonymous=True, disable_signals=True)
    if rospy.get_param('/use_sim_time', False):
        raise ValueError('shadow observation requires ROS wall time')
    os.environ.setdefault('ROSCLAW_LIMO_DECISIONS_KEY_FILE',
                          os.path.expanduser('~/rosclaw/jev/.env'))
    source = RosObservation()
    observations = []
    sample_durations, sample_odom_ages = [], []
    unavailable_samples = [0]
    report_lock = threading.Lock()
    last_print = [0.0]
    read_snapshot = source.snapshot

    def measured_snapshot():
        started = _monotonic()
        sample = read_snapshot()
        finished = _monotonic()
        with report_lock:
            sample_durations.append(finished - started)
            if sample is None:
                unavailable_samples[0] += 1
            else:
                sample_odom_ages.append(finished - sample['odom_at'])
        return sample

    source.snapshot = measured_snapshot

    def report(item):
        with report_lock:
            observations.append(item['heartbeat_interval_sec'])
            now = _monotonic()
            if now - last_print[0] >= 1:
                print(json.dumps(item, allow_nan=False))
                last_print[0] = now

    def invoke(evidence, cancelled):
        return run_decision_process(['/usr/bin/python3', os.path.join(WORKER, 'limo_mapless_decision.py')],
                                    evidence, cancelled=cancelled, clock=_monotonic, timeout=6.0)

    session = ShadowSession(source, make_decisions_request(source, invoke), report,
                            duration_sec=args.seconds)
    session.start()
    try:
        while not rospy.is_shutdown() and not session.stop_event.is_set():
            time.sleep(0.1)
    finally:
        final = session.status()
        stopped = session.stop(timeout=2.0)
        with report_lock:
            timing = timing_statistics(observations)
            sample_timing = timing_statistics(sample_durations)
            odom_timing = timing_statistics(sample_odom_ages)
        print(json.dumps({'mode': 'shadow_complete', 'final': final,
                          'delivered_report_interval_timing': timing,
                          'sample_processing_timing': sample_timing,
                          'sample_odom_age': odom_timing,
                          'unavailable_samples': unavailable_samples[0],
                          'observation_status': source.diagnostic_status(),
                          'threads_still_alive': stopped['alive']}, allow_nan=False))
        rospy.signal_shutdown('shadow observation complete')


if __name__ == '__main__':
    main()
