#!/usr/bin/env python2
"""Reproducible in-memory scan snapshot/policy benchmark; never uses ROS or devices.

Run: PYTHONDONTWRITEBYTECODE=1 python2 standalone/benchmark_scan_pipeline.py
The timings are local synthetic measurements, not a physical safety guarantee.
"""

from __future__ import division  # noqa: UP010 - Python 2 benchmark

import argparse
import json
import math
import os
import sys
import timeit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
worker = os.path.join(ROOT, "worker")
if not os.path.isdir(worker):
    worker = os.path.join(ROOT, "worktrees", "limo-mcp", "worker")
sys.path.insert(0, worker)
from limo_scan_fusion import RollingScanFusion  # noqa: E402
from policy import assess_front  # noqa: E402


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(math.ceil(fraction * len(ordered))) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--scans", type=int, default=10)
    parser.add_argument("--rays", type=int, default=443)
    args = parser.parse_args()
    if not (1 <= args.rounds <= 10000 and 1 <= args.scans <= 10 and 3 <= args.rays <= 8192):
        parser.error("rounds, scans, or rays out of bounds")
    fusion = RollingScanFusion((0.105, 0.0, 0.0))
    pose = None
    for frame in range(args.scans):
        stamp = 100.0 + frame * 0.1
        pose = (frame * 0.001, 0.0, frame * 0.0002)
        ranges = [2.5 if (index * 37 + frame * 19) % 100 < 43 else 0.0
                  for index in range(args.rays)]
        packet = {"stamp": stamp, "ranges": ranges, "angle_min": -math.pi / 2,
                  "angle_increment": math.pi / (args.rays - 1),
                  "range_min": 0.1, "range_max": 10.0}
        if not fusion.add_scan(packet, pose, stamp, stamp):
            raise RuntimeError("synthetic scan unexpectedly rejected")
    for _ in range(5):
        scene = fusion.snapshot(pose, stamp, stamp)
        expected = assess_front(scene, stamp)
    snapshot_ms, policy_ms, combined_ms = [], [], []
    for _ in range(args.rounds):
        before = timeit.default_timer()
        scene = fusion.snapshot(pose, stamp, stamp)
        middle = timeit.default_timer()
        result = assess_front(scene, stamp)
        after = timeit.default_timer()
        if result != expected:
            raise RuntimeError("synthetic policy result changed within a run")
        snapshot_ms.append(1000 * (middle - before))
        policy_ms.append(1000 * (after - middle))
        combined_ms.append(1000 * (after - before))
    measurements = {}
    for name, values in (("snapshot_ms", snapshot_ms), ("policy_ms", policy_ms),
                         ("combined_ms", combined_ms)):
        measurements[name] = {"median": round(percentile(values, 0.5), 3),
                              "p95": round(percentile(values, 0.95), 3),
                              "max": round(max(values), 3)}
    print(json.dumps({"python": sys.version.split()[0], "scans": args.scans,
                      "rays_per_scan": args.rays, "retained_rays": len(scene["rays"]),
                      "latest_valid_samples": scene["coverage"]["latest_valid_samples"],
                      "latest_valid_fraction": scene["coverage"]["latest_valid_fraction"],
                      "policy_result": expected, "rounds": args.rounds,
                      "measurements": measurements}, sort_keys=True))


if __name__ == "__main__":
    main()
