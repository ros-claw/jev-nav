"""Front-only, planar LiDAR samples for exploration analysis; never motion authority.

The caller must supply calibrated base-to-sensor extrinsics and synchronized,
gravity-level 2-D odometry poses in one continuous frame. The pose passed to
``add_scan`` must represent the whole scan at its timestamp; there is no
motion deskew, interpolation, uncertainty propagation, or TF lookup. Reset or
replace this object when odometry resets, calibration changes, or clocks reset.
Pose yaw angles must be normalized to [-pi, pi]. Calls mutate shared state and
must be serialized by the caller.

A finite reading at range_min <= r < range_max is treated as a hit. Zero,
out-of-range values, NaN, infinities, and maximum-range/no-return values are
unknown. The physical front sector is -pi/2..pi/2 in the sensor frame, and
samples outside it are ignored even if a driver reports values there. Sensor
mount yaw must therefore describe the physical forward-facing sector.

Coverage includes raw-value counts from the last accepted scan's front sector:
NaN, numeric zero, finite valid hits, and all other values. These are numerical
categories; a value alone does not establish why a sensor or driver emitted it.
When data is stale or a packet is rejected, counts may still describe the last
accepted scan. Check the status, sensor_fresh, and scan_stamp before using them.
If scan_stamp is None, the zero counts are placeholders, not a measured scan;
this also applies when the current pose is unusable.

Each retained observation is an ideal zero-width segment beginning at
range_min and ending at a hit, not a free-space cell. Past returns may be
occluded, stale, moved, or measured before a new obstacle appeared. The
same-pose coverage union counts only matching sample indices at exactly matching
scan poses and angular/range geometry; it says nothing about areas between rays
or current coverage after movement. Hits are retained independently of later
longer rays until they expire. When the scan cap fills, acquisition fails closed
instead of silently dropping unexpired occupied hits.

This module cannot establish clearance, swept-footprint safety, blind near-field
coverage, rear or 3-D coverage, braking behavior, or permission to move. No
output is executable; other independent safeguards and validation are required.
"""

from __future__ import division  # noqa: UP010 - Python 2 worker

import math
import sys

try:
    INTEGER_TYPES = (int, long)  # noqa: F821 - Python 2 ROS worker
except NameError:
    INTEGER_TYPES = (int,)
NUMBER_TYPES = INTEGER_TYPES + (float,)


def _finite(value):
    try:
        return (not isinstance(value, bool) and isinstance(value, NUMBER_TYPES)
                and not math.isnan(value) and not math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _to_world(pose, point):
    x, y, yaw = pose
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (x + cosine * point[0] - sine * point[1],
            y + sine * point[0] + cosine * point[1])


def _to_base(pose, point):
    x, y, yaw = pose
    cosine, sine = math.cos(yaw), math.sin(yaw)
    dx, dy = point[0] - x, point[1] - y
    return (cosine * dx + sine * dy, -sine * dx + cosine * dy)


def _valid_pose(pose):
    return isinstance(pose, (tuple, list)) and len(pose) == 3 and all(
        _finite(value) for value in pose
    ) and -math.pi <= pose[2] <= math.pi


class RollingScanFusion(object):  # noqa: UP004 - Python 2 ROS worker
    def __init__(self, sensor_pose, max_evidence_age=1.0, max_scan_age=0.5,
                 max_pose_skew=0.05, max_current_pose_age=0.1, max_scans=20):
        if (not _valid_pose(sensor_pose)
                or not all(_finite(value) for value in
                           (max_evidence_age, max_scan_age, max_pose_skew, max_current_pose_age))
                or not 0 < max_scan_age <= max_evidence_age
                or not 0 <= max_pose_skew <= max_scan_age
                or not 0 <= max_current_pose_age <= max_scan_age
                or isinstance(max_scans, bool) or not isinstance(max_scans, INTEGER_TYPES)
                or not 1 <= max_scans <= 200):
            raise ValueError("invalid scan fusion configuration")
        self.sensor_pose = tuple(sensor_pose)
        self.max_evidence_age = max_evidence_age
        self.max_scan_age = max_scan_age
        self.max_pose_skew = max_pose_skew
        self.max_current_pose_age = max_current_pose_age
        self.max_scans = max_scans
        self._scans = []
        self._latest = None
        self._latest_stamp = None
        self._last_now = None
        self._fault = None

    def _advance_clock(self, now):
        if not _finite(now) or now < 0:
            self._fault = "invalid_time"
            return False
        if self._last_now is not None and now < self._last_now:
            self._scans = []
            self._latest = None
            self._fault = "clock_regression"
            return False
        self._last_now = now
        self._scans = [record for record in self._scans
                       if now - record["stamp"] <= self.max_evidence_age]
        return True

    def add_scan(self, scan, pose, pose_stamp, now):
        """Ingest a timestamped scan and aligned odometry pose; return accepted bool.

        ``scan`` is a dict with stamp, ranges, angle_min, angle_increment,
        range_min, and range_max. Ranges must be a list or tuple, scan angles
        must lie within [-pi, pi] apart from 1e-5 rad of float32 rounding,
        and all times must share the same clock.
        Invalid packets latch a fault until a later valid packet is accepted.
        """
        if not self._advance_clock(now):
            return False
        try:
            stamp = scan["stamp"]
            ranges = scan["ranges"]
            angle_min = scan["angle_min"]
            increment = scan["angle_increment"]
            range_min = scan["range_min"]
            range_max = scan["range_max"]
            if (not isinstance(ranges, (list, tuple)) or not 1 <= len(ranges) <= 8192
                    or not _valid_pose(pose)
                    or not all(_finite(value) for value in
                               (stamp, pose_stamp, angle_min, increment, range_min, range_max))
                    or stamp < 0 or not 0 <= now - stamp <= self.max_scan_age
                    or pose_stamp < 0 or pose_stamp > now
                    or abs(pose_stamp - stamp) > self.max_pose_skew
                    or (self._latest_stamp is not None and stamp <= self._latest_stamp)
                    or not 0 < range_min < range_max or increment == 0):
                raise ValueError("invalid scan")
            end_angle = angle_min + (len(ranges) - 1) * increment
            angle_limit = math.pi + 1e-5
            if not _finite(end_angle) or not (-angle_limit <= angle_min <= angle_limit
                                               and -angle_limit <= end_angle <= angle_limit):
                raise ValueError("invalid scan angles")
            if len(self._scans) >= self.max_scans:
                self._fault = "capacity_exceeded"
                return False
            # Reject coordinate magnitudes that cannot resolve the blind-zone
            # radius to 0.01%, rather than producing deceptively precise rays.
            scale = max([1.0] + [abs(value) for value in pose[:2]]
                        + [abs(value) for value in self.sensor_pose[:2]] + [range_max])
            if scale * sys.float_info.epsilon * 16 >= range_min * 1e-4:
                raise ValueError("insufficient coordinate precision")
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            self._fault = "invalid_scan"
            return False
        sensor_origin = _to_world(pose, self.sensor_pose[:2])
        sensor_yaw = pose[2] + self.sensor_pose[2]
        if not all(_finite(value) for value in sensor_origin):
            self._fault = "invalid_scan"
            return False
        rays = []
        valid_indices = set()
        front_samples = 0
        valid_samples = 0
        nan_bins = 0
        zero_bins = 0
        other_bins = 0
        for index, distance in enumerate(ranges):
            angle = angle_min + index * increment
            if not -math.pi / 2 <= angle <= math.pi / 2:
                continue
            front_samples += 1
            if isinstance(distance, float) and math.isnan(distance):
                nan_bins += 1
                continue
            if (not isinstance(distance, bool) and isinstance(distance, NUMBER_TYPES)
                    and distance == 0):
                zero_bins += 1
                continue
            if not _finite(distance) or not 0 < distance < range_max or distance < range_min:
                other_bins += 1
                continue
            yaw = sensor_yaw + angle
            cosine, sine = math.cos(yaw), math.sin(yaw)
            start = (sensor_origin[0] + range_min * cosine,
                     sensor_origin[1] + range_min * sine)
            hit = (sensor_origin[0] + distance * cosine,
                   sensor_origin[1] + distance * sine)
            if not all(_finite(value) for value in start + hit):
                self._fault = "invalid_scan"
                return False
            valid_samples += 1
            valid_indices.add(index)
            rays.append({
                "start_world": start,
                "hit_world": hit,
                "stamp": stamp,
                "index": index,
            })
        signature = (tuple(pose), angle_min, increment, range_min, range_max, len(ranges))
        record = {"stamp": stamp, "rays": rays, "signature": signature,
                  "valid_indices": valid_indices}
        self._scans.append(record)
        self._latest = {"stamp": stamp, "signature": signature,
                        "front_samples": front_samples, "valid_samples": valid_samples,
                        "nan_bins": nan_bins, "zero_bins": zero_bins, "other_bins": other_bins}
        self._latest_stamp = stamp
        self._fault = None
        return True

    def snapshot(self, pose, pose_stamp, now):
        """Return evidence transformed to a fresh current base pose, never a permit.

        ``rays`` is populated only with fresh, fault-free sensor and pose inputs;
        each entry carries its own acquisition stamp. ``occupied_hits`` also
        retains unexpired hits during a sensor outage, but cannot be transformed
        if the current pose is invalid. Coverage refers to the most recent scan
        pose, not necessarily the current pose; check ``sensor_fresh`` and status.
        """
        result = {"status": "no_scan", "executable": False, "coverage": {
            "sensor_fresh": False, "front_samples": 0, "latest_valid_samples": 0,
            "latest_nan_bins": 0, "latest_zero_bins": 0, "latest_finite_valid_samples": 0,
            "latest_other_bins": 0,
            "latest_valid_fraction": None, "same_pose_valid_samples": 0,
            "same_pose_valid_fraction": None, "same_pose_scans": 0,
            "scan_stamp": None, "scan_pose": None,
        }, "rays": [], "occupied_hits": []}
        if not self._advance_clock(now):
            result["status"] = self._fault
            return result
        if (not _valid_pose(pose) or not _finite(pose_stamp) or pose_stamp < 0
                or not 0 <= now - pose_stamp <= self.max_current_pose_age):
            result["status"] = "pose_stale"
            return result
        latest = self._latest
        if latest is None:
            result["status"] = self._fault or "no_scan"
            return result
        if self._scans:
            minimum_range = min(record["signature"][3] for record in self._scans)
            scale = max([1.0] + [abs(value) for value in pose[:2]])
            if scale * sys.float_info.epsilon * 16 >= minimum_range * 1e-4:
                result["status"] = "pose_invalid"
                return result
        coverage = result["coverage"]
        coverage.update({
            "front_samples": latest["front_samples"],
            "latest_valid_samples": latest["valid_samples"],
            "latest_nan_bins": latest["nan_bins"], "latest_zero_bins": latest["zero_bins"],
            "latest_finite_valid_samples": latest["valid_samples"],
            "latest_other_bins": latest["other_bins"],
            "latest_valid_fraction": (latest["valid_samples"] / latest["front_samples"]
                                      if latest["front_samples"] else None),
            "scan_stamp": latest["stamp"], "scan_pose": latest["signature"][0],
        })
        union = set()
        for record in self._scans:
            if record["signature"] == latest["signature"]:
                union.update(record["valid_indices"])
                coverage["same_pose_scans"] += 1
        coverage["same_pose_valid_samples"] = len(union)
        if latest["front_samples"]:
            coverage["same_pose_valid_fraction"] = len(union) / latest["front_samples"]
        coverage["sensor_fresh"] = (self._fault is None
                                    and now - latest["stamp"] <= self.max_scan_age)
        if self._fault:
            result["status"] = self._fault
        elif not coverage["sensor_fresh"]:
            result["status"] = "sensor_stale"
        elif not latest["valid_samples"]:
            result["status"] = "no_valid_front_rays"
        else:
            result["status"] = "ok"
        rays = []
        # The pose is constant throughout this snapshot. Compute its rotation
        # once rather than repeating trigonometry for every retained endpoint.
        cosine, sine = math.cos(pose[2]), math.sin(pose[2])
        pose_x, pose_y = pose[0], pose[1]
        for record in self._scans:
            for ray in record["rays"]:
                sx = ray["start_world"][0] - pose_x
                sy = ray["start_world"][1] - pose_y
                hx = ray["hit_world"][0] - pose_x
                hy = ray["hit_world"][1] - pose_y
                start = (cosine * sx + sine * sy, -sine * sx + cosine * sy)
                hit = (cosine * hx + sine * hy, -sine * hx + cosine * hy)
                if any(math.isnan(value) or math.isinf(value) for value in start + hit):
                    result["status"] = "pose_invalid"
                    result["coverage"]["sensor_fresh"] = False
                    return result
                rays.append({"start_base": start, "hit_base": hit, "stamp": ray["stamp"]})
        result["occupied_hits"] = [{"point_base": ray["hit_base"], "stamp": ray["stamp"]}
                                   for ray in rays]
        if result["status"] == "ok":
            result["rays"] = rays
        return result
