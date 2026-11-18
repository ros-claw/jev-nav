"""Bounded scan and TF diagnostics only; Python 2 compatible, no actuation."""

from calibration_bundle_core import _rigid
from local_observation_core import _frame, _number


def make_planar_scan(scan, scan_to_base, now):
    current, stamp = _number(now), _number(scan["stamp"])
    if stamp <= 0 or stamp > current or current - stamp > 2:
        raise ValueError("stale or future laser scan")
    frame = _frame(scan["frame_id"])
    angle, increment = _number(scan["angle_min"]), _number(scan["angle_increment"])
    minimum, maximum = _number(scan["range_min"]), _number(scan["range_max"])
    time_increment, scan_time = _number(scan["time_increment"]), _number(scan["scan_time"])
    ranges = scan["ranges"]
    if (
        not isinstance(ranges, (list, tuple))
        or not 1 <= len(ranges) <= 10000
        or increment == 0
        or not 0 < minimum < maximum <= 100
        or not 0 <= time_increment <= 1
        or not 0 <= scan_time <= 2
        or time_increment * (len(ranges) - 1) > 2
    ):
        raise ValueError("invalid scan metadata")
    clean = []
    for value in ranges:
        try:
            clean.append(_number(value))
        except ValueError:
            clean.append(None)
    return {
        "scan": {
            "stamp": stamp,
            "frame_id": frame,
            "angle_min": angle,
            "angle_increment": increment,
            "range_min": minimum,
            "range_max": maximum,
            "time_increment": time_increment,
            "scan_time": scan_time,
            "ranges": clean,
            "transform": {
                "source": frame,
                "target": "base_link",
                "stamp": stamp,
                "static": False,
                "matrix": _rigid(scan_to_base),
            },
        },
        "diagnostic_only": True,
        "transform_verified": False,
        "usable_for_real_execution": False,
        "deskewed": False,
    }
