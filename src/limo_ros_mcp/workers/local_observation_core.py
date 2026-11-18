"""Build a bounded, non-authoritative ROS sensor bundle. Python 2 compatible."""

import base64
import math

try:
    STRING_TYPES = (basestring,)
    INTEGER_TYPES = (int, long)
except NameError:
    STRING_TYPES = (str,)
    INTEGER_TYPES = (int,)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected finite numeric value")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError("expected finite numeric value")
    return value


def _frame(value):
    if not isinstance(value, STRING_TYPES) or not value or len(value) > 128:
        raise ValueError("invalid frame")
    return value


def _matrix(raw):
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("invalid transform")
    if any(not isinstance(row, (list, tuple)) or len(row) != 4 for row in raw):
        raise ValueError("invalid transform")
    return [[_number(value) for value in row] for row in raw]


def make_bundle(scan, depth, info, scan_matrix, depth_matrix, now):
    """Return fixed-format measurements, never clearance or authorization."""
    current = _number(now)
    scan_stamp, depth_stamp = _number(scan["stamp"]), _number(depth["stamp"])
    if any(
        stamp <= 0 or stamp > current or current - stamp > 2.0
        for stamp in (scan_stamp, depth_stamp)
    ):
        raise ValueError("sensor observations are stale or in the future")
    if abs(scan_stamp - depth_stamp) > 0.1:
        raise ValueError("scan and depth are not time aligned")
    scan_frame, depth_frame = _frame(scan["frame_id"]), _frame(depth["frame_id"])
    width, height, step = depth["width"], depth["height"], depth["step"]
    if (
        type(width) not in INTEGER_TYPES
        or type(height) not in INTEGER_TYPES
        or type(step) not in INTEGER_TYPES
        or not 1 <= width <= 1280
        or not 1 <= height <= 720
    ):
        raise ValueError("invalid image dimensions")
    sizes = {"16UC1": 2, "32FC1": 4}
    encoding = depth["encoding"]
    if encoding not in sizes or type(depth["is_bigendian"]) is not bool:
        raise ValueError("unsupported depth image encoding")
    data = depth["data"]
    if (
        not isinstance(data, (bytes, bytearray))
        or step < width * sizes[encoding]
        or step * height > 2 * 1024 * 1024
        or len(data) != step * height
    ):
        raise ValueError("depth data is incomplete or exceeds size limit")
    if info["frame_id"] != depth_frame or info["width"] != width or info["height"] != height:
        raise ValueError("depth image and CameraInfo frame or dimensions mismatch")
    info_stamp = _number(info['stamp'])
    if info_stamp <= 0 or info_stamp > current or abs(info_stamp - depth_stamp) > .1:
        raise ValueError('depth image and CameraInfo timestamps mismatch')
    if (
        type(info["binning_x"]) not in INTEGER_TYPES
        or type(info["binning_y"]) not in INTEGER_TYPES
        or info["binning_x"] not in (0, 1)
        or info["binning_y"] not in (0, 1)
    ):
        raise ValueError("binned camera calibration requires explicit adjustment")
    roi = info["roi"]
    if (
        not isinstance(roi, dict)
        or roi.get("do_rectify") is not False
        or any(
            type(roi.get(name)) not in INTEGER_TYPES
            for name in ("x_offset", "y_offset", "width", "height")
        )
        or roi["x_offset"] != 0
        or roi["y_offset"] != 0
        or (roi["width"], roi["height"]) not in ((0, 0), (width, height))
    ):
        raise ValueError("cropped or rectified ROI requires explicit calibration")
    k, d, rect, p = info["K"], info["D"], info["R"], info["P"]
    if len(k) != 9 or len(d) < 4 or len(rect) != 9 or len(p) != 12:
        raise ValueError("incomplete camera calibration")
    k = [_number(x) for x in k]
    d = [_number(x) for x in d]
    rect = [_number(x) for x in rect]
    p = [_number(x) for x in p]
    if any(abs(x) > 1e-9 for x in d):
        raise ValueError("depth calibration distortion is not zero")
    identity = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    if any(abs(rect[i] - identity[i]) > 1e-6 for i in range(9)):
        raise ValueError("depth image is not rectified")
    if (
        k[0] <= 0
        or k[4] <= 0
        or abs(k[8] - 1) > 1e-6
        or any(abs(k[i]) > 1e-6 for i in (1, 3, 6, 7))
        or any(abs(p[i]) > 1e-6 for i in (1, 3, 4, 7, 8, 9, 11))
        or abs(p[10] - 1) > 1e-6
        or any(abs(k[a] - p[b]) > 1e-6 for a, b in ((0, 0), (2, 2), (4, 5), (5, 6)))
    ):
        raise ValueError("invalid or inconsistent pinhole calibration")
    angle_min = _number(scan["angle_min"])
    angle_increment = _number(scan["angle_increment"])
    min_range, max_range = _number(scan["range_min"]), _number(scan["range_max"])
    ranges = scan["ranges"]
    if (
        not isinstance(ranges, (list, tuple))
        or not 1 <= len(ranges) <= 10000
        or angle_increment == 0
        or not 0 < min_range < max_range <= 100
    ):
        raise ValueError("invalid laser scan")
    sanitized = []
    for raw in ranges:
        try:
            sanitized.append(_number(raw))
        except ValueError:
            sanitized.append(None)
    return {
        "scan": {
            "stamp": scan_stamp,
            "frame_id": scan_frame,
            "angle_min": angle_min,
            "angle_increment": angle_increment,
            "range_min": min_range,
            "range_max": max_range,
            "ranges": sanitized,
            "transform": {
                "source": scan_frame,
                "target": "base_link",
                "static": False,
                "stamp": scan_stamp,
                "matrix": _matrix(scan_matrix),
            },
        },
        "depth": {
            "stamp": depth_stamp,
            "frame_id": depth_frame,
            "width": width,
            "height": height,
            "step": step,
            "encoding": encoding,
            "is_bigendian": depth["is_bigendian"],
            "data_base64": base64.b64encode(data).decode("ascii"),
            "camera_info": {
                "frame_id": depth_frame,
                    "stamp": info_stamp,
                "rectified": True,
                "width": width,
                "height": height,
                "fx": p[0],
                "fy": p[5],
                "cx": p[2],
                "cy": p[6],
            },
            "transform": {
                "source": depth_frame,
                "target": "base_link",
                "static": False,
                "stamp": depth_stamp,
                "matrix": _matrix(depth_matrix),
            },
        },
        "calibration_verified": False,
        "usable_for_real_execution": False,
    }
