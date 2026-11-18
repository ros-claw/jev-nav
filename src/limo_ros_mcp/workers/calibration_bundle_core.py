"""Bounded mismatched-metadata research bundle; never safe for motion (Python 2)."""

import base64

from local_observation_core import INTEGER_TYPES, STRING_TYPES, _frame, _matrix, _number


def _rigid(raw):
    matrix = _matrix(raw)
    if any(abs(matrix[3][i] - target) > 1e-6 for i, target in enumerate((0, 0, 0, 1))):
        raise ValueError("invalid transform")
    for i in range(3):
        for j in range(3):
            dot = sum(matrix[k][i] * matrix[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-5:
                raise ValueError("nonorthogonal transform")
    a = matrix
    determinant = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )
    if abs(determinant - 1.0) > 1e-5:
        raise ValueError("reflected transform")
    return matrix


def make_diagnostic(scan, depth, info, scan_to_camera, now, scan_to_base=None):
    """Preserve source frame mismatch; TF and calibration remain unverified."""
    current = _number(now)
    scan_stamp, depth_stamp, info_stamp = map(
        _number, (scan["stamp"], depth["stamp"], info["stamp"])
    )
    if any(
        stamp <= 0 or stamp > current or current - stamp > 2
        for stamp in (scan_stamp, depth_stamp, info_stamp)
    ):
        raise ValueError("stale or future sensor data")
    if max(scan_stamp, depth_stamp, info_stamp) - min(scan_stamp, depth_stamp, info_stamp) > 0.1:
        raise ValueError("sensor timestamps differ")
    scan_frame, image_frame = _frame(scan["frame_id"]), _frame(depth["frame_id"])
    if image_frame != "camera_color_optical_frame":
        raise ValueError("unexpected claimed camera optical frame")
    info_frame = info["frame_id"]
    if not isinstance(info_frame, STRING_TYPES) or len(info_frame) > 128:
        raise ValueError("invalid CameraInfo frame")
    width, height, step = depth["width"], depth["height"], depth["step"]
    sizes = {"16UC1": 2, "32FC1": 4}
    if (
        any(type(x) not in INTEGER_TYPES for x in (width, height, step))
        or not 1 <= width <= 1280
        or not 1 <= height <= 720
        or depth["encoding"] not in sizes
        or type(depth["is_bigendian"]) is not bool
        or step < width * sizes[depth["encoding"]]
        or step * height > 2 * 1024 * 1024
        or not isinstance(depth["data"], (bytes, bytearray))
        or len(depth["data"]) != step * height
    ):
        raise ValueError("invalid bounded depth image")
    if any(
        type(info[key]) not in INTEGER_TYPES or not 1 <= info[key] <= 2048
        for key in ("width", "height")
    ):
        raise ValueError("invalid CameraInfo dimensions")
    info_arrays = {}
    for key, length in (("K", 9), ("R", 9), ("P", 12)):
        if not isinstance(info[key], (list, tuple)) or len(info[key]) != length:
            raise ValueError("invalid CameraInfo matrix")
        info_arrays[key] = [_number(value) for value in info[key]]
    if not isinstance(info["D"], (list, tuple)) or not 1 <= len(info["D"]) <= 14:
        raise ValueError("invalid distortion metadata")
    info_arrays["D"] = [_number(value) for value in info["D"]]
    distortion = info["distortion_model"]
    if not isinstance(distortion, STRING_TYPES) or len(distortion) > 64:
        raise ValueError("invalid distortion model")
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
            "stamp": scan_stamp,
            "frame_id": scan_frame,
            "angle_min": angle,
            "angle_increment": increment,
            "range_min": minimum,
            "range_max": maximum,
            "time_increment": time_increment,
            "scan_time": scan_time,
            "ranges": clean,
        },
        "depth": {
            "stamp": depth_stamp,
            "frame_id": image_frame,
            "width": width,
            "height": height,
            "step": step,
            "encoding": depth["encoding"],
            "is_bigendian": depth["is_bigendian"],
            "data_base64": base64.b64encode(depth["data"]).decode("ascii"),
        },
        "camera_info": {
            "stamp": info_stamp,
            "frame_id": info_frame,
            "width": info["width"],
            "height": info["height"],
            "distortion_model": distortion,
            "frame_matches_image": info_frame == image_frame,
            "dimensions_match_image": (info["width"], info["height"]) == (width, height),
            "matrices": info_arrays,
        },
        "transform": {
            "source": scan_frame,
            "target": image_frame,
            "stamp": scan_stamp,
            "matrix": _rigid(scan_to_camera),
        },
        "base_transform": (
            {
                "source": scan_frame,
                "target": "base_link",
                "stamp": scan_stamp,
                "matrix": _rigid(scan_to_base),
            }
            if scan_to_base is not None
            else None
        ),
        "diagnostic_only": True,
        "calibration_verified": False,
        "transform_verified": False,
        "usable_for_real_execution": False,
        "deskewed": False,
    }
