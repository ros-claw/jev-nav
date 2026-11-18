"""Pure validation for synchronized read-only model observations."""

import hashlib
import json
import math
import re
import struct
import zlib

try:
    NUMBER_TYPES = (int, long, float)  # noqa: F821 - ROS Melodic Python 2
    STRING_TYPES = (basestring,)  # noqa: F821
except NameError:
    NUMBER_TYPES = (int, float)
    STRING_TYPES = (str,)


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, NUMBER_TYPES):
        return False
    try:
        return not math.isnan(value) and not math.isinf(value)
    except (ValueError, OverflowError):
        return False


def _frame(value):
    return (
        isinstance(value, STRING_TYPES)
        and 1 <= len(value) <= 128
        and re.match(r"^/?[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*\Z", value) is not None
    )


def _depth_map(depth):
    clean = {}
    for name in ("left", "center", "right"):
        item = depth[name]
        ratio, minimum = item["valid_ratio"], item["minimum_m"]
        if (
            not _finite(ratio)
            or not 0 <= ratio <= 1
            or (minimum is not None and (not _finite(minimum) or minimum <= 0))
            or ((ratio == 0) != (minimum is None))
        ):
            raise ValueError("invalid depth statistics")
        clean[name] = {
            "valid_ratio": float(ratio),
            "minimum_m": None if minimum is None else float(minimum),
        }
    return clean


def _ray_sector(angle):
    if -math.pi / 2 - 1e-5 <= angle < -math.pi / 6:
        return "right", max(angle, -math.pi / 2)
    if -math.pi / 6 <= angle <= math.pi / 6:
        return "center", angle
    if math.pi / 6 < angle <= math.pi / 2 + 1e-5:
        return "left", min(angle, math.pi / 2)
    return None, angle


def _forward_view(scan, transform):
    bounds = {
        "right": (-math.pi / 2, -math.pi / 6),
        "center": (-math.pi / 6, math.pi / 6),
        "left": (math.pi / 6, math.pi / 2),
    }
    sectors = {
        name: {
            "total_rays": 0,
            "valid_rays": 0,
            "nearest_to_base_m": None,
            "nearest_bearing_from_base_rad": None,
            "_angles": [],
            "_missing": 0,
            "max_invalid_return_run_rad": 0.0,
        }
        for name in bounds
    }
    base_hits = {
        name: {"nearest_m": None, "bearing_rad": None}
        for name in ("right", "center", "left", "other")
    }
    rear_total = rear_valid = 0
    rear_nearest = None
    for index, distance in enumerate(scan["ranges"]):
        angle = scan["angle_min"] + index * scan["angle_increment"] + transform["yaw"]
        angle = math.atan2(math.sin(angle), math.cos(angle))
        name, sector_angle = _ray_sector(angle)
        valid = _finite(distance) and scan["range_min"] <= distance <= scan["range_max"]
        if valid:
            x = transform["x"] + distance * math.cos(angle)
            y = transform["y"] + distance * math.sin(angle)
            measured = math.hypot(x, y)
            bearing = math.atan2(y, x) if measured > 1e-9 else None
            if bearing is not None and -math.pi / 2 <= bearing < -math.pi / 6:
                base_name = "right"
            elif bearing is not None and -math.pi / 6 <= bearing <= math.pi / 6:
                base_name = "center"
            elif bearing is not None and math.pi / 6 < bearing <= math.pi / 2:
                base_name = "left"
            else:
                base_name = "other"
            hit = base_hits[base_name]
            if hit["nearest_m"] is None or measured < hit["nearest_m"]:
                hit["nearest_m"], hit["bearing_rad"] = measured, bearing
        if name is None:
            if abs(angle) >= 2.1:
                rear_total += 1
                if valid:
                    rear_valid += 1
                    rear_nearest = measured if rear_nearest is None else min(rear_nearest, measured)
            continue
        sector = sectors[name]
        sector["total_rays"] += 1
        if valid:
            sector["valid_rays"] += 1
            sector["_angles"].append(sector_angle)
            sector["_missing"] = 0
            if sector["nearest_to_base_m"] is None or measured < sector["nearest_to_base_m"]:
                sector["nearest_to_base_m"] = measured
                sector["nearest_bearing_from_base_rad"] = bearing
        else:
            sector["_missing"] += 1
            sector["max_invalid_return_run_rad"] = max(
                sector["max_invalid_return_run_rad"],
                sector["_missing"] * abs(scan["angle_increment"]),
            )
    for name, sector in sectors.items():
        del sector["_missing"]
        angles = sorted(sector.pop("_angles"))
        low, high = bounds[name]
        gaps = (
            (
                [angles[0] - low, high - angles[-1]]
                + [angles[i] - angles[i - 1] for i in range(1, len(angles))]
            )
            if angles
            else [high - low]
        )
        sector["max_unobserved_angle_rad"] = max(gaps)
        if not sector["total_rays"]:
            sector["max_invalid_return_run_rad"] = None
    sectors["rear_total_rays"], sectors["rear_valid_rays"] = rear_total, rear_valid
    sectors["rear_nearest_to_base_m"], sectors["base_hits"] = rear_nearest, base_hits
    return sectors


def scene_digest(png, snapshot):
    """Bind image bytes to canonical scene metadata; this is not authentication."""
    if not isinstance(png, bytes) or not isinstance(snapshot, dict):
        raise ValueError("invalid scene digest input")
    canonical = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return (
        "sha256:"
        + hashlib.sha256(b"limo-scene-v1\0" + hashlib.sha256(png).digest() + canonical).hexdigest()
    )


def validate_png(png):
    """Validate the bounded, noninterlaced grayscale/RGB PNG format emitted by cv2."""
    try:
        if (
            not isinstance(png, bytes)
            or len(png) > 4000000
            or not png.startswith(b"\x89PNG\r\n\x1a\n")
        ):
            raise ValueError("invalid PNG signature or size")
        offset = 8
        chunks = []
        dimensions = None
        seen_data = False
        data_closed = False
        ended = False
        for _ in range(1024):
            if offset + 12 > len(png):
                break
            length = struct.unpack("!I", png[offset : offset + 4])[0]
            end = offset + 12 + length
            if end > len(png):
                raise ValueError("truncated PNG chunk")
            kind = png[offset + 4 : offset + 8]
            payload = png[offset + 8 : offset + 8 + length]
            crc = struct.unpack("!I", png[offset + 8 + length : end])[0]
            if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
                raise ValueError("PNG checksum mismatch")
            if dimensions is None:
                if kind != b"IHDR" or length != 13:
                    raise ValueError("missing PNG header")
                width, height, bits, color, compression, filtering, interlace = struct.unpack(
                    "!IIBBBBB", payload
                )
                if (
                    not 1 <= width <= 1920
                    or not 1 <= height <= 1080
                    or width * height > 1920 * 1080
                    or bits not in (8, 16)
                    or color not in (0, 2, 4, 6)
                    or compression != 0
                    or filtering != 0
                    or interlace != 0
                ):
                    raise ValueError("unsupported PNG image")
                dimensions = (width, height, bits, color)
            elif kind == b"IHDR":
                raise ValueError("duplicate PNG header")
            elif kind == b"IDAT":
                if data_closed:
                    raise ValueError("non-contiguous PNG data")
                seen_data = True
                chunks.append(payload)
            elif kind == b"IEND":
                if length or not seen_data or end != len(png):
                    raise ValueError("invalid PNG end")
                ended = True
                break
            else:
                if seen_data:
                    data_closed = True
                if kind[0:1].isupper() and kind != b"PLTE":
                    raise ValueError("unsupported critical PNG chunk")
            offset = end
        if not ended or dimensions is None:
            raise ValueError("missing PNG data or end")
        width, height, bits, color = dimensions
        channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
        row_size = (width * channels * bits + 7) // 8 + 1
        expected = row_size * height
        decoder = zlib.decompressobj()
        raw = decoder.decompress(b"".join(chunks) + b"\0", expected + 1)
        if (
            len(raw) != expected
            or decoder.unconsumed_tail
            or decoder.unused_data != b"\0"
            or any(
                raw[index : index + 1] not in (b"\0", b"\1", b"\2", b"\3", b"\4")
                for index in range(0, expected, row_size)
            )
        ):
            raise ValueError("invalid PNG pixel stream")
        return True
    except (TypeError, struct.error, zlib.error, OverflowError) as exc:
        raise ValueError("invalid PNG image: " + str(exc))  # noqa: B904 - Python 2


def validate_snapshot(snapshot, now=None, max_age=3.0):
    """Reject malformed or incoherent advisory metadata at each process boundary."""
    try:
        if not isinstance(snapshot, dict) or snapshot["clock_domain"] != "ros_wall_epoch":
            raise ValueError("invalid scene clock domain")
        depth, laser, base, frames = (
            snapshot[key] for key in ("depth", "laser", "base", "sensor_frames")
        )
        stamps = [
            snapshot[key] for key in ("image_stamp", "laser_stamp", "status_stamp", "odom_stamp")
        ]
        stamps.append(depth["stamp"])
        if (
            not all(_finite(stamp) and stamp > 0 for stamp in stamps)
            or max(stamps) - min(stamps) > 0.300001
        ):
            raise ValueError("incoherent scene timestamps")
        if now is not None and (
            not _finite(now)
            or now <= 0
            or not _finite(max_age)
            or not 0 < max_age <= 30
            or any(not -0.25 <= now - stamp <= max_age for stamp in stamps)
        ):
            raise ValueError("stale scene snapshot")
        if (
            frames["transform_verified"] is not False
            or frames["camera_orientation_verified"] is not False
            or any(not _frame(frames[name]) for name in ("laser", "color", "depth"))
        ):
            raise ValueError("invalid scene frames")
        if not all(_finite(frames["laser_to_base"][name]) for name in ("x_m", "y_m", "yaw_rad")):
            raise ValueError("invalid laser transform")
        if (
            laser["forward_view_stamp"] != snapshot["laser_stamp"]
            or not isinstance(laser["forward_view"], dict)
            or not _finite(laser["usable_ratio"])
            or not 0 <= laser["usable_ratio"] <= 1
        ):
            raise ValueError("invalid laser coverage")
        total, valid = laser["total_rays"], laser["valid_rays"]
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or isinstance(valid, bool)
            or not isinstance(valid, int)
            or not 1 <= total <= 10000
            or not 0 <= valid <= total
            or abs(laser["usable_ratio"] - float(valid) / total) > 1e-8
        ):
            raise ValueError("invalid scan totals")
        increment, scan_time, scan_end = (
            laser[name] for name in ("time_increment_sec", "scan_time_sec", "scan_end_stamp")
        )
        timing_known = increment > 0 or total == 1
        if (
            not all(_finite(value) for value in (increment, scan_time))
            or not 0 <= increment <= 1
            or not 0 <= scan_time <= 10
            or (total - 1) * increment > 10
            or laser["deskewed"] is not False
            or laser["scan_timing"] != ("known" if timing_known else "unknown")
            or (
                timing_known
                and (
                    not _finite(scan_end)
                    or abs(scan_end - (snapshot["laser_stamp"] + (total - 1) * increment)) > 1e-6
                    or (now is not None and scan_end > now + 0.25)
                )
            )
            or (not timing_known and scan_end is not None)
        ):
            raise ValueError("invalid scan timing")
        view = laser["forward_view"]
        for name in ("left", "center", "right"):
            sector = view[name]
            rays, hits = sector["total_rays"], sector["valid_rays"]
            nearest, bearing = sector["nearest_to_base_m"], sector["nearest_bearing_from_base_rad"]
            missing = sector["max_invalid_return_run_rad"]
            if (
                not isinstance(sector, dict)
                or not _finite(sector["max_unobserved_angle_rad"])
                or not 0 <= sector["max_unobserved_angle_rad"] <= math.pi / 3 + 1e-6
                or isinstance(rays, bool)
                or not isinstance(rays, int)
                or isinstance(hits, bool)
                or not isinstance(hits, int)
                or not 0 <= hits <= rays <= total
                or sector["max_unobserved_angle_rad"] + 1e-8 < (math.pi / 3) / (hits + 1)
                or ((rays == 0) != (missing is None))
                or (missing is not None and (not _finite(missing) or missing < 0))
                or ((hits == 0) != (nearest is None))
                or (nearest is not None and (not _finite(nearest) or nearest < 0))
                or (bearing is not None and (not _finite(bearing) or abs(bearing) > math.pi + 1e-6))
                or (hits == 0 and bearing is not None)
                or (nearest is not None and nearest > 1e-9 and bearing is None)
            ):
                raise ValueError("invalid laser sector")
        rear_total, rear_valid, rear_distance = (
            view[name] for name in ("rear_total_rays", "rear_valid_rays", "rear_nearest_to_base_m")
        )
        if (
            isinstance(rear_total, bool)
            or not isinstance(rear_total, int)
            or isinstance(rear_valid, bool)
            or not isinstance(rear_valid, int)
            or not 0 <= rear_valid <= rear_total <= total
            or ((rear_valid == 0) != (rear_distance is None))
            or (rear_distance is not None and (not _finite(rear_distance) or rear_distance < 0))
        ):
            raise ValueError("invalid rear scan coverage")
        if (
            rear_total + sum(view[name]["total_rays"] for name in ("left", "center", "right"))
            > total
            or rear_valid + sum(view[name]["valid_rays"] for name in ("left", "center", "right"))
            > valid
        ):
            raise ValueError("overlapping laser coverage counts")
        hit_count = sum(
            view["base_hits"][name]["nearest_m"] is not None
            for name in ("right", "center", "left", "other")
        )
        if hit_count > valid or (valid > 0 and hit_count == 0):
            raise ValueError("base hits exceed valid rays")
        for name in ("right", "center", "left", "other"):
            hit = view["base_hits"][name]
            distance, bearing = hit["nearest_m"], hit["bearing_rad"]
            if (
                (distance is not None and (not _finite(distance) or distance < 0))
                or (bearing is not None and (not _finite(bearing) or abs(bearing) > math.pi + 1e-6))
                or (distance is None and bearing is not None)
                or (distance is not None and distance > 1e-9 and bearing is None)
            ):
                raise ValueError("invalid base-relative lidar hit")
            if bearing is not None:
                if -math.pi / 2 <= bearing < -math.pi / 6:
                    hit_sector = "right"
                elif -math.pi / 6 <= bearing <= math.pi / 6:
                    hit_sector = "center"
                elif math.pi / 6 < bearing <= math.pi / 2:
                    hit_sector = "left"
                else:
                    hit_sector = "other"
                if hit_sector != name:
                    raise ValueError("base-relative hit is in the wrong sector")
        for name in ("front_min_m", "left_min_m", "right_min_m", "rear_min_m"):
            value = laser[name]
            if value is not None and (not _finite(value) or value <= 0):
                raise ValueError("invalid laser distance")
            if valid == 0 and value is not None:
                raise ValueError("laser hit without valid rays")
        for field, sector_name in (
            ("front_min_m", "center"),
            ("left_min_m", "left"),
            ("right_min_m", "right"),
        ):
            if (laser[field] is None) != (view[sector_name]["valid_rays"] == 0):
                raise ValueError("laser minimum contradicts sector coverage")
        fields = ("error_code", "motion_mode", "battery_voltage", "linear_x_mps", "angular_z_radps")
        if not all(_finite(base[name]) for name in fields):
            raise ValueError("invalid base state")
        sectors = _depth_map(depth["sectors"])
        measured = any(sectors[name]["valid_ratio"] > 0 for name in sectors)
        if (
            depth["available"] is not True
            or depth["distance_measured"] is not measured
            or depth["units"] != "optical_depth_m"
        ):
            raise ValueError("invalid depth evidence")
        if "bands" in depth:
            for name in ("top", "middle", "bottom"):
                _depth_map(depth["bands"][name])
        return True
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("invalid scene snapshot: " + str(exc))  # noqa: B904 - Python 2


def make_snapshot(data, now, wall_now):
    """Build advisory data from one coherent header-time capture, never a clearance."""
    try:
        if (
            not _finite(now)
            or now <= 0
            or not _finite(wall_now)
            or wall_now <= 0
            or abs(now - wall_now) > 0.5
        ):
            raise ValueError("invalid capture time")
        stamps = data["stamps"]
        limits = {"color": 0.8, "depth": 0.8, "scan": 0.8, "odom": 0.8, "status": 1.2}
        for name, limit in limits.items():
            if (
                not _finite(stamps[name])
                or stamps[name] <= 0
                or not -0.1 <= now - stamps[name] <= limit
            ):
                raise ValueError("stale or invalid sensor stamp")
        if max(stamps[name] for name in limits) - min(stamps[name] for name in limits) > 0.3:
            raise ValueError("sensor header stamps differ too much")
        scan = data["scan"]
        angle_min, step, range_min, range_max = (
            scan[name] for name in ("angle_min", "angle_increment", "range_min", "range_max")
        )
        if (
            not all(_finite(v) for v in (angle_min, step, range_min, range_max))
            or step == 0
            or not 0 < range_min < range_max <= 100
        ):
            raise ValueError("invalid scan metadata")
        ranges = scan["ranges"]
        if not isinstance(ranges, (list, tuple)) or not 1 <= len(ranges) <= 10000:
            raise ValueError("invalid scan ranges")
        increment, scan_time = scan["time_increment"], scan["scan_time"]
        if (
            not _finite(increment)
            or not _finite(scan_time)
            or not 0 <= increment <= 1
            or not 0 <= scan_time <= 10
            or (len(ranges) - 1) * increment > 10
        ):
            raise ValueError("invalid scan timing")
        timing_known = increment > 0 or len(ranges) == 1
        scan_end = stamps["scan"] + (len(ranges) - 1) * increment if timing_known else None
        if scan_end is not None and scan_end > now + 0.25:
            raise ValueError("scan ends in the future")
        frames, transform = data["frames"], data["laser_tf"]
        if (
            any(not _frame(frames[name]) for name in ("laser", "color", "depth"))
            or transform["source"].lstrip("/") != frames["laser"].lstrip("/")
            or transform["target"].lstrip("/") != "base_link"
            or transform["verified"] is not False
            or not all(_finite(transform[name]) for name in ("stamp", "x", "y", "yaw"))
            or abs(transform["stamp"] - stamps["scan"]) > 1e-6
        ):
            raise ValueError("invalid laser transform or sensor frames")
        nearest = {"front": None, "left": None, "right": None, "rear": None}
        valid = 0
        for index, distance in enumerate(ranges):
            if not _finite(distance) or not range_min <= distance <= range_max:
                continue
            valid += 1
            angle = angle_min + index * step + transform["yaw"]
            angle = math.atan2(math.sin(angle), math.cos(angle))
            name, _ = _ray_sector(angle)
            name = "front" if name == "center" else name or "rear"
            if nearest[name] is None or distance < nearest[name]:
                nearest[name] = float(distance)
        base = data["base"]
        fields = ("error_code", "motion_mode", "battery_voltage", "linear_x_mps", "angular_z_radps")
        if not all(_finite(base[name]) for name in fields):
            raise ValueError("invalid base data")
        clean_depth = _depth_map(data["depth"])
        depth_result = {
            "available": True,
            "distance_measured": any(clean_depth[name]["valid_ratio"] > 0 for name in clean_depth),
            "stamp": float(stamps["depth"]),
            "units": "optical_depth_m",
            "sectors": clean_depth,
        }
        if "depth_bands" in data:
            depth_result["bands"] = {
                name: _depth_map(data["depth_bands"][name]) for name in ("top", "middle", "bottom")
            }
        snapshot = {
            "clock_domain": "ros_wall_epoch",
            "image_stamp": float(stamps["color"]),
            "laser_stamp": float(stamps["scan"]),
            "status_stamp": float(stamps["status"]),
            "odom_stamp": float(stamps["odom"]),
            "laser": {
                "front_min_m": nearest["front"],
                "left_min_m": nearest["left"],
                "right_min_m": nearest["right"],
                "rear_min_m": nearest["rear"],
                "usable_ratio": float(valid) / len(ranges),
                "total_rays": len(ranges),
                "valid_rays": valid,
                "time_increment_sec": increment,
                "scan_time_sec": scan_time,
                "scan_end_stamp": float(scan_end) if scan_end is not None else None,
                "scan_timing": "known" if timing_known else "unknown",
                "deskewed": False,
                "sector_frame": "laser_rays_oriented_by_unverified_base_tf",
                "forward_view": _forward_view(scan, transform),
                "forward_view_stamp": float(stamps["scan"]),
            },
            "base": {name: base[name] for name in fields},
            "depth": depth_result,
            "sensor_frames": {
                "laser": frames["laser"],
                "color": frames["color"],
                "depth": frames["depth"],
                "transform_verified": False,
                "laser_to_base": {
                    "x_m": transform["x"],
                    "y_m": transform["y"],
                    "yaw_rad": transform["yaw"],
                },
                "camera_orientation_verified": False,
            },
        }
        validate_snapshot(snapshot)
        return snapshot
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("invalid scene data: " + str(exc))  # noqa: B904 - Python 2 ROS worker
