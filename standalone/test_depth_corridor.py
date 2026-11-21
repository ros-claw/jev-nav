"""Offline projection and unknown-space tests for depth corridor diagnostics."""

import copy
import math

import numpy as np
import pytest


def diagnose(*args):
    try:
        from depth_corridor import diagnose_depth_corridor
    except ImportError:
        pytest.fail("depth corridor diagnostics are not implemented", pytrace=False)
    return diagnose_depth_corridor(*args)


def intrinsics(width=1, height=1, fx=1.0, fy=1.0, cx=0.0, cy=0.0):
    return {"width": width, "height": height, "fx": fx, "fy": fy,
            "cx": cx, "cy": cy, "rectified": True}


def transform(x=0.0, y=0.0, z=0.25):
    # optical right/down/forward -> base left/up/forward, plus translation
    return [[0.0, 0.0, 1.0, x], [-1.0, 0.0, 0.0, y],
            [0.0, -1.0, 0.0, z], [0.0, 0.0, 0.0, 1.0]]


def config():
    return {"forward_min_m": 0.1, "forward_max_m": 2.0, "half_width_m": 1.5,
            "ground_z_m": 0.0, "ground_tolerance_m": 0.0625,
            "obstacle_min_height_m": 0.125, "obstacle_max_height_m": 0.5,
            "min_depth_m": 0.125, "max_depth_m": 10.0, "sample_stride": 1}


def run(array, camera=None, matrix=None, limits=None, encoding="32FC1"):
    h, w = array.shape
    return diagnose(array, encoding, camera if camera is not None else intrinsics(w, h),
                    matrix if matrix is not None else transform(),
                    limits if limits is not None else config())


def test_optical_projection_applies_rotation_translation_and_lateral_sign():
    image = np.array([[float("nan"), 1.0]], dtype=np.float32)
    result = run(image, intrinsics(2, 1), transform(x=0.25))
    assert result["status"] == "observed"
    assert result["obstacle_count"] == 1
    point = result["nearest_obstacle"]
    assert point["pixel_u"] == 1 and point["pixel_v"] == 0
    assert point["x_m"] == pytest.approx(1.25)
    assert point["y_m"] == pytest.approx(-1.0)
    assert point["z_m"] == pytest.approx(0.25)
    assert point["optical_depth_m"] == pytest.approx(1.0)
    assert point["planar_distance_m"] == pytest.approx(math.hypot(1.25, 1.0))


def test_native_depth_preserves_projected_counts_and_nearest_measurements(monkeypatch):
    import depth_corridor
    native = depth_corridor._native_scan
    if native is None or not hasattr(native, 'fast_depth_corridor'):
        pytest.skip('optional native depth kernel is unavailable')
    image = np.random.RandomState(971).randint(0, 3600, (400, 640)).astype(np.uint16)
    camera = intrinsics(640, 400, fx=360., fy=360., cx=320., cy=200.)
    limits = dict(config(), sample_stride=4, min_depth_m=.3, max_depth_m=3.)
    actual = run(image, camera=camera, limits=limits, encoding='16UC1')
    monkeypatch.setattr(depth_corridor, '_native_scan', None)
    expected = run(image, camera=camera, limits=limits, encoding='16UC1')
    for key in ('nearest_ground', 'nearest_obstacle'):
        actual_point, expected_point = actual.pop(key), expected.pop(key)
        assert actual_point['pixel_u'] == expected_point['pixel_u']
        assert actual_point['pixel_v'] == expected_point['pixel_v']
        assert actual_point['planar_distance_m'] == pytest.approx(expected_point['planar_distance_m'])
    assert actual == expected


def test_nearest_obstacle_uses_planar_distance_not_optical_axis_depth():
    image = np.array([[float("nan"), float("nan"), 0.6, float("nan"), 0.5]], dtype=np.float32)
    point = run(image, intrinsics(5, 1, cx=2.0))["nearest_obstacle"]
    assert point["pixel_u"] == 2
    assert point["optical_depth_m"] == pytest.approx(0.6)
    assert point["planar_distance_m"] == pytest.approx(0.6)


def test_optical_pixel_down_projects_to_lower_base_height():
    image = np.array([[float("nan")], [1.0]], dtype=np.float32)
    result = run(image, intrinsics(1, 2, fy=2.0), transform(z=0.75))
    assert result["obstacle_count"] == 1
    assert result["nearest_obstacle"]["pixel_v"] == 1
    assert result["nearest_obstacle"]["z_m"] == pytest.approx(0.25)


@pytest.mark.parametrize("height,ground,obstacle,other", [
    (-0.0625, 1, 0, 0), (0.0625, 1, 0, 0),
    (-0.0626, 0, 0, 1), (0.0626, 0, 0, 1),
    (0.125, 0, 1, 0), (0.5, 0, 1, 0), (0.5001, 0, 0, 1),
])
def test_ground_and_obstacle_height_boundaries(height, ground, obstacle, other):
    result = run(np.array([[1.0]], dtype=np.float32), matrix=transform(z=height))
    assert result["ground_count"] == ground
    assert result["obstacle_count"] == obstacle
    assert result["unclassified_corridor_count"] == other


def test_invalid_and_maximum_depth_values_remain_unknown():
    image = np.array([[0.0, float("nan")], [float("inf"), 10.0]], dtype=np.float32)
    result = run(image)
    assert result["status"] == "no_valid_depth"
    assert result["sample_count"] == 4
    assert result["valid_count"] == 0 and result["invalid_count"] == 4
    assert result["nearest_obstacle"] is None
    assert result["unobserved_space"] is True
    assert result["executable"] is False
    assert result["geometric_clearance_verified"] is False


def test_depth_minimum_is_inclusive_and_maximum_is_unknown():
    limits = config()
    limits["min_depth_m"], limits["max_depth_m"] = 0.125, 2.0
    result = run(np.array([[0.125, 2.0]], dtype=np.float32), limits=limits)
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 1


def test_uint16_depth_is_interpreted_as_millimeters():
    image = np.array([[1000]], dtype=np.uint16)
    result = run(image, encoding="16UC1")
    assert result["obstacle_count"] == 1
    assert result["nearest_obstacle"]["optical_depth_m"] == pytest.approx(1.0)


def test_every_sample_is_counted_as_invalid_or_one_endpoint_class():
    image = np.array([[1.0, 0.0], [float("nan"), 1.0]], dtype=np.float32)
    result = run(image)
    assert result["sample_count"] == 4
    assert result["invalid_count"] == 2 and result["valid_count"] == 2
    assert result["obstacle_count"] == 1
    assert result["unclassified_corridor_count"] == 1
    assert result["ground_count"] == 0 and result["outside_corridor_count"] == 0


def test_valid_points_outside_corridor_do_not_imply_corridor_observation():
    result = run(np.array([[1.0]], dtype=np.float32), matrix=transform(x=-1.25))
    assert result["status"] == "no_corridor_samples"
    assert result["outside_corridor_count"] == 1
    assert result["unobserved_space"] is True
    assert result["executable"] is False


@pytest.mark.parametrize("offset,expected", [(-0.875, 1), (1.0, 1), (-0.876, 0), (1.001, 0)])
def test_forward_corridor_bounds_are_inclusive(offset, expected):
    limits = config()
    limits["forward_min_m"] = 0.125
    result = run(np.array([[1.0]], dtype=np.float32), matrix=transform(x=offset), limits=limits)
    assert result["obstacle_count"] == expected


def test_ground_only_is_not_a_clearance_certificate():
    result = run(np.array([[1.0]], dtype=np.float32), matrix=transform(z=0.0))
    assert result["ground_count"] == 1 and result["obstacle_count"] == 0
    assert result["nearest_ground"]["z_m"] == pytest.approx(0.0)
    assert result["executable"] is False
    assert result["geometric_clearance_verified"] is False
    assert result["unobserved_space"] is True
    assert "clear" not in result and "safe" not in result


def test_stride_skips_pixels_without_claiming_unobserved_space_is_clear():
    limits = config()
    limits["sample_stride"] = 2
    image = np.array([[3.0, 0.5, 3.0]], dtype=np.float32)
    result = run(image, limits=limits)
    assert result["sample_count"] == 2
    assert result["status"] == "no_corridor_samples" and result["obstacle_count"] == 0
    assert result["unobserved_space"] is True
    assert result["executable"] is False


@pytest.mark.parametrize("field,value", [
    ("fx", 0.0), ("fy", float("nan")), ("cx", float("inf")),
    ("rectified", False), ("width", 2),
])
def test_invalid_or_unconfirmed_intrinsics_fail_without_authority(field, value):
    camera = intrinsics()
    camera[field] = value
    result = run(np.array([[1.0]], dtype=np.float32), camera=camera)
    assert result["status"] == "invalid_input"
    assert result["executable"] is False


@pytest.mark.parametrize("kind", ["scale", "reflection", "last_row", "nan"])
def test_invalid_rigid_transform_is_rejected(kind):
    matrix = copy.deepcopy(transform())
    if kind == "scale":
        matrix[0][2] = 2.0
    elif kind == "reflection":
        matrix[1][0] = 1.0
    elif kind == "last_row":
        matrix[3][0] = 0.1
    else:
        matrix[0][3] = float("nan")
    result = run(np.array([[1.0]], dtype=np.float32), matrix=matrix)
    assert result["status"] == "invalid_input"
    assert result["executable"] is False


@pytest.mark.parametrize("field,value", [
    ("sample_stride", 0), ("sample_stride", True), ("half_width_m", -1.0),
    ("forward_max_m", 0.0), ("obstacle_min_height_m", 0.01),
    ("max_depth_m", float("inf")),
])
def test_invalid_corridor_or_depth_limits_are_rejected(field, value):
    limits = config()
    limits[field] = value
    result = run(np.array([[1.0]], dtype=np.float32), limits=limits)
    assert result["status"] == "invalid_input"
    assert result["executable"] is False


def test_mismatched_depth_encoding_and_array_type_are_rejected():
    assert run(np.array([[1000]], dtype=np.uint16), encoding="32FC1")["status"] == "invalid_input"
    assert run(np.array([[1.0]], dtype=np.float32), encoding="16UC1")["status"] == "invalid_input"
