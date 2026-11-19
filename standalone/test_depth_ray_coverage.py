"""Synthetic, offline section-ray diagnostics; no ROS, devices or network."""
import json

import numpy as np
import pytest
from depth_ray_coverage import diagnose_depth_ray_coverage


def fixture():
    image = np.full((3, 3), 2.0, dtype=np.float32)
    depth = {"array": image, "encoding": "32FC1", "stamp": 10.0,
             "frame_id": "camera_depth_optical_frame"}
    intrinsics = {"width": 3, "height": 3, "fx": 2.0, "fy": 2.0, "cx": 1.0, "cy": 1.0,
                  "rectified": True, "stamp": 10.0, "frame_id": "camera_depth_optical_frame"}
    transform = {"matrix": [[0.0, 0.0, 1.0, 0.1], [-1.0, 0.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0]],
                 "source_frame": "camera_depth_optical_frame", "target_frame": "base_link",
                 "source_stamp": 10.0, "reference_stamp": 10.1}
    lidar = {"frame_id": "base_link", "stamp": 10.0, "reference_stamp": 10.1,
             "rays": [{"start_base": (0.1, 0.0, 0.2), "hit_base": (2.1, 0.0, 0.2), "stamp": 10.0}]}
    section = {"expected_base_frame": "base_link", "expected_camera_frame": "camera_depth_optical_frame",
               "x_m": 1.1, "half_width_m": 0.5, "z_min_m": -0.3, "z_max_m": 0.7,
               "blocking_min_x_m": -0.18, "pixels": [(0, 1), (1, 1), (2, 1)],
               "min_depth_m": 0.3, "max_depth_m": 3.0, "near_margin_m": 0.05,
               "surface_tolerance_m": 0.0625, "free_margin_m": 0.125,
               "lidar_min_range_m": 0.1, "lidar_max_range_m": 5.0, "lidar_free_margin_m": 0.1}
    timing = {"now": 10.1, "reference_stamp": 10.1}
    return [depth, intrinsics, transform, lidar, section, timing]


def diagnose(args):
    return diagnose_depth_ray_coverage(*args)


def test_geometric_section_samples_distinguish_free_surface_occlusion_and_hits():
    args = fixture()
    image = args[0]["array"]
    image[1, 0], image[1, 1], image[1, 2] = 2.0, 0.5, 1.0
    result = diagnose(args)
    assert result["status"] == "diagnostic"
    assert result["executable"] is False and result["candidate"] is False
    assert result["continuous_coverage_verified"] is False and result["unobserved_space"] is True
    assert result["safe_merge_available"] is False
    depth = result["depth_samples"]
    assert [item["status"] for item in depth] == ["sampled_free", "occluded", "measured"]
    assert depth[0]["crossing_base_m"] == pytest.approx((1.1, 0.5, 0.2))
    assert depth[1]["crossing_base_m"] == pytest.approx((1.1, 0.0, 0.2))
    assert depth[2]["crossing_base_m"] == pytest.approx((1.1, -0.5, 0.2))
    assert result["lidar_samples"][0]["status"] == "sampled_free"
    # A depth hit before the section survives a conflicting lidar free ray.
    assert any(hit["source"] == "depth" and hit["point_base_m"] == pytest.approx((0.6, 0, 0.2))
               for hit in result["blocking_hits"])
    assert result["merge_reason"] == "between_rays_and_3d_unobserved"
    json.dumps(result, allow_nan=False)


def test_different_heights_are_kept_as_distinct_3d_samples():
    args = fixture()
    args[4]["pixels"] = [(1, 0), (1, 1)]
    result = diagnose(args)
    a, b = result["depth_samples"]
    assert a["status"] == b["status"] == "sampled_free"
    assert a["crossing_base_m"] == pytest.approx((1.1, 0.0, 0.7))
    assert b["crossing_base_m"] == pytest.approx((1.1, 0.0, 0.2))
    assert result["safe_merge_available"] is False
    assert "combined_max_gap_m" not in result


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), 0.35, 3.0])
def test_missing_near_and_out_of_range_depths_are_unknown(bad):
    args = fixture()
    args[4]["pixels"] = [(1, 1)]
    args[0]["array"][1, 1] = bad
    result = diagnose(args)
    assert result["depth_samples"][0]["status"] == "unknown"
    assert not [hit for hit in result["blocking_hits"] if hit["source"] == "depth"]
    assert result["unobserved_space"] is True


def test_16uc1_uint16_zero_and_finite_mm_with_nonnative_endian():
    args = fixture()
    args[0]["array"] = np.array([[0, 0, 0], [1500, 0, 1000], [0, 0, 0]], dtype=">u2")
    args[0]["encoding"] = "16UC1"
    result = diagnose(args)
    assert [sample["status"] for sample in result["depth_samples"]] == ["sampled_free", "unknown", "measured"]


def test_surface_and_free_margin_boundaries_remain_conservative():
    args = fixture()
    args[4]["pixels"] = [(1, 1)]
    for value, expected in ((1.0625, "measured"), (0.9375, "measured"),
                            (1.125, "unknown"), (1.126, "sampled_free"),
                            (0.937, "occluded")):
        args[0]["array"][1, 1] = value
        assert diagnose(args)["depth_samples"][0]["status"] == expected


def test_lidar_missing_occluded_and_measured_hits_are_not_free():
    args = fixture()
    args[3]["rays"] = [
        {"start_base": (0.1, 0.0, 0.2), "hit_base": (0.6, 0.0, 0.2), "stamp": 10.0},
        {"start_base": (0.1, 0.0, 0.2), "hit_base": (1.1, 0.0, 0.2), "stamp": 10.0},
        {"start_base": (0.1, 0.0, 0.2), "hit_base": (float("nan"), 0.0, 0.2), "stamp": 10.0},
        {"start_base": (0.1, 0.0, 0.2), "hit_base": (0.1, 0.0, 0.2), "stamp": 10.0},
    ]
    result = diagnose(args)
    assert [item["status"] for item in result["lidar_samples"]] == ["occluded", "measured", "unknown", "unknown"]
    assert len([hit for hit in result["blocking_hits"] if hit["source"] == "lidar"]) == 2
    json.dumps(result, allow_nan=False)


def test_outside_section_and_camera_near_field_remain_unknown():
    args = fixture()
    args[4]["pixels"] = [(0, 0), (1, 1)]
    args[4]["z_max_m"] = 0.3
    args[4]["x_m"] = 0.4  # Camera optical depth at section is exactly the 0.3 m minimum.
    result = diagnose(args)
    assert all(item["status"] == "unknown" for item in result["depth_samples"])
    assert result["unobserved_space"] is True


def test_camera_rays_use_full_rotated_and_translated_transform():
    args = fixture()
    args[4]["pixels"] = [(1, 1)]
    args[2]["matrix"][0][3] = 0.2
    args[2]["matrix"][1][3] = 0.25
    result = diagnose(args)
    sample = result["depth_samples"][0]
    assert sample["crossing_base_m"] == pytest.approx((1.1, 0.25, 0.2))
    assert sample["target_optical_depth_m"] == pytest.approx(0.9)
    assert sample["status"] == "sampled_free"
    # Turn optical forward leftward; it cannot reach a forward x section.
    args[2]["matrix"] = [[1., 0., 0., 0.1], [0., 0., 1., 0.0],
                           [0., -1., 0., 0.2], [0., 0., 0., 1.]]
    assert diagnose(args)["depth_samples"][0]["status"] == "unknown"


@pytest.mark.parametrize("problem", ["depth_stale", "lidar_stale", "future", "sensor_skew", "info_stale",
                                      "reference_stale", "tf_source_time", "tf_reference_time", "lidar_ref_time"])
def test_stale_future_or_unsynchronized_data_produce_no_coverage(problem):
    args = fixture()
    if problem == "depth_stale":
        args[0]["stamp"] = 9.0
        args[2]["source_stamp"] = 9.0
    elif problem == "lidar_stale":
        args[3]["stamp"] = 9.0
        args[3]["rays"][0]["stamp"] = 9.0
    elif problem == "future":
        args[0]["stamp"] = 10.2
        args[2]["source_stamp"] = 10.2
    elif problem == "sensor_skew":
        args[3]["stamp"] = 9.85
        args[3]["rays"][0]["stamp"] = 9.85
    elif problem == "info_stale":
        args[1]["stamp"] = 8.0
    elif problem == "reference_stale":
        args[5]["now"] = 10.2
    elif problem == "tf_source_time":
        args[2]["source_stamp"] = 9.99
    elif problem == "tf_reference_time":
        args[2]["reference_stamp"] = 10.0
    else:
        args[3]["reference_stamp"] = 10.0
    result = diagnose(args)
    assert result["status"] != "diagnostic"
    assert result["depth_samples"] == [] and result["lidar_samples"] == []
    assert result["candidate"] is False and result["unobserved_space"] is True


@pytest.mark.parametrize("problem", ["image_frame", "info_frame", "tf_source", "tf_target", "lidar_frame",
                                      "nonrigid", "reflection", "invalid_intrinsics", "distorted", "wrong_dtype",
                                      "duplicate_pixel", "pixel_outside", "bad_ray_stamp", "bad_depth_limits"])
def test_invalid_frame_calibration_image_or_configuration_is_rejected(problem):
    args = fixture()
    if problem == "image_frame":
        args[0]["frame_id"] = "wrong"
    elif problem == "info_frame":
        args[1]["frame_id"] = "wrong"
    elif problem == "tf_source":
        args[2]["source_frame"] = "wrong"
    elif problem == "tf_target":
        args[2]["target_frame"] = "wrong"
    elif problem == "lidar_frame":
        args[3]["frame_id"] = "wrong"
    elif problem == "nonrigid":
        args[2]["matrix"][0][2] = 2.0
    elif problem == "reflection":
        args[2]["matrix"][1][0] = 1.0
    elif problem == "invalid_intrinsics":
        args[1]["fx"] = 0.0
    elif problem == "distorted":
        args[1]["rectified"] = False
    elif problem == "wrong_dtype":
        args[0]["array"] = args[0]["array"].astype(np.float64)
    elif problem == "duplicate_pixel":
        args[4]["pixels"] = [(1, 1), (1, 1)]
    elif problem == "pixel_outside":
        args[4]["pixels"] = [(3, 1)]
    elif problem == "bad_ray_stamp":
        args[3]["rays"][0]["stamp"] = 9.9
    else:
        args[4]["max_depth_m"] = args[4]["min_depth_m"]
    result = diagnose(args)
    assert result["status"] == "invalid_input"
    assert result["depth_samples"] == [] and result["lidar_samples"] == []
    assert result["executable"] is False


def test_empty_or_missing_observations_cannot_claim_coverage():
    args = fixture()
    args[4]["pixels"] = []
    args[3]["rays"] = []
    result = diagnose(args)
    assert result["status"] == "diagnostic"
    assert result["depth_samples"] == [] and result["lidar_samples"] == []
    assert result["safe_merge_available"] is False and result["unobserved_space"] is True
