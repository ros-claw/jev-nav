import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "worker" / "limo_mapless_worker.py"
spec = importlib.util.spec_from_file_location("limo_mapless_worker", path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def request():
    return {
        "protocol": "rosclaw.limo.worker.v1",
        "operation": "MAPLESS_EXPLORE",
        "schema_version": "limo.explore.v1",
        "action_id": "explore-1",
        "body_id": "limo",
        "body_snapshot_hash": "a" * 64,
        "duration_sec": 10.0,
    }


def test_worker_accepts_only_exact_bounded_exploration_contract():
    assert worker.validate_request(request())["duration_sec"] == 10.0
    for extra in ({"duration_sec": 31}, {"body_id": "other"}, {"unexpected": 1}):
        proposal = request()
        proposal.update(extra)
        with pytest.raises(ValueError):
            worker.validate_request(proposal)


def test_laser_endpoints_are_transformed_before_direction_sectors():
    scan = worker.summarize_scan([2.0, 3.0, 4.0], -1.0, 1.0, 0.1, 12.0, 0.1, 0.0, 0.0)
    assert 3.09 < scan["front"] < 3.11
    assert scan["left"] > 4.0
    assert scan["right"] > 2.0


def test_depth_frame_and_camera_info_must_match():
    image = {"frame_id": "color_optical", "width": 640, "height": 400, "stamp": 10.0}
    info = {"frame_id": "color_optical", "width": 640, "height": 400, "stamp": 10.05}
    assert worker.depth_metadata_matches(image, info)
    for patch in ({"frame_id": ""}, {"width": 320}, {"stamp": 11.0}):
        invalid = dict(info)
        invalid.update(patch)
        assert not worker.depth_metadata_matches(image, invalid)


def test_stop_requires_multiple_new_fresh_odometry_samples():
    verifier = worker.StoppedOdomVerifier(stop_requested_at=10.0)
    assert not verifier.observe(0.0, 0.0, 0.0, observed_at=10.0, now=10.0)
    assert not verifier.observe(0.0, 0.0, 0.0, observed_at=10.1, now=10.1)
    assert not verifier.observe(0.1, 0.0, 0.0, observed_at=10.2, now=10.2)
    assert not verifier.observe(0.0, 0.0, 0.0, observed_at=10.3, now=10.3)
    assert not verifier.observe(0.0, 0.0, 0.0, observed_at=10.4, now=10.4)
    assert verifier.observe(0.0, 0.0, 0.0, observed_at=10.5, now=10.5)


def test_motion_evidence_counts_translation_and_wrapped_yaw():
    assert worker.motion_delta((0.0, 0.0, 3.13), (0.03, 0.04, -3.13))[0] == pytest.approx(0.05)
    assert worker.motion_delta((0.0, 0.0, 3.13), (0.0, 0.0, -3.13))[1] == pytest.approx(
        0.0231853, abs=1e-6
    )
