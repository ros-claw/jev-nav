"""The shadow runner reports heartbeat delays without implying robot motion."""

from shadow_live import timing_statistics


def test_timing_summary_counts_only_positive_heartbeat_intervals():
    result = timing_statistics([0.0, 0.04, 0.05, 0.07, 0.2])
    assert result['samples'] == 4
    assert result['median_sec'] == 0.06
    assert result['p95_sec'] == 0.2
    assert result['maximum_sec'] == 0.2


def test_timing_summary_handles_no_heartbeats():
    assert timing_statistics([])['median_sec'] is None
