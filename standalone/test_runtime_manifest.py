"""Version evidence records file digests, never file contents."""
import os

from runtime_manifest import capture_files, same_files, same_processes


def test_file_manifest_detects_runtime_change_without_recording_contents(tmpdir):
    path = tmpdir.join('controller.py')
    path.write('private contents')
    first = capture_files({'controller': str(path)})
    assert len(first['controller']['sha256']) == 64
    assert 'private contents' not in str(first)
    assert same_files(first, {'controller': str(path)})
    path.write('changed contents')
    assert not same_files(first, {'controller': str(path)})


def test_missing_files_are_explicit_and_not_equal_to_success(tmpdir):
    path = str(tmpdir.join('missing'))
    first = capture_files({'controller': path})
    assert first['controller']['available'] is False
    assert not same_files(first, {'controller': path})


def test_process_comparison_requires_available_matching_identity():
    process = {'available': True, 'pid': 10, 'start_ticks': 20, 'sha256': 'a', 'executable': '/tmp/driver'}
    assert same_processes([process], [dict(process)]) is True
    changed = dict(process, start_ticks=21)
    assert same_processes([process], [changed]) is False
    assert same_processes([], []) is None
