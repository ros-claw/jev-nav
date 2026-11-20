#!/usr/bin/env python2
"""Combine two live ROS sensing processes with an in-memory relay gate.

This is read-only: it never starts the ROS relay, sends model requests, opens a
serial port or publishes velocity. A gate acceptance is hypothetical only;
vehicle state, pose continuity, swept clearance and physical stopping are not
verified by this program.
"""
from __future__ import division, print_function

import argparse
import collections
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

from frame_mux import FrameMux
from frame_pipe import FrameReader
from nav_timeline import Timeline
from relay import RelayGate, _monotonic
from runtime_manifest import capture_files, capture_process, same_files, same_processes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--observe-pid', action='append', type=int, default=[])
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 60 or len(args.observe_pid) > 8 or any(pid <= 1 for pid in args.observe_pid):
        parser.error('seconds or observed process IDs outside bounds')
    return args


def _open_private(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w')


def _spawn(stream, prefix, seconds):
    read_fd = write_fd = output = process = None
    try:
        read_fd, write_fd = os.pipe()
        fcntl.fcntl(read_fd, fcntl.F_SETFD, fcntl.fcntl(read_fd, fcntl.F_GETFD) | fcntl.FD_CLOEXEC)
        fcntl.fcntl(write_fd, fcntl.F_SETFD, fcntl.fcntl(write_fd, fcntl.F_GETFD) & ~fcntl.FD_CLOEXEC)
        output = _open_private(prefix + '.' + stream + '.log')
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'live_independent_probe.py')
        command = [sys.executable, '-u', script, '--stream', stream, '--seconds', str(seconds + 5.),
                   '--timeline', prefix + '.' + stream + '.jsonl', '--events-fd', str(write_fd)]
        # The parent has not created threads or opened robot devices. The
        # read ends are close-on-exec; only this private write end is needed.
        process = subprocess.Popen(command, stdout=output, stderr=output, close_fds=False)
        os.close(write_fd)
        write_fd = None
        output.close()
        output = None
        reader = FrameReader(read_fd, stream)
        read_fd = None
        return process, reader
    except BaseException:
        try:
            if process is not None:
                _stop_child(process)
        finally:
            if output is not None:
                output.close()
            for fd in (read_fd, write_fd):
                if fd is not None:
                    os.close(fd)
        raise


def _stop_child(process):
    if process.poll() is None:
        try:
            process.send_signal(signal.SIGINT)
        except OSError:
            pass
    deadline = _monotonic() + 3.
    while process.poll() is None and _monotonic() < deadline:
        time.sleep(.02)
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    return process.wait()


def main(argv=None):
    args = parse_args(argv)
    children, readers = {}, {}
    timeline = None
    counts = collections.Counter()
    fault = None
    gate = None
    sequence = accepted = 0
    first_command = last_command = None
    longest = 0.
    root = os.path.dirname(os.path.abspath(__file__))
    names = ('multiprocess_probe.py', 'live_independent_probe.py', 'independent_worker.py',
             'independent_perception.py', 'processed_evidence.py', 'frame_pipe.py', 'frame_mux.py',
             'explore.py', 'forward_trial.py', 'policy.py', 'projected_depth_gate.py',
             'depth_corridor.py', 'observation_guards.py', 'native_scan_loader.py',
             'nav_timeline.py', 'relay.py', 'runtime_manifest.py')
    files = {name: os.path.join(root, name) for name in names}
    files['scan_fusion'] = os.path.join(os.path.dirname(root), 'worker', 'limo_scan_fusion.py')
    files['mapless_worker'] = os.path.join(os.path.dirname(root), 'worker', 'limo_mapless_worker.py')
    files['python'] = sys.executable
    from native_scan_loader import load_native_scan
    native = load_native_scan()
    if native is not None:
        files['native_scan'] = native.__file__
    captured_files = capture_files(files)
    manifest = {'files': captured_files, 'observed_processes': [capture_process(pid) for pid in args.observe_pid],
                'native_accelerator_loaded': native is not None,
                'note': 'process and file identities are snapshots, not deployment or physical-stop proof'}
    with _open_private(args.output_prefix + '.manifest.json') as manifest_file:
        json.dump(manifest, manifest_file, sort_keys=True, allow_nan=False)
    try:
        for stream in ('scan', 'depth'):
            children[stream], readers[stream] = _spawn(stream, args.output_prefix, args.seconds)
        timeline = Timeline(args.output_prefix + '.mux.jsonl', 'mux-' + str(int(time.time())))
        mux = FrameMux(max_processing_lag=.2)
        started = _monotonic()
        next_tick = started
        while _monotonic() - started < args.seconds + 3.:
            now = _monotonic()
            for stream in ('scan', 'depth'):
                rows, error = readers[stream].poll()
                for row in rows:
                    if not mux.accept(row):
                        fault = 'invalid_sensor_event'
                        break
                    fields = {'clock_domain': 'monotonic', 'stage': stream + '_' + row['kind'],
                              'sequence': row['sequence'], 'epoch': row['epoch'],
                              'sensor_acquired_at': row['acquired_at'], 'sensor_received_at': row['received_at']}
                    if row['kind'] == 'result':
                        fields.update({'perception_end': row['completed_at'], 'valid_until': row['valid_until'],
                                       'status': row['status'], 'reason': row['reason']})
                    if not timeline.record('sensor', _monotonic(), fields):
                        mux.channel_failed(stream)
                        fault = 'timeline_incomplete'
                        break
                    if row['kind'] == 'result' and row['status'] == 'obstacle':
                        fault = row['reason']
                        break
                if fault is not None:
                    break
                if error is not None or children[stream].poll() is not None:
                    prior = mux.read_current(_monotonic)
                    fault = prior['reason'] if prior['state'] == 'fault' else error or 'producer_exit'
                    mux.channel_failed(stream)
                    break
            state = mux.read_current(_monotonic)
            now = state['checked_at']
            counts[state['state'] + ':' + state['reason']] += 1
            if gate is not None:
                gate.command(now)
                if gate.stopped:
                    longest = max(longest, last_command - first_command)
                    gate = None
            if state['state'] != 'ready' or fault is not None:
                if gate is not None:
                    gate.stop(state['reason'])
                    longest = max(longest, last_command - first_command)
                    gate = None
            elif now >= next_tick:
                if gate is None:
                    gate = RelayGate(now, max_motion_sec=30., require_intent_deadline=True)
                    sequence = 0
                    first_command = now
                frame = {'sequence': sequence, 'stamp': now, 'linear': .02, 'angular': 0.,
                         'expires_at': min(state['valid_until'], now + .16)}
                if gate.accept(frame, now):
                    sequence += 1
                    accepted += 1
                    last_command = now
                    if not timeline.record('control', now, {'clock_domain': 'monotonic',
                                           'stage': 'hypothetical_command', 'linear': .02, 'angular': 0.,
                                           'command_expires_at': frame['expires_at'], 'sequence': sequence - 1}):
                        fault = 'timeline_incomplete'
                else:
                    fault = 'hypothetical_relay_fault'
                next_tick = now + .05
            if not timeline.record('control', now, {'clock_domain': 'monotonic', 'stage': 'mux_check',
                                   'status': state['state'], 'reason': state['reason'],
                                   'pending': state['pending'], 'valid_until': state['valid_until']}):
                fault = 'timeline_incomplete'
            if fault is not None or state['state'] == 'fault':
                fault = fault or state['reason']
                break
            time.sleep(.01)
        if gate is not None:
            gate.stop('probe_finished')
            longest = max(longest, last_command - first_command)
    finally:
        exits = {}
        for stream, process in children.items():
            try:
                exits[stream] = _stop_child(process)
            except Exception:
                exits[stream] = 'cleanup_error'
                fault = 'cleanup_error'
        for reader in readers.values():
            reader.close()
        complete = timeline.close(5.) if timeline is not None else False
    hashes_match = same_files(captured_files, files)
    after_processes = [capture_process(pid) for pid in args.observe_pid]
    processes_match = same_processes(manifest['observed_processes'], after_processes)
    with _open_private(args.output_prefix + '.end-manifest.json') as manifest_file:
        json.dump({'observed_processes': after_processes, 'file_hashes_match_start_end': hashes_match,
                   'processes_match_start_end': processes_match}, manifest_file, sort_keys=True, allow_nan=False)
    print(json.dumps({'mode': 'read_only_multiprocess_probe', 'motion_requested': False,
                      'odom_continuity_checked': False, 'base_status_checked': False,
                      'counts': dict(counts), 'fault': fault, 'hypothetical_commands': accepted,
                      'longest_hypothetical_segment_sec': longest, 'child_exits': exits,
                      # The mux writer flushed its own events. Child shutdown
                      # can produce later events recorded only in child logs.
                      'timeline_complete': complete, 'mux_writer_complete': complete,
                      'shutdown_sensor_events_drained_by_mux': False,
                      'file_hashes_match_start_end': hashes_match,
                      'processes_match_start_end': processes_match}, sort_keys=True))
    return 0 if complete and fault is None and hashes_match and processes_match is not False else 1


if __name__ == '__main__':
    sys.exit(main())
