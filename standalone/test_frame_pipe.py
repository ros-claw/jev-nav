"""Private nonblocking processed-frame pipe; no ROS or motor I/O."""
import json
import os

from frame_pipe import FrameReader, FrameWriter, PipeTimeline


def ingress():
    return {'kind': 'ingress', 'stream': 'scan', 'sequence': 0, 'epoch': 1,
            'acquired_at': 10., 'received_at': 10.02}


def result(status='unknown'):
    return {'kind': 'result', 'stream': 'scan', 'sequence': 0, 'epoch': 1,
            'acquired_at': 10., 'received_at': 10.02, 'completed_at': 10.08,
            'valid_until': 10.25 if status == 'candidate' else None,
            'status': status, 'reason': 'sampled_forward_corridor'}


def test_pipe_preserves_timestamped_ingress_and_completion():
    read_fd, write_fd = os.pipe()
    reader, writer = FrameReader(read_fd, 'scan'), FrameWriter(write_fd, 'scan')
    try:
        assert writer.send(ingress()) and writer.send(result())
        rows, fault = reader.poll()
        assert rows == [ingress(), result()] and fault is None
    finally:
        writer.close()
        reader.close()


def test_malformed_frame_and_eof_are_faults_not_stale_results():
    read_fd, write_fd = os.pipe()
    reader = FrameReader(read_fd, 'scan')
    os.write(write_fd, b'{bad}\n')
    assert reader.poll()[1] == 'invalid_frame'
    os.close(write_fd)
    reader.close()
    read_fd, write_fd = os.pipe()
    reader = FrameReader(read_fd, 'scan')
    os.close(write_fd)
    assert reader.poll()[1] == 'producer_eof'
    reader.close()


def test_writer_rejects_payload_and_out_of_order_times():
    read_fd, write_fd = os.pipe()
    writer = FrameWriter(write_fd, 'scan')
    bad = ingress()
    bad['authorization'] = 'secret'
    assert not writer.send(bad)
    assert not writer.send(ingress())
    os.close(read_fd)


def test_partial_frame_does_not_become_a_complete_result():
    read_fd, write_fd = os.pipe()
    reader = FrameReader(read_fd, 'scan')
    wire = (json.dumps(ingress()) + '\n').encode('ascii')
    os.write(write_fd, wire[:10])
    assert reader.poll() == ([], None)
    os.write(write_fd, wire[10:])
    assert reader.poll() == ([ingress()], None)
    os.close(write_fd)
    reader.close()


def test_timeline_exports_only_sensor_metadata_and_closes_on_worker_fault():
    class Sink(object):
        def record(self, *_args):
            return True
    read_fd, write_fd = os.pipe()
    reader = FrameReader(read_fd, 'scan')
    pipe = PipeTimeline(Sink(), FrameWriter(write_fd, 'scan'), 'scan')
    data = {'stage': 'scan_ingress', 'epoch': 1, 'sequence': 0,
            'sensor_acquired_at': 10., 'sensor_received_at': 10.02, 'callback_at': 10.01}
    assert pipe.record('sensor', 10.02, data)
    assert reader.poll() == ([ingress()], None)
    assert not pipe.record('sensor', 10.03, {'stage': 'scan_result', 'sequence': 0,
                                             'reason': 'processor_exception'})
    assert reader.poll()[1] == 'producer_eof'
    reader.close()


def test_eof_preserves_complete_obstacle_record_but_never_hides_eof():
    read_fd, write_fd = os.pipe()
    reader, writer = FrameReader(read_fd, 'scan'), FrameWriter(write_fd, 'scan')
    blocked = result('obstacle')
    blocked['reason'] = 'front_blocked'
    assert writer.send(blocked)
    writer.close()
    rows, fault = reader.poll()
    assert rows == [blocked] and fault == 'producer_eof'
    reader.close()


def test_bad_tail_preserves_prior_obstacle_for_diagnosis_but_faults():
    read_fd, write_fd = os.pipe()
    reader, writer = FrameReader(read_fd, 'scan'), FrameWriter(write_fd, 'scan')
    blocked = result('obstacle')
    blocked['reason'] = 'front_blocked'
    assert writer.send(blocked)
    os.write(write_fd, b'{bad}\n')
    rows, fault = reader.poll()
    assert rows == [blocked] and fault == 'invalid_frame'
    writer.close()
    reader.close()
