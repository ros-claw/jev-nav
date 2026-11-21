import os
import signal
import time

import numpy as np
import pytest

from depth_projection_process import DepthProjectionProcess, _project
from projected_depth_gate import project_depth_frame
from relay import _monotonic


def request(array, padding=0):
    height, width = array.shape
    step = width * array.dtype.itemsize + padding
    data = b''.join(row.tobytes() + b'\x00' * padding for row in array)
    params = {
        'encoding': '16UC1', 'width': width, 'height': height,
        'step': step, 'is_bigendian': int(array.dtype.byteorder == '>'),
        'intrinsics': {'width': width, 'height': height, 'fx': 120., 'fy': 120.,
                       'cx': width / 2., 'cy': height / 2., 'rectified': True},
        'matrix': [[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.2], [0., 0., 0., 1.]],
        'stamp': 100.,
        'config': {'forward_min_m': .18, 'forward_max_m': 1., 'half_width_m': .15,
                   'ground_z_m': -.1, 'ground_tolerance_m': .025, 'obstacle_min_height_m': .03,
                   'obstacle_max_height_m': .4, 'min_depth_m': .3, 'max_depth_m': 3., 'sample_stride': 4},
    }
    return params, data


@pytest.mark.parametrize('dtype,padding', [('<u2', 0), ('>u2', 12)])
def test_worker_matches_in_process_projection(dtype, padding):
    array = np.full((160, 200), 800, dtype=dtype)
    array[::3, ::2] = 0
    params, data = request(array, padding)
    expected = project_depth_frame(array, params['encoding'], params['intrinsics'],
                                   params['matrix'], params['stamp'], params['config'])
    assert expected['obstacle_count'] > 0 and expected['invalid_count'] > 0
    with DepthProjectionProcess() as projector:
        assert projector.project(params, data) == expected
        assert projector.project(dict(params, stamp=101.), data)['stamp'] == 101.
        assert projector.process.poll() is None


def test_malformed_frame_is_unavailable():
    params, data = request(np.ones((80, 100), dtype='<u2'))
    assert _project(params, data[:-1]) is None
    assert _project(dict(params, step=1), data) is None
    assert _project(dict(params, encoding='rgb8'), data) is None
    with DepthProjectionProcess() as projector:
        assert projector.project(params, b'') is None
        assert projector.project(dict(params, stamp=float('nan')), data) is None
        assert projector.project(params, data) is not None


def test_out_of_budget_response_is_drained_before_the_next_frame(monkeypatch):
    import depth_projection_process
    params, data = request(np.full((80, 100), 800, dtype='<u2'))
    with DepthProjectionProcess() as projector:
        monkeypatch.setattr(depth_projection_process, 'PROJECT_BUDGET_SEC', 0.)
        assert projector.project(params, data) is None
        assert projector.summary()['counts'] == {'over_budget': 1}
        monkeypatch.setattr(depth_projection_process, 'PROJECT_BUDGET_SEC', .065)
        assert projector.project(dict(params, stamp=101.), data)['stamp'] == 101.


def test_stalled_worker_cannot_return_previous_projection():
    params, data = request(np.full((80, 100), 800, dtype='<u2'))
    projector = DepthProjectionProcess()
    try:
        assert projector.project(params, data) is not None
        os.kill(projector.process.pid, signal.SIGSTOP)
        started = _monotonic()
        assert projector.project(dict(params, stamp=101.), data) is None
        assert _monotonic() - started < .3
        assert projector.project(params, data) is None
        assert projector.summary()['counts']['transport_timeout'] == 1
    finally:
        projector.close()
    assert projector.process.poll() is not None


def test_timeout_restarts_worker_and_requires_a_new_exchange():
    params, data = request(np.full((80, 100), 800, dtype='<u2'))
    with DepthProjectionProcess() as projector:
        previous = projector.process
        os.kill(previous.pid, signal.SIGSTOP)
        assert projector.project(params, data) is None
        deadline = _monotonic() + 3.
        result = None
        while result is None and _monotonic() < deadline:
            result = projector.project(dict(params, stamp=101.), data)
            time.sleep(.01)
        assert previous.poll() is not None
        assert projector.process.pid != previous.pid
        assert result['stamp'] == 101.
