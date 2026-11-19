"""Offline checks for the isolated, observation-only native ROS probe."""
import math
import os
import subprocess
import sys
import time

import pytest
from native_probe_tf import valid_namespace
from native_sensing_probe import _sampled_front_cells, _stop_group


@pytest.mark.parametrize('value,expected', [
    ('limo_native_probe_123456abcdef', True),
    ('limo_native_probe_123456abcde', False),
    ('limo_native_probe_123456abcdef\n', False),
    ('/tf', False), ('cmd_vel', False), ('', False), (None, False),
])
def test_private_topic_namespace_cannot_escape_the_probe(value, expected):
    assert valid_namespace(value) is expected


def test_cleanup_stops_its_own_process_group():
    with open(os.devnull, 'wb') as output:
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                   stdout=output, stderr=output, preexec_fn=os.setsid)
        try:
            started = time.time()
            _stop_group(process)
            assert process.poll() is not None
            assert time.time() - started < 4.0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_front_summary_uses_robot_heading_and_keeps_unknown_cells():
    class Obj(object):
        def __init__(self, **values):
            self.__dict__.update(values)
    class Stamp(object):
        def __sub__(self, other):
            return self
        def to_sec(self):
            return 0.0
    stamp = Stamp()
    zero_rotation = Obj(x=0.0, y=0.0, z=0.0, w=1.0)
    origin = Obj(position=Obj(x=-.1, y=0.0), orientation=zero_rotation)
    grid = Obj(header=Obj(frame_id='odom', stamp=stamp),
               info=Obj(origin=origin, resolution=.2, width=1, height=2), data=[0, -1])
    turn = Obj(x=0.0, y=0.0, z=math.sin(math.pi/4), w=math.cos(math.pi/4))
    pose = Obj(position=Obj(x=0.0, y=0.0), orientation=turn)
    odom = Obj(header=Obj(frame_id='odom', stamp=stamp), child_frame_id='base_link',
               pose=Obj(pose=pose))
    assert _sampled_front_cells(grid, odom) == {'unknown': 1, 'free': 0, 'inflated': 0, 'occupied': 0}
