import math
from collections import namedtuple

from navigation import Grid, Pose, View
import room_revisit


Header = namedtuple('Header', 'stamp frame_id')
Scan = namedtuple('Scan', 'header angle_min angle_increment ranges range_min range_max')


def test_revisit_requires_live_box_and_walls_and_keeps_original_observation_time(monkeypatch):
    generation = ['boot', 100, '/driver']
    monkeypatch.setattr(room_revisit, 'driver_generation', lambda pid: generation)
    stamp = type('Time', (), {'to_sec': lambda self: 1500.})()
    values = [.27 if -31 <= angle <= -13 else 2.0 for angle in range(-70, 71)]
    scan = Scan(Header(stamp, 'laser_link'), math.radians(-70), math.radians(1), values, .08, 8.)
    state = {'version': 1, 'frame': 'odom', 'cells': [(-.125, -.125, 1380.)],
             'pose': [0., 0., 0.], 'recorded_stop': 1440., 'resolution': .05,
             'closest_rays': [(angle, .27) for angle in range(-26, -15)],
             'sectors': room_revisit.scene_sectors(scan), 'driver_pid': 2, 'driver_generation': generation}
    revisit = room_revisit.RoomRevisit(state)
    grid = Grid(-.5, -.5, .05, 20, 20, (0,) * 400)
    view = View(100., 100., 100., 100., 100., 0, Pose(0., 0., 0.), grid, True)
    assert revisit.registered_cells(view, scan, 100., 1500.) == [(-.125, -.125, -20.)]
    assert not revisit.registered_cells(view, scan, 100., 1500.5)
    moved = view._replace(pose=Pose(.025, 0., 0.))
    assert not revisit.registered_cells(moved, scan, 100., 1500.)
    changed_walls = [v if v < 1. else v + .30 for v in values]
    assert not revisit.registered_cells(view, scan._replace(ranges=changed_walls), 100., 1500.)
