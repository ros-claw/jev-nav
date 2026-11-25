import math
from collections import namedtuple

import pytest

from navigation import Grid, Pose, View, sweep_clear
from room_perception import RoomPerception
from run_navigation import RosNavigationInput
from spatial_memory import MapFix, SpatialMemory
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


def test_roam_revisit_waits_for_registration_and_keeps_the_prior_obstacle(monkeypatch):
    generation = ['boot', 100, '/driver']
    monkeypatch.setattr(room_revisit, 'driver_generation', lambda pid: generation)
    stamp = type('Time', (), {'to_sec': lambda self: 1500.})()
    values = [.27 if -31 <= angle <= -13 else 2.0 for angle in range(-70, 71)]
    scan = Scan(Header(stamp, 'laser_link'), math.radians(-70), math.radians(1), values, .08, 8.)
    state = {'version': 1, 'frame': 'odom', 'cells': [(-.125, -.125, 1380.)],
             'pose': [0., 0., 0.], 'recorded_stop': 1440., 'resolution': .05,
             'closest_rays': [(angle, .27) for angle in range(-26, -15)],
             'sectors': room_revisit.scene_sectors(scan), 'driver_pid': 2, 'driver_generation': generation,
             'hazard_discs': [[.28, 0., .05]]}
    revisit = room_revisit.RoomRevisit(state)
    source = RosNavigationInput.__new__(RosNavigationInput)
    source.revisit = revisit
    source.revisit_scan = source.latest_scan = None
    source.revisit_registered = 0
    source.revisit_hazards = None
    source.rospy = type('ROS', (), {'Time': type('Time', (), {'now': staticmethod(lambda: stamp)})})()
    room = RoomPerception()
    origin = Pose(0., 0., 0.)
    view = View(100., 100., 100., 100., 100., 0, origin, Grid(-.5, -.5, .05, 20, 20, (0,) * 400), True)

    assert not source.register_revisit(room, view, 100.).ready
    source.latest_scan = scan._replace(ranges=[v if v < 1. else v + .3 for v in values])
    assert not source.register_revisit(room, view, 100.).ready
    assert not source.register_revisit(room, view, 100.).ready

    source.latest_scan = scan
    registered = source.register_revisit(room, view, 100.)
    memory = SpatialMemory('room')
    assert memory.update_fix(MapFix('room', 1, origin, {1: origin}))
    memory.record_depth([(.34, -.15, .025, .02, .045)])
    effective = memory.apply(room.view(registered, 100.), continuous=True)
    assert source.revisit is None and source.revisit_registered == 1
    assert (.28, 0., .05) in effective.grid.hazard_discs and len(effective.grid.hazard_discs) == 2
    assert not sweep_clear(effective.grid, origin, .12, 0.)
    next_epoch = view._replace(epoch=1)
    assert not source.register_revisit(room, next_epoch, 100.).grid.hazard_discs
    assert source.revisit_hazards is None

    with pytest.raises(ValueError):
        room_revisit.RoomRevisit(dict(state, hazard_discs=[[.28, 0., float('inf')]]))
