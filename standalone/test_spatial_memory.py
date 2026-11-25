import json

import pytest

from navigation import Grid, Pose, Request, View, sweep_clear
from spatial_memory import MapFix, SpatialMemory
from spatial_runtime import closure_for_graph


def view(epoch=0):
    return View(1., 1., 1., 1., 1., epoch, Pose(0., 0., 0.),
                Grid(-1., -1., .05, 40, 40, (0,) * 1600), True)


def test_hazard_survives_restart_and_follows_its_corrected_keyframe():
    initial = MapFix('room', 1, Pose(1., 0., 0.), {5: Pose(1., 0., 0.)})
    memory = SpatialMemory('room')
    assert memory.update_fix(initial)
    memory.record_depth([(.24, 0., .025, .018, .04)])
    assert not sweep_clear(memory.apply(view()).grid, view().pose, .12, 0.)

    restored = SpatialMemory('room', json.loads(json.dumps(memory.snapshot())))
    corrected = MapFix('room', 2, Pose(2., 0., 0.), {5: Pose(2., 0., 0.)})
    assert not restored.apply(view()).ready
    assert not restored.update_fix(corrected, closure=7)
    assert restored.update_fix(corrected, closure=5)
    assert not sweep_clear(restored.apply(view()).grid, view().pose, .12, 0.)
    shifted = corrected._replace(revision=3, anchors={5: Pose(2., .4, 0.)})
    assert restored.update_fix(shifted)
    assert sweep_clear(restored.apply(view()).grid, view().pose, .12, 0.)
    restored.clear_depth([(.24, .4, .05, .10)])
    assert restored.hazards
    restored.clear_depth([(.24, .4, .0, .10)])
    assert not restored.hazards
    restored.reset_registration()
    assert not restored.apply(view(epoch=1)).ready


def test_visited_cells_are_persisted_and_unchecked_actions_do_not_become_targets():
    memory = SpatialMemory('room')
    assert memory.update_fix(MapFix('room', 1, Pose(0., 0., 0.), {7: Pose(0., 0., 0.)}))
    assert memory.visit(Pose(.12, 0., 0.))
    assert not memory.visit(Pose(.13, 0., 0.))
    request = Request(8, 1., 0, Pose(0., 0., 0.), ('forward', 'left', 'stop'), None, ())
    targets = memory.targets(request, .12, {'forward': 0., 'left': 1.35})
    assert [item['action'] for item in targets] == ['forward', 'left']
    assert targets[0]['visits'] == 1 and all(t['map_id'] == 'room' for t in targets)
    grid = view().grid
    cells = list(grid.cells)
    cells[20 * grid.width + 27] = -1
    targets = memory.targets(request, .12, {'forward': 0., 'left': 1.35}, grid._replace(cells=tuple(cells)))
    assert targets[0]['frontier_cells'] == 1 and targets[0]['distance_m'] == .12
    with pytest.raises(ValueError, match='map'):
        SpatialMemory('different-room', memory.snapshot())


def test_only_same_acquisition_loop_closure_can_register_an_existing_map():
    graph = (5., MapFix('room', 405000000000, Pose(0., 0., 0.), {7: Pose(0., 0., 0.)}))
    assert closure_for_graph(graph, (5.02, 405020000000, 7), 4.) is None
    assert closure_for_graph(graph, (5., 405000000000, 7), 4.) == 7
    assert closure_for_graph(graph, (5., 405000000000, 7), 5.1) is None


def test_new_hazards_prefer_a_keyframe_that_persists_across_graph_revisions():
    origin, near = Pose(0., 0., 0.), Pose(.26, 0., 0.)
    memory = SpatialMemory('room')
    memory.update_fix(MapFix('room', 1, origin, {5: origin, 9: near}))
    memory.update_fix(MapFix('room', 2, origin, {5: origin, 10: near}))
    memory.record_depth([(.24, 0., .025, .02, .05)])
    assert {hazard.anchor for hazard in memory.hazards.values()} == {5}
    memory.update_fix(MapFix('room', 3, origin, {5: origin, 11: near}))
    effective = memory.apply(view())
    assert effective.ready and not sweep_clear(effective.grid, effective.pose, .12, 0.)


def test_a_missing_keyframe_keeps_its_last_projected_hazard_only_in_the_same_odometry_epoch():
    origin = Pose(0., 0., 0.)
    memory = SpatialMemory('room')
    memory.update_fix(MapFix('room', 1, origin, {5: origin}))
    memory.record_depth([(.24, 0., .025, .02, .05)])
    memory.update_fix(MapFix('room', 2, Pose(1., 0., 0.), {7: Pose(1., 0., 0.)}))
    effective = memory.apply(view())
    assert effective.ready and not sweep_clear(effective.grid, effective.pose, .12, 0.)
    memory.update_fix(MapFix('room', 3, Pose(1., 0., 0.), {7: Pose(1., 0., 0.), 5: Pose(1., .4, 0.)}))
    effective = memory.apply(view())
    assert sweep_clear(effective.grid, effective.pose, .12, 0.)
    memory.reset_registration()
    assert not memory.update_fix(MapFix('room', 4, origin, {7: origin}), closure=7)
    assert not memory.apply(view(epoch=1)).ready
