from navigation import Grid, Pose, View
from room_perception import DepthEvidence, RoomPerception


def observation(at, cells, x=.3, epoch=0, scan_at=None, origin=-.25):
    grid = Grid(origin, -.025, .05, 12, 1, tuple(cells))
    return View(at, at if scan_at is None else scan_at, at, at, at, epoch,
                Pose(x, 0., 0.), grid, True)


def test_only_previously_seen_rear_unknown_can_use_history():
    memory = RoomPerception()
    memory.view(observation(1., [0] * 12, x=0.), 1.)
    current = observation(21., [-1] * 12, x=.15)
    result = memory.view(current, 21.)
    assert result.grid.cells[5] == 0
    assert result.grid.cells[-1] == -1
    assert memory.oldest_age == 20.
    assert memory.view(observation(32., [-1] * 12, x=.15), 32.).grid.cells[5] == -1


def test_recent_obstacle_and_sensor_loss_veto_historical_clearance():
    memory = RoomPerception()
    memory.view(observation(1., [0] * 12, x=0.), 1.)
    cells = [-1] * 12
    cells[5] = 100
    assert memory.view(observation(2., cells), 2.).grid.cells[5] == 100
    assert memory.view(observation(3., [-1] * 12), 3.).grid.cells[5] == 100
    stale = observation(4., [-1] * 12, scan_at=3.0)
    assert memory.view(stale, 4.) is stale
    new_epoch = observation(5., [-1] * 12, epoch=1)
    assert memory.view(new_epoch, 5.).grid.cells[5] == -1


def test_registered_revisit_only_fills_blind_unknown_and_live_hits_invalidate_it():
    memory = RoomPerception()
    unknown = [-1] * 12
    initial = observation(120., unknown, x=.15)
    assert memory.retain_revisit(initial, [(.025, 0., 25.), (.325, 0., 26.)], 120.) == 2
    projected = memory.view(initial, 120.)
    assert projected.grid.cells[5] == 0 and projected.grid.cells[11] == -1
    assert memory.revisited_cells == 1 and memory.oldest_age == 95.
    occupied = list(unknown)
    occupied[5] = 100
    assert memory.view(observation(121., occupied, x=.15), 121.).grid.cells[5] == 100
    assert memory.view(observation(125., unknown, x=.15), 125.).grid.cells[5] == -1
    assert memory.view(observation(126., unknown, x=.15, epoch=1), 126.).grid.cells[5] == -1


def test_registered_revisit_remains_local_until_the_chassis_leaves_the_area():
    memory = RoomPerception()
    unknown = [-1] * 12
    start = observation(120., unknown, x=.15)
    assert memory.retain_revisit(start, [(.025, 0., 25.)], 120.)
    assert memory.view(observation(215., unknown, x=.15), 215.).grid.cells[5] == 0
    memory.view(observation(216., unknown, x=.47), 216.)
    assert memory.view(observation(217., unknown, x=.15), 217.).grid.cells[5] == -1


def test_isolated_occupied_cell_expires_to_unknown_instead_of_restoring_older_free_space():
    memory = RoomPerception()
    memory.view(observation(1., [0] * 12, x=0.), 1.)
    occupied = [-1] * 12
    occupied[5] = 100
    assert memory.view(observation(2., occupied), 2.).grid.cells[5] == 100
    assert memory.view(observation(2.3, [-1] * 12), 2.3).grid.cells[5] == 100
    assert memory.view(observation(3.6, [-1] * 12), 3.6).grid.cells[5] == -1


def test_repeatedly_observed_obstacle_is_retained_across_later_occlusion():
    memory = RoomPerception()
    occupied = [-1] * 12
    occupied[5] = 100
    for at in (2., 2.09, 2.18, 2.27, 2.36):
        assert memory.view(observation(at, occupied), at).grid.cells[5] == 100
    assert memory.view(observation(18., [-1] * 12), 18.).grid.cells[5] == 100
    assert memory.view(observation(34., [-1] * 12), 34.).grid.cells[5] == -1


def test_confirmed_depth_hit_survives_laser_free_and_age_until_depth_volume_is_cleared():
    memory = RoomPerception()
    occupied = [-1] * 12
    occupied[5] = 100
    for seq, at in enumerate((2., 2.10, 2.20, 2.31)):
        hit = DepthEvidence(seq, 'camera_depth_optical_frame', {5: (.018, .041)}, {})
        memory.view(observation(at, occupied)._replace(depth_evidence=hit), at)
    for at in (2.40, 38.):
        assert memory.view(observation(at, [0] * 12), at).grid.cells[5] == 100
    too_high = DepthEvidence(4, 'camera_depth_optical_frame', {}, {5: (.06, .12)})
    assert memory.view(observation(38.1, [0] * 12)._replace(depth_evidence=too_high), 38.1).grid.cells[5] == 100
    for seq, at in enumerate((38.2, 38.3, 38.4, 38.51), 5):
        clear = DepthEvidence(seq, 'camera_depth_optical_frame', {}, {5: (.0, .07)})
        view = memory.view(observation(at, [0] * 12)._replace(depth_evidence=clear), at)
    assert view.grid.cells[5] == 0


def test_republished_depth_frame_counts_once_and_old_epoch_does_not_restore_hazards():
    memory = RoomPerception()
    occupied = [-1] * 12
    occupied[5] = 100
    hit = DepthEvidence(7, 'camera_depth_optical_frame', {5: (.02, .04)}, {})
    for at in (2., 2.10, 2.20, 2.31):
        memory.view(observation(at, occupied)._replace(depth_evidence=hit), at)
    assert memory.view(observation(4., [0] * 12), 4.).grid.cells[5] == 0
    for seq, at in enumerate((4.10, 4.20, 4.30, 4.42), 8):
        fresh = hit._replace(sequence=seq)
        memory.view(observation(at, occupied)._replace(depth_evidence=fresh), at)
    assert memory.view(observation(4.5, [0] * 12), 4.5).grid.cells[5] == 100
    assert memory.view(observation(4.6, [0] * 12, epoch=1), 4.6).grid.cells[5] == 0


def test_only_newly_confirmed_or_changed_depth_evidence_is_exported_to_the_mapper():
    memory = RoomPerception()
    occupied = [-1] * 12
    occupied[5] = 100
    for seq, at in enumerate((2., 2.10, 2.20, 2.31)):
        hit = DepthEvidence(seq, 'fixed_depth', {5: (.02, .04)}, {})
        memory.view(observation(at, occupied)._replace(depth_evidence=hit), at)
    assert len(list(memory.confirmed_depth(only_changed=True))) == 1
    memory.view(observation(2.41, occupied)._replace(depth_evidence=hit._replace(sequence=4)), 2.41)
    assert list(memory.confirmed_depth(only_changed=True)) == []
    assert len(list(memory.confirmed_depth())) == 1


def test_history_stays_aligned_with_rolling_grid_origin():
    memory = RoomPerception()
    values = [-1] * 12
    values[5] = 0
    memory.view(observation(1., values, x=0.), 1.)
    translated = observation(2., [-1] * 12, origin=-.20)
    result = memory.view(translated, 2.)
    assert [index for index, cell in enumerate(result.grid.cells) if cell == 0] == [4]


def test_sensor_front_plane_can_bisect_a_grid_cell():
    memory = RoomPerception()
    memory.view(observation(1., [0] * 12, x=0.), 1.)
    result = memory.view(observation(2., [-1] * 12, x=0.), 2.)
    assert result.grid.cells[7] == 0  # .125 m center, with rear edge at .100 m
    assert result.grid.cells[8] == -1
