from navigation import Grid, Pose, View
from room_perception import RoomPerception


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
