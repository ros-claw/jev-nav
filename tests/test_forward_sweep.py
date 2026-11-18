"""A front-facing grid can describe forward sweep without knowing the rear."""

import importlib.util
import math
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "worker" / "limo_forward_sweep.py"
spec = importlib.util.spec_from_file_location("limo_forward_sweep", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def check(grid, pose=(0.0, 0.0, 0.0), distance=0.1, now=10.5):
    function = getattr(module, "check_forward_sweep", None)
    assert callable(function), "straight-ahead sweep evaluation is unavailable"
    return function(grid, pose, (0.15, 0.15, 0.1), distance, now)


def forward_grid():
    # Only the strip in front of the initial footprint is known; rear is unknown.
    rows = [[-1 for _ in range(10)] for _ in range(8)]
    for row in (2, 3, 4, 5):
        for col in (4, 5):
            rows[row][col] = 0
    return {"resolution": 0.1, "origin": (-0.3, -0.4), "cells": rows, "stamp": 10.0}


def test_straight_forward_sweep_does_not_require_unknown_rear():
    result = check(forward_grid())
    assert result["status"] == "clear_in_simulation"
    assert result["executable"] is False


def test_forward_unknown_and_obstacles_are_not_clear():
    grid = forward_grid()
    grid["cells"][3][5] = -1
    assert check(grid)["status"] == "unknown"
    grid["cells"][3][5] = 100
    assert check(grid)["status"] == "blocked"


def test_obstacle_in_current_body_footprint_is_not_ignored():
    grid = forward_grid()
    grid["cells"][3][2] = 100
    assert check(grid)["status"] == "blocked"


@pytest.mark.parametrize("cell_value,expected", [(100, "blocked"), (-1, "unknown")])
def test_touching_the_final_front_boundary_is_not_clear(cell_value, expected):
    grid = {
        "resolution": 0.1,
        "origin": (-0.5, -0.5),
        "cells": [[0] * 12 for _ in range(12)],
        "stamp": 10.0,
    }
    # For this grid col 7 starts at x=0.2, the final front boundary.
    grid["cells"][5][7] = cell_value
    assert (
        module.check_forward_sweep(grid, (0, 0, 0), (0.1, 0.1, 0.1), 0.1, 10.5)["status"]
        == expected
    )


def test_unknown_and_obstacle_remain_visible_at_large_world_coordinates():
    p = 1e8
    grid = {
        "resolution": 0.1,
        "origin": (p - 0.5, p - 0.5),
        "cells": [[-1] * 10 for _ in range(10)],
        "stamp": 10.0,
    }
    assert check(grid, pose=(p, p, 0))["status"] == "unknown"
    grid["cells"] = [[0] * 10 for _ in range(10)]
    grid["cells"][4][7] = 100
    assert check(grid, pose=(p, p, 0))["status"] == "blocked"


def test_grid_fails_closed_when_world_coordinates_exceed_its_precision():
    p = 1e16
    grid = {
        "resolution": 0.1,
        "origin": (p, p),
        "cells": [[0] * 10 for _ in range(10)],
        "stamp": 10.0,
    }
    assert check(grid, pose=(p, p, 0))["status"] == "invalid"


def test_obstacle_touching_the_back_of_the_existing_footprint_is_blocked():
    grid = {
        "resolution": 0.125,
        "origin": (0, 0),
        "cells": [[0] * 8 for _ in range(8)],
        "stamp": 10.0,
    }
    grid["cells"][3][1] = 100
    assert (
        module.check_forward_sweep(grid, (0.375, 0.375, 0), (0.125, 0.125, 0.125), 0.1, 10.5)[
            "status"
        ]
        == "blocked"
    )


def test_rotated_forward_sweep_uses_world_frame_and_outside_is_unknown():
    grid = {
        "resolution": 0.1,
        "origin": (-0.5, -0.5),
        "cells": [[0] * 10 for _ in range(10)],
        "stamp": 10.0,
    }
    assert check(grid, pose=(0, 0, math.pi / 4))["status"] == "clear_in_simulation"
    grid["cells"][6][6] = 100
    assert check(grid, pose=(0, 0, math.pi / 4))["status"] == "blocked"
    assert check(grid, pose=(0.45, 0, 0))["status"] == "unknown"


@pytest.mark.parametrize(
    "distance,now", [(0.1, 20), (-0.1, 10.5), (float("nan"), 10.5), (0.1, 9.0)]
)
def test_stale_or_invalid_input_is_never_clear(distance, now):
    assert check(forward_grid(), distance=distance, now=now)["status"] != "clear_in_simulation"
