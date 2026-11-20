import unittest
import sweep_simulation as sim


def grid():
    return {'resolution': .1, 'origin': (0, 0), 'cells': [[0] * 20 for _ in range(20)], 'stamp': 100.0}


class SweepTests(unittest.TestCase):
    def test_known_free_corridor_is_clear_in_synthetic_grid(self):
        result = sim.check_sweep(grid(), (.8, .8, 0), .1, 0, 1.0, .12, 101)
        self.assertEqual(result['status'], 'clear_in_simulation')
        self.assertFalse(result['executable'])

    def test_obstacle_in_swept_footprint_blocks(self):
        g = grid(); g['cells'][8][9] = 100
        self.assertEqual(sim.check_sweep(g, (.8, .8, 0), .1, 0, 1, .12, 101)['status'], 'blocked')

    def test_unknown_or_outside_grid_blocks(self):
        g = grid(); g['cells'][8][9] = -1
        self.assertEqual(sim.check_sweep(g, (.8, .8, 0), .1, 0, 1, .12, 101)['status'], 'unknown')
        self.assertEqual(sim.check_sweep(grid(), (.04, .04, 0), .1, 0, 1, .12, 101)['status'], 'unknown')

    def test_stale_and_invalid_motion_cannot_be_clear(self):
        for speed, turn, duration, now in ((.1, 0, 1, 110), (.21, 0, 1, 101),
                                            (.1, .6, 1, 101), (.1, 0, 2, 101),
                                            (float('nan'), 0, 1, 101)):
            self.assertNotEqual(sim.check_sweep(grid(), (.8, .8, 0), speed, turn, duration, .12, now)['status'], 'clear_in_simulation')


if __name__ == '__main__': unittest.main()
