import unittest
import replay_controller as rc


def grid(stamp=100):
    return {'resolution': .1, 'origin': (0, 0), 'cells': [[0]*20 for _ in range(20)], 'stamp': stamp}


def decision(choice='forward', round_id='r1'):
    return {'status': 'ok', 'choice': choice, 'round_id': round_id, 'executable': False}


class ControllerTests(unittest.TestCase):
    def test_open_grid_forward_is_simulated_only(self):
        c = rc.ReplayController()
        self.assertTrue(c.begin(decision(), 'r1', 100, 100.8))
        result = c.step(100.1, grid(100.1), (.8, .8, 0))
        self.assertEqual(result['simulated_linear_mps'], .1)
        self.assertFalse(result['executable'])

    def test_forward_does_not_require_unknown_rear_in_synthetic_grid(self):
        g = grid(100.1)
        g['cells'] = [[-1] * 20 for _ in range(20)]
        for row in (6, 7, 8, 9):
            g['cells'][row][9] = 0
        c = rc.ReplayController()
        self.assertTrue(c.begin(decision(), 'r1', 100, 100.8))
        result = c.step(100.1, g, (.8, .8, 0))
        self.assertEqual(result['simulated_linear_mps'], .1)
        self.assertFalse(result['executable'])

    def test_turn_with_unknown_rear_remains_unknown_in_simulation(self):
        g = grid(100.1)
        g['cells'] = [[-1] * 20 for _ in range(20)]
        for row in (6, 7, 8, 9):
            g['cells'][row][9] = 0
        c = rc.ReplayController()
        self.assertTrue(c.begin(decision('left'), 'r1', 100, 100.8))
        self.assertEqual(c.step(100.1, g, (.8, .8, 0))['reason'], 'unknown')

    def test_new_obstacle_stops_and_latches(self):
        c = rc.ReplayController(); c.begin(decision(), 'r1', 100, 100.8)
        g = grid(100.1); g['cells'][8][9] = 100
        result = c.step(100.1, g, (.8, .8, 0))
        self.assertEqual(result['simulated_linear_mps'], 0)
        self.assertEqual(result['reason'], 'blocked')
        self.assertEqual(c.step(100.2, grid(100.2), (.8, .8, 0))['simulated_linear_mps'], 0)

    def test_expired_lease_or_decision_or_unknown_stops(self):
        c = rc.ReplayController(); c.begin(decision(), 'r1', 100, 100.8)
        self.assertEqual(c.step(100.3, grid(100.3), (.8, .8, 0))['reason'], 'lease_expired')
        c = rc.ReplayController(); c.begin(decision(), 'r1', 100, 100.8)
        g = grid(100.1); g['cells'][8][9] = -1
        self.assertEqual(c.step(100.1, g, (.8, .8, 0))['reason'], 'unknown')
        c = rc.ReplayController(); c.begin(decision(), 'r1', 100, 100.1)
        self.assertEqual(c.step(100.11, grid(100.11), (.8, .8, 0))['reason'], 'decision_expired')

    def test_wrong_round_or_stop_never_starts(self):
        c = rc.ReplayController()
        self.assertFalse(c.begin(decision(round_id='old'), 'r1', 100, 100.8))
        self.assertFalse(c.begin(decision('stop'), 'r1', 100, 100.8))
        self.assertEqual(c.step(100.1, grid(100.1), (.8, .8, 0))['simulated_linear_mps'], 0)


if __name__ == '__main__': unittest.main()
