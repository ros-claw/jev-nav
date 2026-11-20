"""Decision-to-SHADOW intent tests, no robot I/O."""
import unittest

import shadow_intent


def decision():
    return {'status': 'ok', 'choice': 'forward', 'executable': False,
            'round_id': 'round-1', 'observation_hash': 'sha256:' + 'b' * 64,
            'expires_wall_time': 102.0}


class ShadowIntentTests(unittest.TestCase):
    def test_live_decision_is_bound_to_snapshot_and_bounded_monotonic_expiry(self):
        request = shadow_intent.make_intent(decision(), 'action-1', 'sha256:' + 'a' * 64,
                                             wall_now=100, monotonic_now=50, duration=.25)
        self.assertEqual(request['primitive'], 'forward')
        self.assertEqual(request['observation_hash'], decision()['observation_hash'])
        self.assertEqual(request['round_id'], 'round-1')
        self.assertEqual(request['created_monotonic'], 50)
        self.assertLessEqual(request['expires_monotonic'], 51)
        self.assertGreaterEqual(request['expires_monotonic'], 50.25)
        self.assertNotIn('velocity', request)

    def test_expired_unbound_or_non_advisory_decisions_reject(self):
        cases = [('expires_wall_time', 100.1), ('observation_hash', 'fake'),
                 ('status', 'expired'), ('executable', True), ('choice', 'reverse'),
                 ('round_id', ''), ('expires_wall_time', float('nan'))]
        for key, value in cases:
            data = decision(); data[key] = value
            with self.assertRaises(ValueError):
                shadow_intent.make_intent(data, 'action-1', 'sha256:' + 'a' * 64,
                                           wall_now=100, monotonic_now=50, duration=.25)

    def test_bad_duration_and_bad_identity_reject(self):
        for action, duration in (('', .25), ('action-1', 1.1), ('action-1', -1), ('action-1', True)):
            with self.assertRaises(ValueError):
                shadow_intent.make_intent(decision(), action, 'sha256:' + 'a' * 64,
                                           wall_now=100, monotonic_now=50, duration=duration)


if __name__ == '__main__': unittest.main()
