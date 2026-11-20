import unittest
import hashlib
import json
import decision_observer as d
from test_live_observer import synchronized_snapshot

PNG = b'\x89PNG\r\n\x1a\n' + b'test'


def snapshot():
    return {'round_id': 'round1', 'image_stamp': 100, 'laser_stamp': 100,
            'status_stamp': 100, 'odom_stamp': 100,
            'laser': {'front_min_m': 3.0, 'left_min_m': 2.0, 'right_min_m': 2.0,
                      'usable_ratio': .7},
            'base': {'error_code': 0, 'motion_mode': 0, 'battery_voltage': 12.0,
                     'linear_x_mps': 0.0, 'angular_z_radps': 0.0},
            'depth': {'available': True, 'distance_measured': False}}


class SensorDecisionTests(unittest.TestCase):
    def test_synchronized_scene_preserves_frame_context_and_rejects_skew(self):
        s = synchronized_snapshot(); s['round_id'] = 'synced'
        def transport(payload, key, timeout):
            evidence = next(x['data'] for x in payload['input'][0]['content'] if x['type'] == 'input_json')
            self.assertEqual(evidence['sensor_frames']['laser'], 'laser_link')
            self.assertFalse(evidence['sensor_frames']['transform_verified'])
            self.assertIn('unverified', payload['questions'][0]['instructions'].lower())
            return {'answers': [{'type': 'choice', 'name': 'next_action', 'choice': 'forward'}]}
        result = d.choose_with_evidence(PNG, s, 'key', transport, clock=lambda: 100.2)
        self.assertEqual(result['status'], 'ok')
        s['laser_stamp'] = 93
        s['laser']['forward_view_stamp'] = 93
        calls = []
        result = d.choose_with_evidence(PNG, s, 'key', lambda *a: calls.append(True), clock=lambda: 100.2)
        self.assertFalse(calls)
        self.assertNotEqual(result['status'], 'ok')

    def test_scene_and_body_are_passed_to_model(self):
        def transport(payload, key, timeout):
            contents = payload['input'][0]['content']
            self.assertIn('explor', contents[0]['text'].lower())
            evidence = next(x['data'] for x in contents if x['type'] == 'input_json')
            self.assertEqual(evidence['laser']['front_min_m'], 3.0)
            self.assertEqual(evidence['base']['motion_mode'], 0)
            self.assertFalse(evidence['depth']['distance_measured'])
            self.assertIn('explor', payload['questions'][0]['instructions'].lower())
            self.assertIn('one direction', payload['questions'][0]['instructions'].lower())
            return {'answers': [{'type': 'choice', 'name': 'next_action', 'choice': 'forward',
                                 'probabilities': [{'value': 'forward', 'probability': .6},
                                                   {'value': 'left', 'probability': .2},
                                                   {'value': 'right', 'probability': .1},
                                                   {'value': 'stop', 'probability': .1}]}]}
        result = d.choose_with_evidence(PNG, snapshot(), 'key', transport, clock=lambda: 101)
        self.assertEqual(result['choice'], 'forward')
        self.assertFalse(result['executable'])
        self.assertEqual(result['round_id'], 'round1')
        self.assertEqual(result['direction_ranking'], ['forward', 'left', 'right'])
        expected = hashlib.sha256(hashlib.sha256(PNG).digest() + json.dumps(
            snapshot(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.assertEqual(result['observation_hash'], 'sha256:' + expected)
        self.assertEqual(result['expires_wall_time'], 108.0)

    def test_missing_or_old_or_future_evidence_fails_closed(self):
        for field, value in (('laser_stamp', 80), ('image_stamp', 110), ('status_stamp', None),
                             ('laser', None), ('base', None)):
            s = snapshot(); s[field] = value
            calls = []
            result = d.choose_with_evidence(PNG, s, 'key', lambda *a: calls.append(True), clock=lambda: 101)
            self.assertFalse(calls)
            self.assertEqual(result['choice'], 'stop')
            self.assertNotEqual(result['status'], 'ok')

    def test_moving_or_faulted_body_does_not_issue_model_request(self):
        for field, value in (('error_code', 1), ('motion_mode', 1), ('linear_x_mps', .1)):
            s = snapshot(); s['base'][field] = value
            calls = []
            result = d.choose_with_evidence(PNG, s, 'key', lambda *a: calls.append(True), clock=lambda: 101)
            self.assertFalse(calls)
            self.assertEqual(result['choice'], 'stop')
            self.assertNotEqual(result['status'], 'ok')

    def test_continuous_mode_requests_advice_with_current_velocity(self):
        s = snapshot()
        s['base']['linear_x_mps'] = 0.1
        s['base']['angular_z_radps'] = 0.2
        def transport(payload, key, timeout):
            evidence = next(x['data'] for x in payload['input'][0]['content'] if x['type'] == 'input_json')
            self.assertEqual(evidence['base']['linear_x_mps'], 0.1)
            self.assertEqual(evidence['base']['angular_z_radps'], 0.2)
            return {'answers': [{'type': 'choice', 'name': 'next_action', 'choice': 'left'}]}
        result = d.choose_with_evidence(PNG, s, 'key', transport, clock=lambda: 101, allow_moving=True)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['choice'], 'left')
        self.assertFalse(result['executable'])

    def test_model_response_expires_with_snapshot(self):
        times = iter([101, 110])
        transport = lambda *a: {'answers': [{'type': 'choice', 'name': 'next_action', 'choice': 'forward'}]}
        result = d.choose_with_evidence(PNG, snapshot(), 'key', transport, clock=lambda: next(times))
        self.assertEqual(result['choice'], 'stop')
        self.assertEqual(result['status'], 'expired')

    def test_bad_laser_values_do_not_become_clearance(self):
        for value in (float('nan'), -1):
            s = snapshot(); s['laser']['front_min_m'] = value
            calls = []
            result = d.choose_with_evidence(PNG, s, 'key', lambda *a: calls.append(True), clock=lambda: 101)
            self.assertFalse(calls)
            self.assertEqual(result['choice'], 'stop')
            self.assertNotEqual(result['status'], 'ok')

    def test_partial_lidar_unknowns_reach_model_as_unknown_not_as_zero(self):
        s = snapshot()
        s['laser']['left_min_m'] = None
        s['laser']['right_min_m'] = None
        def transport(payload, key, timeout):
            evidence = next(part['data'] for part in payload['input'][0]['content'] if part['type'] == 'input_json')
            self.assertIsNone(evidence['laser']['left_min_m'])
            self.assertIsNone(evidence['laser']['right_min_m'])
            self.assertEqual(evidence['laser']['front_min_m'], 3.0)
            return {'answers': [{'type': 'choice', 'name': 'next_action', 'choice': 'forward'}]}
        result = d.choose_with_evidence(PNG, s, 'key', transport, clock=lambda: 101)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['choice'], 'forward')
        self.assertFalse(result['executable'])

    def test_all_lidar_sectors_unknown_remain_unavailable(self):
        s = snapshot()
        for key in ('front_min_m', 'left_min_m', 'right_min_m'):
            s['laser'][key] = None
        calls = []
        result = d.choose_with_evidence(PNG, s, 'key', lambda *args: calls.append(True), clock=lambda: 101)
        self.assertFalse(calls)
        self.assertNotEqual(result['status'], 'ok')

    def test_measured_depth_must_be_fresh(self):
        s = snapshot(); s['depth'] = {'available': True, 'distance_measured': True, 'stamp': 80,
            'units': 'optical_depth_m', 'sectors': {name: {'minimum_m': 1.0, 'valid_ratio': .5}
                                                    for name in ('left', 'center', 'right')}}
        calls = []
        def transport(*args):
            calls.append(True)
            return {'answers': [{'type': 'choice', 'name': 'next_action', 'choice': 'forward'}]}
        result = d.choose_with_evidence(PNG, s, 'key', transport, clock=lambda: 101)
        self.assertFalse(calls)
        self.assertEqual(result['choice'], 'stop')
        self.assertNotEqual(result['status'], 'ok')

    def test_forward_view_must_share_the_fresh_observation_window(self):
        s = snapshot()
        s['laser']['forward_view'] = {'rear_total_rays': 0, 'rear_valid_rays': 0}
        s['laser']['forward_view_stamp'] = 80
        calls = []
        result = d.choose_with_evidence(PNG, s, 'key', lambda *args: calls.append(True), clock=lambda: 101)
        self.assertFalse(calls)
        self.assertNotEqual(result['status'], 'ok')

    def test_request_uses_frozen_snapshot_when_caller_mutates_original(self):
        source = snapshot()
        def transport(payload, key, timeout):
            source['laser']['front_min_m'] = .01
            data = next(part['data'] for part in payload['input'][0]['content']
                        if part['type'] == 'input_json')
            self.assertEqual(data['laser']['front_min_m'], 3.0)
            return {'answers': [{'type': 'choice', 'name': 'next_action', 'choice': 'forward'}]}
        result = d.choose_with_evidence(PNG, source, 'key', transport, clock=lambda: 101)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['choice'], 'forward')


if __name__ == '__main__':
    unittest.main()
