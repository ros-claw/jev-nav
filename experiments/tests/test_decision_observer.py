import tempfile
import unittest
from pathlib import Path

import decision_observer as d


PNG = b'\x89PNG\r\n\x1a\n' + b'example-image'


class DecisionTests(unittest.TestCase):
    def test_valid_choice_is_advisory(self):
        def transport(payload, key, timeout):
            return {'answers': [{'type': 'choice', 'name': 'next_action', 'choice': 'forward'}]}
        result = d.choose(PNG, 'key', transport, 5, lambda: 0)
        self.assertEqual(result['choice'], 'forward')
        self.assertFalse(result['executable'])

    def test_refusal_invalid_response_and_network_failure_stop(self):
        for answer in ({'answers': [{'type': 'refusal'}]},
                       {'answers': [{'type': 'choice', 'name': 'next_action', 'choice': 'teleport'}]},
                       {'answers': []}, None):
            def transport(payload, key, timeout):
                if answer is None:
                    raise TimeoutError('sensitive details')
                return answer
            self.assertEqual(d.choose(PNG, 'key', transport, 5, lambda: 0)['choice'], 'stop')

    def test_late_response_stops(self):
        times = iter([0, 6])
        response = lambda payload, key, timeout: {'answers': [{'type': 'choice', 'name': 'next_action', 'choice': 'forward'}]}
        self.assertEqual(d.choose(PNG, 'key', response, 5, lambda: next(times))['choice'], 'stop')

    def test_invalid_image_does_not_call_network(self):
        for image in (b'', b'not png', b'\x89PNG\r\n\x1a\n' + b'x' * 4000001):
            calls = []
            result = d.choose(image, 'key', lambda *args: calls.append(True), 5, lambda: 0)
            self.assertFalse(calls)
            self.assertEqual(result['choice'], 'stop')

    def test_key_requires_one_value(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config'
            path.write_text('none')
            with self.assertRaises(ValueError): d.read_key(path)
            path.write_text('sk-' + 'a' * 20 + '\nsk-' + 'b' * 20)
            with self.assertRaises(ValueError): d.read_key(path)
            path.write_text('key: sk-' + 'a' * 20)
            self.assertEqual(d.read_key(path), 'sk-' + 'a' * 20)


if __name__ == '__main__':
    unittest.main()
