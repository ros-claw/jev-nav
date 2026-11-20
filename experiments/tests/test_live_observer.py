import asyncio
import base64
import hashlib
import importlib.util
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import live_observer as live


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC')


def result(payload, image=None):
    blocks = [SimpleNamespace(type='text', text=json.dumps(payload))]
    if image is not None:
        blocks.append(SimpleNamespace(type='image', data=base64.b64encode(image).decode(), mimeType='image/png'))
    return SimpleNamespace(isError=False, structuredContent=payload, content=blocks)


def responses():
    return {
        'limo_capture_camera_frame': result({'ok': True, 'stamp_sec': 100}, PNG),
        'limo_get_laser_summary': result({'ok': True, 'summary': {
            'header': {'stamp_sec': 100}, 'usable_ratio': .5,
            'sectors': {'front_min_m': 3, 'left_min_m': 2, 'right_min_m': 1}}}),
        'limo_get_base_state': result({'ok': True, 'summaries': {
            'status': {'header': {'stamp_sec': 100}, 'error_code': 0, 'motion_mode': 0, 'battery_voltage': 12},
            'odometry': {'header': {'stamp_sec': 100}, 'velocity': {'linear_x_mps': 0, 'angular_z_radps': 0}}}}),
        'limo_get_depth_summary': result({'ok': True, 'stamp_sec': 100,
            'units': 'optical_depth_m', 'safety_clearance_verified': False,
            'bands': {'bottom': {'center': {'minimum_m': .4, 'valid_ratio': .2}}},
            'sectors': {'left': {'minimum_m': .7, 'valid_ratio': .2},
                        'center': {'minimum_m': .8, 'valid_ratio': .1},
                        'right': {'minimum_m': .9, 'valid_ratio': .3}}}),
        'limo_capture_planar_scan': result({
            'ok': True, 'schema_version': 'limo.planar_scan.v1',
            'command_dispatched': False, 'diagnostic_only': True,
            'transform_verified': False, 'usable_for_real_execution': False,
            'deskewed': False,
            'scan': {'stamp': 100, 'frame_id': 'laser', 'angle_min': -math.pi/2,
                     'angle_increment': math.pi/180, 'range_min': .1, 'range_max': 12,
                     'ranges': [2.0]*181,
                     'transform': {'source': 'laser', 'target': 'base_link', 'stamp': 100,
                                   'matrix': [[1,0,0,.105],[0,1,0,0],[0,0,1,.08],[0,0,0,1]]}},
        }),
    }


class FakeSession:
    def __init__(self, values): self.values = values
    async def call_tool(self, name, args): return self.values[name]


def scene_core():
    path = Path(live.__file__).parent / 'worktrees/limo-mcp/src/limo_ros_mcp/workers/scene_observation_core.py'
    spec = importlib.util.spec_from_file_location('scene_observation_test', path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def synchronized_snapshot():
    core = scene_core()
    data = {'stamps': {name: 100.0 for name in ('color', 'scan', 'status', 'odom', 'depth')},
            'scan': {'angle_min': 0, 'angle_increment': .1, 'range_min': .1,
                     'range_max': 12., 'time_increment': 0., 'scan_time': .1, 'ranges': [3.]},
            'frames': {'laser': 'laser_link', 'color': 'color_optical', 'depth': 'depth_optical'},
            'laser_tf': {'source': 'laser_link', 'target': 'base_link', 'stamp': 100.,
                         'x': .105, 'y': 0., 'yaw': 0., 'verified': False},
            'base': {'error_code': 0, 'motion_mode': 0, 'battery_voltage': 12.,
                     'linear_x_mps': 0., 'angular_z_radps': 0.},
            'depth': {name: {'minimum_m': 2., 'valid_ratio': .5}
                      for name in ('left', 'center', 'right')}}
    return core.make_snapshot(data, 100.1, 100.1)


class ObserverTests(unittest.TestCase):
    def run_async(self, coro):
        loop = asyncio.new_event_loop()
        try:
            with patch('live_observer.time.time', return_value=100.1):
                return loop.run_until_complete(coro)
        finally: loop.close()

    def test_combines_real_fields_without_inventing_depth_distance(self):
        image, snapshot = self.run_async(live.collect(FakeSession(responses())))
        self.assertEqual(image, PNG)
        self.assertEqual(snapshot['laser']['front_min_m'], 3)
        self.assertIsNone(snapshot['laser']['rear_min_m'])
        self.assertEqual(snapshot['laser']['forward_view']['rear_total_rays'], 0)
        self.assertGreater(snapshot['laser']['forward_view']['center']['valid_rays'], 0)
        self.assertEqual(snapshot['laser']['forward_view_stamp'], 100)
        self.assertEqual(snapshot['base']['motion_mode'], 0)
        self.assertEqual(snapshot['depth']['stamp'], 100)
        self.assertTrue(snapshot['depth']['distance_measured'])
        self.assertEqual(snapshot['depth']['sectors']['center']['valid_ratio'], .1)
        self.assertEqual(snapshot['depth']['bands']['bottom']['center']['minimum_m'], .4)

    def test_synchronized_capture_uses_one_model_scene_and_separate_image(self):
        source = synchronized_snapshot()
        metadata = {'ok': True, 'schema_version': 'limo.synchronized_scene.v1',
                    'sync': 'ros_header_approximate', 'diagnostic_only': True,
                    'command_dispatched': False, 'usable_for_real_execution': False,
                    'snapshot': source, 'png_sha256': 'sha256:' + hashlib.sha256(PNG).hexdigest(),
                    'scene_sha256': scene_core().scene_digest(PNG, source)}
        collector = getattr(live, 'collect_synchronized', None)
        self.assertTrue(callable(collector), 'synchronized collector unavailable')
        image, snapshot = self.run_async(collector(FakeSession(
            {'limo_capture_synchronized_scene': result(metadata, PNG)})))
        self.assertEqual(image, PNG)
        self.assertEqual(snapshot['image_stamp'], 100.0)
        self.assertIsNone(snapshot['laser']['rear_min_m'])
        self.assertTrue(snapshot['round_id'])

    def test_synchronized_capture_rejects_executable_or_malformed_output(self):
        metadata = {'ok': True, 'schema_version': 'limo.synchronized_scene.v1',
                    'sync': 'ros_header_approximate', 'diagnostic_only': True,
                    'command_dispatched': False, 'usable_for_real_execution': True,
                    'snapshot': {}}
        collector = getattr(live, 'collect_synchronized', None)
        self.assertTrue(callable(collector), 'synchronized collector unavailable')
        with self.assertRaises(live.ObservationError):
            self.run_async(collector(FakeSession(
                {'limo_capture_synchronized_scene': result(metadata, PNG)})))

    def test_synchronized_capture_rejects_mismatched_image_or_skewed_snapshot(self):
        metadata = {'ok': True, 'schema_version': 'limo.synchronized_scene.v1',
                    'sync': 'ros_header_approximate', 'diagnostic_only': True,
                    'command_dispatched': False, 'usable_for_real_execution': False,
                    'snapshot': synchronized_snapshot(), 'png_sha256': 'sha256:wrong'}
        metadata['scene_sha256'] = scene_core().scene_digest(PNG, metadata['snapshot'])
        with self.assertRaises(live.ObservationError):
            self.run_async(live.collect_synchronized(FakeSession(
                {'limo_capture_synchronized_scene': result(metadata, PNG)})))
        metadata['png_sha256'] = 'sha256:' + hashlib.sha256(PNG).hexdigest()
        metadata['snapshot'] = synchronized_snapshot()
        old = metadata['snapshot']
        for key in ('image_stamp', 'laser_stamp', 'status_stamp', 'odom_stamp'):
            old[key] = 90.0
        old['depth']['stamp'] = 90.0
        old['laser']['forward_view_stamp'] = 90.0
        old['laser']['scan_end_stamp'] = 90.0
        metadata['scene_sha256'] = scene_core().scene_digest(PNG, old)
        with self.assertRaises(live.ObservationError):
            self.run_async(live.collect_synchronized(FakeSession(
                {'limo_capture_synchronized_scene': result(metadata, PNG)})))
        metadata['png_sha256'] = 'sha256:' + hashlib.sha256(PNG).hexdigest()
        metadata['snapshot']['laser_stamp'] = 93.0
        metadata['snapshot']['laser']['forward_view_stamp'] = 93.0
        metadata['scene_sha256'] = scene_core().scene_digest(PNG, metadata['snapshot'])
        with self.assertRaises(live.ObservationError):
            self.run_async(live.collect_synchronized(FakeSession(
                {'limo_capture_synchronized_scene': result(metadata, PNG)})))
        metadata['snapshot'] = synchronized_snapshot()
        metadata['scene_sha256'] = scene_core().scene_digest(PNG, metadata['snapshot'])
        metadata['snapshot']['base']['battery_voltage'] = 12.2
        with self.assertRaises(live.ObservationError):
            self.run_async(live.collect_synchronized(FakeSession(
                {'limo_capture_synchronized_scene': result(metadata, PNG)})))
        metadata['snapshot'] = synchronized_snapshot()
        bad_png = b'\x89PNG\r\n\x1a\nimage'
        metadata['png_sha256'] = 'sha256:' + hashlib.sha256(bad_png).hexdigest()
        metadata['scene_sha256'] = scene_core().scene_digest(bad_png, metadata['snapshot'])
        with self.assertRaises(live.ObservationError):
            self.run_async(live.collect_synchronized(FakeSession(
                {'limo_capture_synchronized_scene': result(metadata, bad_png)})))

    def test_missing_sensor_does_not_produce_valid_snapshot(self):
        values = responses()
        values['limo_get_base_state'] = result({'ok': False})
        with self.assertRaises(live.ObservationError):
            self.run_async(live.collect(FakeSession(values)))

    def test_required_sensor_failure_identifies_which_sensor(self):
        for name, reason in (('limo_capture_camera_frame', 'camera_unavailable'),
                             ('limo_get_laser_summary', 'lidar_unavailable'),
                             ('limo_get_base_state', 'base_unavailable')):
            values = responses()
            values[name] = result({'ok': False})
            with self.assertRaisesRegex(live.ObservationError, reason):
                self.run_async(live.collect(FakeSession(values)))

    def test_optional_depth_transport_error_is_unknown_not_fatal(self):
        values = responses()
        values['limo_get_depth_summary'] = SimpleNamespace(isError=True, structuredContent=None, content=[])
        _image, snapshot = self.run_async(live.collect(FakeSession(values)))
        self.assertFalse(snapshot['depth']['available'])
        self.assertFalse(snapshot['depth']['distance_measured'])

    def test_optional_depth_transport_exception_is_unknown_not_fatal(self):
        class DepthFailureSession(FakeSession):
            async def call_tool(self, name, args):
                if name == 'limo_get_depth_summary':
                    raise TimeoutError('depth timed out')
                return await super().call_tool(name, args)
        _image, snapshot = self.run_async(live.collect(DepthFailureSession(responses())))
        self.assertFalse(snapshot['depth']['available'])

    def test_retry_stops_after_two_failed_rounds_with_reason(self):
        values = responses()
        values['limo_get_laser_summary'] = result({'ok': False})
        with self.assertRaisesRegex(live.ObservationError, 'lidar_unavailable'):
            self.run_async(live.collect_with_retry(FakeSession(values)))

    def test_retry_recaptures_every_sensor_after_a_transient_failure(self):
        class FlakySession:
            def __init__(self): self.calls = []
            async def call_tool(self, name, args):
                self.calls.append(name)
                values = responses()
                if name == 'limo_get_laser_summary' and self.calls.count(name) == 1:
                    return result({'ok': False})
                return values[name]
        session = FlakySession()
        image, snapshot = self.run_async(live.collect_with_retry(session))
        self.assertEqual(image, PNG)
        self.assertTrue(snapshot['round_id'])
        for name in responses():
            self.assertEqual(session.calls.count(name), 2)

    def test_bad_image_is_rejected(self):
        values = responses()
        values['limo_capture_camera_frame'] = result({'ok': True, 'stamp_sec': 100}, b'not an image')
        with self.assertRaises(live.ObservationError):
            self.run_async(live.collect(FakeSession(values)))


if __name__ == '__main__': unittest.main()
