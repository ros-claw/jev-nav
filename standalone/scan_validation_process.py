"""Validate the same laser rays and expiry outside the ROS control process."""

from __future__ import division

import array
import collections
import sys

from depth_projection_process import DepthProjectionProcess, _run


ScanHeader = collections.namedtuple('ScanHeader', 'frame_id')
ScanFrame = collections.namedtuple(
    'ScanFrame', 'header angle_min angle_increment range_min range_max time_increment ranges')


def _check_scan(metadata, payload):
    from explore import RosObservation

    try:
        raw_count = metadata['raw_count']
        values = array.array('d')
        values.frombytes(payload) if hasattr(values, 'frombytes') else values.fromstring(payload)
        if not 1 <= raw_count <= 8192 or len(values) < raw_count or (len(values) - raw_count) % 5:
            return None
        rays, hits = [], []
        for index in range(raw_count, len(values), 5):
            sx, sy, hx, hy, stamp = values[index:index + 5]
            end = (hx, hy)
            rays.append({'start_base': (sx, sy), 'hit_base': end, 'stamp': stamp})
            hits.append({'point_base': end, 'stamp': stamp})
        message = ScanFrame(ScanHeader(metadata['frame_id']), *(
            tuple(metadata['scan_info']) + (values[:raw_count].tolist(),)))
        scene = {'status': metadata['status'], 'coverage': {'sensor_fresh': metadata['sensor_fresh']},
                 'rays': rays, 'occupied_hits': hits}
        observer = RosObservation.__new__(RosObservation)
        observer.epoch = metadata['epoch']
        observer.extrinsics = tuple(metadata['extrinsics'])
        observer._split_validation = True
        stamp, now = metadata['stamp'], metadata['now']
        accepted = observer._cached_newest_scan_safe_locked(
            {'scene': scene, 'pose': tuple(metadata['sample_pose'])},
            {'stamp': stamp, 'message': message},
            {'pose': tuple(metadata['aligned_pose'])}, {'pose': tuple(metadata['pose'])}, now)
        return {'stamp': stamp, 'accepted': accepted, 'reason': observer._scan_veto_issue,
                'valid_until': observer._scan_veto_valid_until if accepted else None}
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None


class ScanValidationProcess(DepthProjectionProcess):
    WORKER_SCRIPT = 'scan_validation_process.py'

    def __init__(self):
        self._histories = collections.deque(maxlen=2)
        DepthProjectionProcess.__init__(self)

    def _packed_history(self, scene):
        for existing, packed in list(self._histories):
            if existing is scene:
                return packed
        rays, hits = scene['rays'], scene['occupied_hits']
        if not isinstance(rays, list) or not isinstance(hits, list) or len(rays) != len(hits) or len(rays) > 50000:
            return None
        packed = array.array('d')
        for ray, hit in zip(rays, hits):
            start, end, stamp = ray['start_base'], ray['hit_base'], ray['stamp']
            if len(start) != 2 or len(end) != 2 or end != hit['point_base'] or stamp != hit['stamp']:
                return None
            packed.extend((start[0], start[1], end[0], end[1], stamp))
        payload = packed.tobytes() if hasattr(packed, 'tobytes') else packed.tostring()
        self._histories.appendleft((scene, payload))
        return payload

    def check(self, epoch, extrinsics, sample, scan, aligned, odom, now):
        try:
            scene = sample['scene']
            history = self._packed_history(scene)
            if history is None:
                return None
            message = scan['message']
            metadata = {'epoch': epoch, 'extrinsics': extrinsics, 'stamp': scan['stamp'], 'now': now,
                        'sample_pose': sample['pose'], 'aligned_pose': aligned['pose'], 'pose': odom['pose'],
                        'frame_id': message.header.frame_id, 'raw_count': len(message.ranges),
                        'scan_info': [message.angle_min, message.angle_increment, message.range_min,
                                      message.range_max, message.time_increment],
                        'status': scene['status'], 'sensor_fresh': scene['coverage']['sensor_fresh']}
            raw = array.array('d', message.ranges)
            raw = raw.tobytes() if hasattr(raw, 'tobytes') else raw.tostring()
            return self.project(metadata, raw + history)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return None


if __name__ == '__main__':
    from explore import RosObservation  # noqa: F401 - load geometry before the ready signal
    sys.exit(_run(_check_scan))
