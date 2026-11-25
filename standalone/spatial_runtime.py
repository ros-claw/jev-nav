"""ROS 1 RTAB-Map adapter for optional persistent room memory."""
from __future__ import division

import json
import math
import os
import tempfile
import threading

from navigation import Pose, ROOM_CURVATURE, ROOM_LENGTH, ready
from native_sensing_probe import _yaw
from relay import _monotonic
from spatial_memory import MapFix, SpatialMemory, compose


def closure_for_graph(graph, closure, epoch_started):
    if closure is not None and closure[0] >= epoch_started and closure[1] == graph[1].revision:
        return closure[2]
    return None


def _planar_pose(point, orientation):
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    values = (point.x, point.y, x, y, z, w)
    if (any(math.isnan(v) or math.isinf(v) for v in values) or abs(x) > .05 or abs(y) > .05
            or abs(x*x + y*y + z*z + w*w - 1.) > .01):
        raise ValueError('invalid planar mapper pose')
    return Pose(point.x, point.y, _yaw(orientation))


class SpatialRuntime(object):
    def __init__(self, rospy, path, map_id):
        from rtabmap_ros.msg import Info, MapGraph

        self.rospy, self.path = rospy, os.path.abspath(path)
        state = None
        if os.path.exists(self.path):
            with open(self.path) as stream:
                state = json.load(stream)
        self.memory = SpatialMemory(map_id, state)
        self._graph = self._closure = self._epoch = None
        self._view_grid = None
        self._epoch_started = _monotonic()
        self._synced = 0.
        self._save_after = 0.
        self.status = 'awaiting_map'
        self.write_error = None
        self.lock = threading.Lock()
        self._wake = threading.Event()
        self._closing = False
        self._pending = None
        self.subscriptions = (rospy.Subscriber('/rtabmap/mapGraph', MapGraph, self._on_graph, queue_size=1),
                              rospy.Subscriber('/rtabmap/info', Info, self._on_info, queue_size=1))
        self._writer = threading.Thread(target=self._write, name='spatial-memory')
        self._writer.daemon = True
        self._writer.start()

    def _stamp(self, header):
        from explore import sample_clocks, source_time

        pair = sample_clocks(_monotonic, lambda: self.rospy.Time.now().to_sec())
        return None if pair is None else source_time(pair[0], pair[1], header.stamp.to_sec(), 3.)

    def _on_graph(self, message):
        try:
            at = self._stamp(message.header)
            if (at is None or message.header.frame_id.lstrip('/') != 'map'
                    or len(message.posesId) != len(message.poses)):
                return
            revision = message.header.stamp.to_nsec()
            current = self._graph
            if current is not None and revision <= current[1].revision:
                return
            anchors = {key: _planar_pose(p.position, p.orientation)
                       for key, p in zip(message.posesId, message.poses) if key > 0}
            transform = message.mapToOdom
            self._graph = at, MapFix(self.memory.map_id, revision,
                                     _planar_pose(transform.translation, transform.rotation), anchors)
        except (AttributeError, ValueError, TypeError):
            return

    def _on_info(self, message):
        at = self._stamp(message.header)
        if at is not None and message.loopClosureId > 0:
            self._closure = at, message.header.stamp.to_nsec(), message.loopClosureId

    def apply(self, view, room, now):
        if not ready(view, now):
            return view
        with self.lock:
            if self._epoch is not None and self._epoch != view.epoch:
                self.memory.reset_registration()
                self._epoch_started = now
            self._epoch = view.epoch
            graph, closure = self._graph, self._closure
            if graph is None or not 0 <= now - graph[0] < 3.:
                self.memory.update_fix(None)
                self.status = 'awaiting_map'
            else:
                matched = closure_for_graph(graph, closure, self._epoch_started)
                self.status = 'registered' if self.memory.update_fix(graph[1], matched) else 'awaiting_relocalization'
            if self.memory.fixed is not None:
                self.memory.visit(view.pose)
                if now >= self._synced and room is not None:
                    self.memory.clear_depth(room.cleared_depth)
                    room.cleared_depth.clear()
                    self.memory.record_depth(room.confirmed_depth(only_changed=True))
                    self._synced = now + .20
                if self.memory.changed and now >= self._save_after:
                    self._queue_save()
                    self._save_after = now + 1.
            projected = self.memory.apply(view)
            if self.memory.fixed is not None and not projected.ready:
                self.status = 'awaiting_hazard_projection'
            self._view_grid = (view.grid_at, projected.grid)
            return projected

    def context(self, request):
        with self.lock:
            if request.epoch != self._epoch or self._view_grid is None:
                return []
            at, grid = self._view_grid
            return self.memory.targets(request, ROOM_LENGTH, ROOM_CURVATURE,
                                       grid if abs(at - request.at) < .20 else None)

    def location(self, pose):
        with self.lock:
            fix = self.memory.fixed
            if fix is None or pose is None:
                return {'pose_valid': False}
            point = compose(fix.odom_to_map, pose)
            anchor = self.memory._anchor(point)
            return {'pose_valid': True, 'map_id': fix.map_id, 'map_revision': fix.revision,
                    'anchor_keyframe_id': anchor, 'x': round(point.x, 3), 'y': round(point.y, 3)}

    def _queue_save(self):
        self._pending = self.memory.snapshot()
        self.memory.changed = False
        self._wake.set()

    def _write(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            with self.lock:
                data, self._pending = self._pending, None
                closing = self._closing
            if data is not None:
                temp = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(self.path), delete=False) as stream:
                        temp = stream.name
                        json.dump(data, stream, allow_nan=False, separators=(',', ':'))
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.rename(temp, self.path)
                    temp = None
                    self.write_error = None
                except (IOError, OSError, ValueError) as error:
                    self.write_error = type(error).__name__
                finally:
                    if temp is not None:
                        os.unlink(temp)
            if closing:
                return

    def close(self):
        for subscription in self.subscriptions:
            subscription.unregister()
        with self.lock:
            self._closing = True
            if self.memory.changed or self.write_error:
                self._queue_save()
            self._wake.set()
        self._writer.join(2.)
