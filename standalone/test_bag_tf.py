"""Offline TF loading for diagnostic bag replay."""
import pytest

from bag_tf import cache_duration, populate_transforms


def test_static_and_dynamic_transforms_use_different_buffers():
    class Buffer(object):
        def __init__(self):
            self.static = []
        def set_transform_static(self, transform, authority):
            self.static.append((transform, authority))
    class Tree(object):
        def __init__(self):
            self._buffer, self.dynamic = Buffer(), []
        def setTransform(self, transform, authority):
            self.dynamic.append((transform, authority))
    class Msg(object):
        def __init__(self, value):
            self.transforms = [value]
    tree = Tree()
    assert populate_transforms(tree, [('/tf_static', Msg('s'), None), ('/tf', Msg('d'), None)]) == 0
    assert tree._buffer.static == [('s', 'recorded_bag')]
    assert tree.dynamic == [('d', 'recorded_bag')]


def test_replay_tf_cache_covers_bag_but_has_a_bounded_duration():
    assert cache_duration(100., 115.) == 16.
    with pytest.raises(ValueError):
        cache_duration(100., 200.)
