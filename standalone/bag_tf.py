"""Bounded TF population for an explicitly optimistic offline bag replay."""
from __future__ import division

import math


def cache_duration(start, end):
    try:
        duration = end - start
        if (math.isnan(duration) or math.isinf(duration) or duration < 0 or duration > 60):
            raise ValueError('bag duration outside offline replay bound')
        return max(1., duration + 1.)
    except (TypeError, OverflowError):
        raise ValueError('invalid bag duration')


def populate_transforms(tree, records):
    failures = 0
    for topic, message, _ in records:
        for transform in message.transforms:
            try:
                if topic == '/tf_static':
                    tree._buffer.set_transform_static(transform, 'recorded_bag')
                elif topic == '/tf':
                    tree.setTransform(transform, 'recorded_bag')
                else:
                    failures += 1
            except Exception:
                failures += 1
    return failures
