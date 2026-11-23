import math
from collections import namedtuple

import pytest

from trail_recovery import front_registration, straight_trail


def test_retrace_requires_bounded_measured_travel_ending_at_the_reference():
    odom = [(index * .05, index * .001, 0., .02, 0., 0.) for index in range(411)]
    trace = {'mode': 'supervised_static_room', 'relay_exited': True,
             'trajectory': {'clock': 'host_monotonic_receipt', 'odom': odom}}
    reference = {'x': .41, 'y': 0., 'yaw': 0.}
    assert straight_trail(trace, reference, .32)['measured_trail_m'] >= .36
    with pytest.raises(ValueError, match='reference'):
        straight_trail(trace, dict(reference, y=.1), .32)
    odom[250] = (12.5, .25, .03, .02, 0., 0.)
    with pytest.raises(ValueError, match='straight'):
        straight_trail(trace, reference, .32)


def test_front_registration_rejects_unrelated_or_insufficient_returns():
    scan = namedtuple('Scan', 'angle_min angle_increment range_min range_max ranges')
    reference = {'closest_rays': [(angle, .30) for angle in range(-4, 5)]}
    sample = scan(math.radians(-4), math.radians(1), .1, 8., [.325] * 9)
    assert front_registration(sample, reference) == pytest.approx(.025, abs=.001)
    assert front_registration(sample._replace(ranges=[.6] * 9), reference) is None
    assert front_registration(sample._replace(ranges=[.325] * 3 + [float('inf')] * 6), reference) is None
