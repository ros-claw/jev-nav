"""Load the optional locally built accelerator or retain the Python fallback."""
from __future__ import division

import os
import sys


def load_native_scan():
    build = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'build', 'py%d' % sys.version_info[0]))
    if not os.path.isdir(build):
        return None
    if build not in sys.path:
        sys.path.insert(0, build)
    try:
        import _limo_scan_accel
        origin = os.path.realpath(_limo_scan_accel.__file__)
        if os.path.dirname(origin) != build:
            return None
        return _limo_scan_accel
    except (AttributeError, ImportError, OSError):
        return None
