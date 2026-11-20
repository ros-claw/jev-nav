#!/usr/bin/env python2
"""Build the optional local scan accelerator; no ROS or robot I/O."""
from __future__ import print_function

import os
import shutil
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

HERE = os.path.dirname(os.path.abspath(__file__))
build_source = os.path.join(HERE, 'build', 'cython')
if not os.path.isdir(build_source):
    os.makedirs(build_source)
source = os.path.join(build_source, '_limo_scan_accel.pyx')
shutil.copyfile(os.path.join(HERE, '_limo_scan_accel.pyx'), source)
extension = Extension('_limo_scan_accel', [source],
                      extra_compile_args=['-O2', '-ffp-contract=off'])
setup(name='limo-scan-accel', ext_modules=cythonize([extension]))
