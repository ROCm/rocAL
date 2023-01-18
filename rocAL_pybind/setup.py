# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from setuptools import setup, find_packages, Extension
from setuptools.dist import Distribution
import sys


if sys.version_info < (3, 0): #TODO: check minimum & max version of python3 required
    sys.exit('rocal Python Package requires Python > 3.0')

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True

setup(
      name='amd-rocal',
      description='AMD ROCm Augmentation Library',
      url='https://github.com/ROCmSoftwarePlatform/rocAL',
      version='1.0.0',
      author='AMD',
      license='Apache License 2.0',
      packages=find_packages(where='@TARGET_NAME@'),
      package_dir={'amd':'@TARGET_NAME@/amd'},
      include_package_data=True,
      package_data={'amd': ['@TARGET_NAME@/lib/@TARGET_PYMODULE_NAME@']},
      distclass=BinaryDistribution
      )
