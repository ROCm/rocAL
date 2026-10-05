#!/usr/bin/env python3
# Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""fn.readers.tfrecord with a feature key missing from the key map, user style (no error handling).

amd.rocal.readers.tfrecord() prints the required keys and calls exit(), i.e. status 0, so the
misconfigured script reports success. A correct library raises (non-zero status).
"""
import tempfile

import amd.rocal.fn as fn
import amd.rocal.types as types
from amd.rocal.pipeline import Pipeline

with tempfile.TemporaryDirectory() as d:
    pipe = Pipeline(batch_size=2, num_threads=1, device_id=0, seed=1, rocal_cpu=True, tensor_layout=types.NHWC)
    with pipe:
        key_map = {"image/class/label": "image/class/label", "image/filename": "image/filename"}
        features = {"image/encoded": None, "image/class/label": None, "image/filename": None}
        fn.readers.tfrecord(d, key_map, features, reader_type=0)
print("CONTROL-INVALID: tfrecord() accepted a feature key that is missing from the key map")
