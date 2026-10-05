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

"""A user-style amd.rocal script whose GPU pipeline fails rocalVerify (a crop far larger than the image).

    rocal_verify_fail.py <jpeg_dir>

No error handling on purpose: a failed build must end the process with a non-zero status. rocAL's
Pipeline.build() prints "Verify graph failed" and calls exit(0) instead (M8), so the script "succeeds".
If build() returns, the controlled failure no longer fails and the check itself is invalid.
"""
import sys

import amd.rocal.fn as fn
import amd.rocal.types as types
from amd.rocal.pipeline import Pipeline

data = sys.argv[1].rstrip("/") + "/"
pipe = Pipeline(batch_size=2, num_threads=1, device_id=0, seed=1, rocal_cpu=False, tensor_layout=types.NHWC)
with pipe:
    jpegs, _ = fn.readers.file(file_root=data)
    images = fn.decoders.image(jpegs, file_root=data, output_type=types.RGB, random_shuffle=False)
    images = fn.crop(images, crop=(100000, 100000))
    pipe.set_outputs(images)
pipe.build()
print("CONTROL-INVALID: build() returned; the oversized crop no longer fails verification")
