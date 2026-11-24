# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Checks that a fixed crop starts where crop_pos_x and crop_pos_y put it.

The positions are fractions of the space around the crop, so the crop starts at
round(pos * (input size - crop size)), with the positions clamped to [0, 1]. They used to be
stored as unsigned pixel offsets, which made 0.25 and 0.5 start at 0 and 1.0 start at 1. See
ROCm/rocAL#533. crop_mirror_normalize positions the crop through the same anchors and the same
clamp, so it is checked with an in-range and an out-of-range position. A NaN position passes
through a clamp unchanged, so it has to be rejected instead, for both operators.
"""

import argparse
import os
import sys
import traceback

import numpy as np

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types

# CMake passes the installed data (<ROCM_PATH>/share/rocal/test/data/images/AMD-tinyDataSet) and
# --gpu on a HIP build. Without arguments the script reads the data from the source tree and runs
# on the CPU only.
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
parser = argparse.ArgumentParser(description='rocAL fixed crop position test')
parser.add_argument('--data', default=os.path.join(ROOT, 'data', 'images', 'AMD-tinyDataSet'),
                    help='folder of the test images')
parser.add_argument('--gpu', action='store_true', help='also run the GPU pipeline (needs a HIP build)')
args = parser.parse_args()
DATA = args.data
SIZE = 300

# (crop height, crop width, crop_pos_x, crop_pos_y); None leaves the default position
CROP_CASES = [
    (200, 200, 0.0, 0.0),
    (200, 200, 0.25, 0.25),
    (200, 200, 1.0, 1.0),
    (200, 200, 0.25, 0.75),  # x and y independent of each other
    (100, 200, 0.5, 0.5),    # rectangular
    (260, 120, 1.0, 0.0),    # rectangular, at opposite edges
    (200, 200, 0.333, 0.666),
    (100, 200, None, None),  # the default position, a centre crop
    (100, 200, 1.5, 1.5),    # clamped to 1.0
    (100, 200, -0.5, -0.5),  # clamped to 0.0
]
CMN_CASES = [
    (100, 200, 0.25, 0.75),
    (100, 200, 1.5, -0.5),   # clamped to (1.0, 0.0)
]

print('rocAL PyBind Crop Position')

failures = []


def check(condition, description):
    print(('PASSED: ' if condition else 'FAILED: ') + description)
    if not condition:
        failures.append(description)


def expected_slice(source, h, w, x, y):
    def offset(pos, input_size, crop_size):
        pos = 0.5 if pos is None else min(max(pos, 0.0), 1.0)
        return int(np.rint(pos * (input_size - crop_size)))
    x1, y1 = offset(x, source.shape[1], w), offset(y, source.shape[0], h)
    return source[y1:y1 + h, x1:x1 + w], (x1, y1)


def new_pipeline(rocal_cpu):
    return Pipeline(batch_size=1, num_threads=1, device_id=0, seed=1, rocal_cpu=rocal_cpu,
                    tensor_layout=types.NHWC, tensor_dtype=types.UINT8)


def decode_and_resize():
    jpegs, _ = fn.readers.file(file_root=DATA)
    images = fn.decoders.image(jpegs, file_root=DATA, output_type=types.RGB, random_shuffle=False)
    return fn.resize(images, resize_width=SIZE, resize_height=SIZE)


def run(rocal_cpu):
    pipe = new_pipeline(rocal_cpu)
    with pipe:
        base = decode_and_resize()
        outputs = [base]
        for h, w, x, y in CROP_CASES:
            position = {} if x is None else {'crop_pos_x': x, 'crop_pos_y': y}
            outputs.append(fn.crop(base, crop=(h, w), **position))
        for h, w, x, y in CMN_CASES:
            # no mirror, mean 0 and std 1 leave the pixels as they are, so the output is the crop
            outputs.append(fn.crop_mirror_normalize(base, crop=(h, w), crop_pos_x=x, crop_pos_y=y,
                                                    mean=[0, 0, 0], std=[1, 1, 1], mirror=0,
                                                    output_layout=types.NHWC,
                                                    output_dtype=types.UINT8))
        pipe.set_outputs(*outputs)
    try:
        pipe.build()
        # a failed run would leave zero-filled outputs, which the source and every crop would
        # then match, so the status has to be OK before anything is compared
        status = pipe.rocal_run()
        if status != types.OK:
            raise RuntimeError('rocal_run returned status %s' % status)
        arrays = []
        for tensor in pipe.get_output_tensors():
            array = np.zeros(tensor.dimensions(), dtype=tensor.dtype())
            tensor.copy_data(array)
            arrays.append(array[0])
    finally:
        pipe.rocal_release()
    return arrays[0], arrays[1:]


def test_the_crop_starts_where_the_position_puts_it(device):
    try:
        source, crops = run(rocal_cpu=(device == 'cpu'))
    except Exception:
        traceback.print_exc()
        check(False, '%s: a pipeline of fixed crops builds and runs' % device)
        return

    named = [('crop', case) for case in CROP_CASES] + [('crop_mirror_normalize', case) for case in CMN_CASES]
    for (op, (h, w, x, y)), crop in zip(named, crops):
        expected, (x1, y1) = expected_slice(source, h, w, x, y)
        position = 'the default position' if x is None else '(%g, %g)' % (x, y)
        check(crop.shape[:2] == (h, w) and np.array_equal(crop, expected),
              '%s: %s %dx%d at %s starts at (%d, %d)' % (device, op, w, h, position, x1, y1))


NaN = float('nan')
# (name, builds the operator from the resized image)
NAN_CASES = [
    ('crop with crop_pos_x NaN', lambda base: fn.crop(base, crop=(200, 200), crop_pos_x=NaN, crop_pos_y=0.5)),
    ('crop with crop_pos_y NaN', lambda base: fn.crop(base, crop=(200, 200), crop_pos_x=0.5, crop_pos_y=NaN)),
    ('crop_mirror_normalize with crop_pos_x NaN',
     lambda base: fn.crop_mirror_normalize(base, crop=(100, 200), crop_pos_x=NaN, crop_pos_y=0.5,
                                           mean=[0, 0, 0], std=[1, 1, 1], mirror=0,
                                           output_layout=types.NHWC, output_dtype=types.UINT8)),
]


def test_a_nan_position_is_rejected(device):
    for name, make in NAN_CASES:
        pipe = new_pipeline(rocal_cpu=(device == 'cpu'))
        message = None
        try:
            with pipe:
                base = decode_and_resize()
                pipe.set_outputs(make(base))
            pipe.build()
            pipe.rocal_run()
        except RuntimeError as error:
            message = str(error)
        finally:
            pipe.rocal_release()
        check(message is not None and 'not NaN' in message, '%s: %s is rejected' % (device, name))


test_the_crop_starts_where_the_position_puts_it('cpu')
test_a_nan_position_is_rejected('cpu')
if args.gpu:
    test_the_crop_starts_where_the_position_puts_it('gpu')
    test_a_nan_position_is_rejected('gpu')
else:
    print('SKIPPED: gpu, not a HIP build (no --gpu)')

print()
if failures:
    print('%d check(s) FAILED' % len(failures))
    sys.exit(1)
print('all checks passed')
