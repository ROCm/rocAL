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

"""Checks that an operator rocAL rejects leaves nothing of itself behind, and that the exception
reports why.

An operator is added to the graph before its init() validates its arguments, so an operator that
init() rejects has to be taken back out. If it is left in, build() creates it anyway - from the
very arguments init() refused - because create_single_graph() iterates every node. See
ROCm/rocAL#537.
"""

import os
import sys
import traceback

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import rocal_pybind as b

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', '..', 'data', 'images', 'AMD-tinyDataSet')
BATCH = 4

print('rocAL PyBind Rejected Operator')

failures = []


def check(condition, description):
    print(('PASSED: ' if condition else 'FAILED: ') + description)
    if not condition:
        failures.append(description)


def operators_in(pipeline):
    """The operator names the pipeline would serialize, which is what build() walks."""
    blob = b.rocalSerialize(pipeline._handle)
    return [field.decode('latin-1') for field in blob.split(b'\n') if b'Node' in field]


def test_an_operator_rejected_by_its_init_is_not_left_in_the_graph():
    pipe = Pipeline(batch_size=BATCH, num_threads=1, device_id=0, rocal_cpu=True)
    with pipe:
        jpegs, _ = fn.readers.file(file_root=DATA)
        images = fn.decoders.image(jpegs, file_root=DATA, output_type=types.RGB)
        good = fn.brightness(images, brightness=1.0)

        raised = False
        try:
            # num_boxes of length 2 for a batch of 4: EraseNode::init() rejects this, after
            # add_node() has already put the node in the graph.
            fn.erase(images, anchor=[0.0, 0.0], shape=[10.0, 10.0],
                     num_boxes=[1, 1], fill_value=[0.0])
        except RuntimeError:
            raised = True
        check(raised, 'an operator its init() rejects raises')

        pipe.set_outputs(good)

    operators = operators_in(pipe)
    check(not any('Erase' in op for op in operators),
          'the rejected operator is not left in the graph')
    check(any('Brightness' in op for op in operators),
          'the operators that were accepted are still in the graph')

    built = False
    try:
        pipe.build()
        built = True
    except Exception:
        traceback.print_exc()
    check(built, 'a pipeline built from the remaining operators still builds')

    if built:
        ran = False
        try:
            pipe.run()
            ran = True
        except Exception:
            traceback.print_exc()
        check(ran, 'a pipeline built from the remaining operators still runs')


def test_two_identical_failures_both_report_the_reason():
    # The context keeps only the last error message, so a second failure with the same text looks
    # to the bindings like a call that recorded nothing. It must still report the reason.
    other = Pipeline(batch_size=2, num_threads=1, device_id=0, rocal_cpu=True)
    with other:
        foreign_jpegs, _ = fn.readers.file(file_root=DATA)
        foreign = fn.decoders.image(foreign_jpegs, file_root=DATA, output_type=types.RGB)

    pipe = Pipeline(batch_size=2, num_threads=1, device_id=0, rocal_cpu=True)
    messages = []
    with pipe:
        jpegs, _ = fn.readers.file(file_root=DATA)
        fn.decoders.image(jpegs, file_root=DATA, output_type=types.RGB)
        for _ in range(2):
            try:
                # an input from another pipeline, twice, failing the same way each time
                fn.resize(foreign, resize_width=64, resize_height=64)
            except RuntimeError as error:
                messages.append(str(error))

    check(len(messages) == 2, 'both calls with a foreign input raise')
    if len(messages) == 2:
        check('Input tensor is invalid' in messages[0],
              'the first failure reports the reason')
        check('Input tensor is invalid' in messages[1],
              'the second, identical failure also reports the reason')


test_an_operator_rejected_by_its_init_is_not_left_in_the_graph()
test_two_identical_failures_both_report_the_reason()

print()
if failures:
    print('%d check(s) FAILED' % len(failures))
    sys.exit(1)
print('all checks passed')
