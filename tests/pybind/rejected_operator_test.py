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
import tempfile
import traceback

import numpy as np

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import rocal_pybind as b

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATA = os.path.join(ROOT, 'data', 'images', 'AMD-tinyDataSet')
VIDEO = os.path.join(ROOT, 'data', 'videos')
BATCH = 4
# A pipeline reports the smallest sample count among its loaders, so this has to stay above the
# number of sequences in data/videos for a leftover video loader to be visible in it.
NUMPY_SAMPLES = 256

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


def write_numpy_samples(directory):
    for index in range(NUMPY_SAMPLES):
        np.save(os.path.join(directory, 'sample_%03d.npy' % index),
                np.full((16, 16), index, dtype=np.float32))


def check_only_the_valid_loader_is_counted(pipeline, description):
    """A rejected loader left in the graph still counts, and drags the pipeline down to its own
    sample count -- zero when its init() failed, the clip's own count when only the arguments
    that follow it were rejected."""
    built = False
    try:
        pipeline.build()
        built = True
    except Exception:
        traceback.print_exc()
    check(built, 'a pipeline whose valid loader follows ' + description + ' builds')

    if built:
        check(pipeline.get_remaining_images() == NUMPY_SAMPLES,
              "the pipeline reports the valid loader's samples, not " + description + "'s")


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


def test_a_loader_rejected_by_its_init_does_not_count_towards_the_pipeline():
    # Loaders are rolled back through the same path, but only the two ImageLoaderNode
    # specializations record a PipelineOperator - the video, numpy, audio, CIFAR-10 and
    # fused-crop ones register a root node, a loader module and a graph id and nothing else. A
    # rejected loader left among them keeps raising the loader count, which is what decides the
    # multi-graph path and how many samples the pipeline reports.
    with tempfile.TemporaryDirectory() as numpy_root:
        write_numpy_samples(numpy_root)

        pipe = Pipeline(batch_size=2, num_threads=1, device_id=0, rocal_cpu=True)
        with pipe:
            raised = False
            try:
                # rocalVideoFileSource checks only the sequence length, so num_shards=0 is left
                # for VideoLoaderNode::init() to reject, once the loader is already registered
                fn.readers.video(sequence_length=1, file_root=VIDEO, num_shards=0,
                                 image_type=types.RGB)
            except RuntimeError:
                raised = True
            check(raised, 'a loader its init() rejects raises')

            # a later loader, entirely valid. Numpy because the rejected video reader has
            # already taken the pipeline's one metadata reader.
            pipe.set_outputs(fn.readers.numpy(file_root=numpy_root, num_shards=1))

        check_only_the_valid_loader_is_counted(pipe, 'a loader rejected by its init()')


def test_a_loader_whose_later_arguments_are_rejected_does_not_count_either():
    # rocalVideoFileResize used to insert and initialize the loader before checking its resize
    # arguments. Those checks therefore ran with the loader already in the graph, and the handler
    # only discards the call's output tensor, which is still null at that point -- so the loader
    # survived with its own sample count intact and capped the pipeline at the rejected clip's.
    with tempfile.TemporaryDirectory() as numpy_root:
        write_numpy_samples(numpy_root)

        pipe = Pipeline(batch_size=2, num_threads=1, device_id=0, rocal_cpu=True)
        with pipe:
            raised = False
            try:
                # no size given by any of the four means accepted
                fn.readers.video_resize(sequence_length=1, file_root=VIDEO,
                                        resize_width=0, resize_height=0,
                                        image_type=types.RGB)
            except RuntimeError:
                raised = True
            check(raised, 'a loader whose resize arguments are rejected raises')

            pipe.set_outputs(fn.readers.numpy(file_root=numpy_root, num_shards=1))

        check_only_the_valid_loader_is_counted(pipe, 'a rejected resize')


test_an_operator_rejected_by_its_init_is_not_left_in_the_graph()
test_a_loader_rejected_by_its_init_does_not_count_towards_the_pipeline()
test_a_loader_whose_later_arguments_are_rejected_does_not_count_either()
test_two_identical_failures_both_report_the_reason()

print()
if failures:
    print('%d check(s) FAILED' % len(failures))
    sys.exit(1)
print('all checks passed')
