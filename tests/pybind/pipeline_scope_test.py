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

"""Checks that the pipeline a `with` block makes current is scoped to that block and to the
thread that entered it. Operators build into Pipeline.current(), so a pipeline that stays current
after its block ends, or that leaks into another thread, silently builds the wrong graph."""

import sys
import threading

from amd.rocal.pipeline import Pipeline

print('rocAL PyBind Pipeline Scope')

failures = []


def check(condition, description):
    print(('PASSED: ' if condition else 'FAILED: ') + description)
    if not condition:
        failures.append(description)


def make_pipeline():
    return Pipeline(batch_size=1, num_threads=1, device_id=0, rocal_cpu=True)


def test_no_pipeline_outside_a_with_block():
    check(Pipeline.current() is None, 'no pipeline is current outside a with block')


def test_nesting_restores_the_outer_pipeline():
    train = make_pipeline()
    val = make_pipeline()
    with train:
        check(Pipeline.current() is train, 'the entered pipeline is current')
        with val:
            check(Pipeline.current() is val, 'the inner pipeline is current inside its block')
        check(Pipeline.current() is train,
              'the outer pipeline is current again after the inner block')
    check(Pipeline.current() is None, 'no pipeline is current after the outer block')


def test_an_exception_still_leaves_the_block():
    pipeline = make_pipeline()
    try:
        with pipeline:
            raise ValueError('raised on purpose')
    except ValueError:
        pass
    check(Pipeline.current() is None, 'a raising with block still stops being current')


def test_threads_do_not_see_each_others_pipelines():
    """Each thread builds its own pipeline at the same time, as one loader per worker does."""
    thread_count = 4
    results = [None] * thread_count
    at_the_barrier = threading.Barrier(thread_count)

    def build(index):
        pipeline = make_pipeline()
        # Enter together, so every thread is inside its own block while the others are too.
        at_the_barrier.wait()
        with pipeline:
            for _ in range(2000):
                if Pipeline.current() is not pipeline:
                    results[index] = False
                    return
        results[index] = Pipeline.current() is None

    # Switch threads far more often than the 5 ms default, to widen the window a leaked
    # process-wide pipeline would be seen through.
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    try:
        threads = [threading.Thread(target=build, args=(index,))
                   for index in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(previous_interval)

    check(all(results), 'each thread only ever sees the pipeline it entered')


def test_a_fresh_thread_starts_with_no_pipeline():
    pipeline = make_pipeline()
    seen = []

    with pipeline:
        thread = threading.Thread(target=lambda: seen.append(Pipeline.current()))
        thread.start()
        thread.join()

    check(seen == [None], 'a pipeline entered in one thread is not current in another')


test_no_pipeline_outside_a_with_block()
test_nesting_restores_the_outer_pipeline()
test_an_exception_still_leaves_the_block()
test_threads_do_not_see_each_others_pipelines()
test_a_fresh_thread_starts_with_no_pipeline()

if failures:
    sys.exit('rocAL PyBind Pipeline Scope: {} check(s) failed'.format(len(failures)))
print('rocAL PyBind Pipeline Scope: all checks passed')
