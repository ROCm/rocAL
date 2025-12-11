# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

from amd.rocal.plugin.pytorch import ROCALNumpyIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import numpy as np
import sys
import os
import shutil
import timeit


def main():
    if len(sys.argv) < 3:
        print("Please pass cpu/gpu batch_size input_channels(optional)")
        exit(0)
    try:
        path = "output_folder/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    rocal_cpu = True if sys.argv[1] == "cpu" else False
    batch_size = int(sys.argv[2])
    input_channels = int(sys.argv[3]) if len(sys.argv) == 4 else 16 # Defaulting to 16 channels if not provided

    local_rank = 0
    world_size = 1

    data_path = 'dummy_inputs'
    input_shape = (224, 224, input_channels)
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    for i in range(10):
        # Creating random numpy arrays with specified number of channels
        np_array = np.random.randint(0, 256, size=input_shape)
        np.save(os.path.join(data_path, f'{i}.npy'), np_array.astype(np.float32))

    # This pipeline reads multi channel numpy files, normalizes the values to (-1,1) and transposes them
    pipe = Pipeline(batch_size=batch_size, num_threads=8, device_id=local_rank, seed=local_rank+10, rocal_cpu=rocal_cpu, tensor_dtype=types.FLOAT,
                    tensor_layout=types.NCHW, prefetch_queue_depth=6, output_memory_type=types.HOST_MEMORY if rocal_cpu else types.DEVICE_MEMORY)
    with pipe:
        numpy_reader_output = fn.readers.numpy(file_root=data_path, output_layout=types.NHWC, shard_id=local_rank, num_shards=world_size, random_shuffle=True, seed=local_rank)
        # Using data mean and stddev as 127.5 to normalize between -1 and 1
        normalized_output = fn.normalize(numpy_reader_output, axes=[0,1], mean=[127.5]*input_shape[-1], stddev=[127.5]*input_shape[-1], output_datatype=types.FLOAT)
        transposed_output = fn.transpose(normalized_output, perm=[2,0,1], output_layout=types.NCHW, output_dtype=types.FLOAT)
        pipe.set_outputs(transposed_output)

    pipe.build()
    imageIteratorPipeline = ROCALNumpyIterator(pipe, device='cpu' if rocal_cpu else 'gpu', device_id=local_rank)
    start = timeit.default_timer()

    for epoch in range(3):
        print(
            "+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++", epoch)
        for i, it in enumerate(imageIteratorPipeline):
            print(
                "************************************** i *************************************", i)
            batch = it[0]
            print(batch.shape)
        imageIteratorPipeline.reset()
    print("*********************************************************************")
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)


if __name__ == "__main__":
    main()
