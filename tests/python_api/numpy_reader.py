# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.generic import ROCALNumpyIterator
import amd.rocal.fn as fn
import amd.rocal.types as types

import os
import shutil
import cv2
import numpy as np
from parse_config import parse_args


def draw_patches(image, idx, args=None):
    # image is expected as a numpy array
    if not args.NHWC:
        image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_folder/numpy_reader/" + str(idx) + ".png", image)


def main():
    args = parse_args()
    # Args
    data_path = args.image_dataset_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    num_threads = args.num_threads
    random_seed = args.seed
    local_rank = args.local_rank
    world_size = args.world_size
    output_layout = types.NHWC if args.NHWC else types.NCHW
    augmentation_name = args.augmentation_name

    try:
        path = "output_folder/numpy_reader/"
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as error:
        print(error)

    if augmentation_name == "log1p":
        # Create int16 numpy files since log1p only supports int16 inputs
        new_data_path = 'log1p_inputs'
        if os.path.exists(new_data_path):
            shutil.rmtree(new_data_path)
        os.makedirs(new_data_path)
        for i in range(10):
            np_array = np.random.randint(0, 256, (128, 128, 128, 4))
            np.save(os.path.join(new_data_path, f'{i}.npy'), np_array.astype(np.int16))
        data_path = new_data_path
        input_layout = types.NDHWC
    if augmentation_name == "normalize" or augmentation_name == "transpose":
        # Create random input tensors with multi channels
        new_data_path = f'{augmentation_name}_inputs'
        if os.path.exists(new_data_path):
            shutil.rmtree(new_data_path)
        os.makedirs(new_data_path)
        for i in range(10):
            np_array = np.random.randint(0, 256, (768, 1152, 16))
            # Normalize kernel only supports F32->F32 variant so creating F32 inputs
            np_dtype = np.float32 if augmentation_name == "normalize" else np.uint8
            np.save(os.path.join(new_data_path, f'{i}.npy'), np_array.astype(np_dtype))
        data_path = new_data_path
        input_layout = types.NHWC
    if augmentation_name == "resize":
        input_layout = types.NHWC

    pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads,
                        device_id=local_rank, seed=random_seed, rocal_cpu=rocal_cpu)

    with pipeline:
        # Numpy reader can be used to read inputs of different dtypes and layouts (including voxel layouts) - need to specify output layout correctly
        numpy_reader_output = fn.readers.numpy(file_root=data_path, shard_id=local_rank, num_shards=world_size, output_layout=input_layout)
        if augmentation_name == "log1p":
            output = fn.log1p(numpy_reader_output)
        elif augmentation_name == "normalize":
            output = fn.normalize(numpy_reader_output, axes=[0,1], mean=[127.5]*16, stddev=[127.5]*16, output_datatype=types.FLOAT)
        elif augmentation_name == "transpose":
            output = fn.transpose(numpy_reader_output, perm=[2,0,1], output_layout=types.NCHW)
        else:
            output = fn.resize(numpy_reader_output, resize_width=400, resize_height=400, output_layout=output_layout)
        pipeline.set_outputs(output)

    pipeline.build()

    cnt = 0
    numpyIteratorPipeline = ROCALNumpyIterator(pipeline)
    print(f"##############################  RUNNING {augmentation_name.upper()} ############################")
    for epoch in range(args.num_epochs):
        print("epoch:: ", epoch)
        for i, [batch] in enumerate(numpyIteratorPipeline):
            if i == 0 and args.print_tensor:
                print(batch)
            for img in batch:
                if args.display:
                    # Can only be used with resize augmentation since other augmentations outputs cannot be visualized
                    draw_patches(img, cnt, args)
                cnt += 1
        numpyIteratorPipeline.reset()
    print("##############################  NUMPY READER SUCCESS  ############################")


if __name__ == '__main__':
    main()
