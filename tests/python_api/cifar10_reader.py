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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.generic import ROCALGenericIterator
import amd.rocal.fn as fn
import sys
import os
import random

def draw_patches(img, idx):
    # image is expected as a numpy array
    import cv2
    img = img.transpose([1, 2, 0])
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_folder/cifar10_reader/" + str(idx) +
                "_" + "train" + ".png", image)

def main():
    if len(sys.argv) < 4:
        print('Please pass cifar10_folder cpu/gpu batch_size')
        exit(0)
    try:
        path = "output_folder/cifar10_reader/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    if (sys.argv[2] == "cpu"):
        rocal_cpu = True
    else:
        rocal_cpu = False
    batch_size = int(sys.argv[3])
    num_threads = 1
    device_id = 0
    local_rank = 0
    world_size = 1
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)

    pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu)

    with pipeline:
        cifar10_reader_output = fn.readers.cifar10(file_root=data_path, shard_id=local_rank, num_shards=world_size)
        pipeline.set_outputs(cifar10_reader_output)

    pipeline.build()

    cifar10IteratorPipeline = ROCALGenericIterator(pipeline)
    cnt = 0
    for epoch in range(1):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , it in enumerate(cifar10IteratorPipeline):
            for img in it[0]:
                cnt += 1
                draw_patches(img[0], cnt)
        cifar10IteratorPipeline.reset()
    print("*********************************************************************")


if __name__ == '__main__':
    main()