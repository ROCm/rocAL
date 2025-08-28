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

from amd.rocal.plugin.generic import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import os, sys
import random
import cv2
import numpy as np


def random_augmentation(probability, augmented, original):
    import random
    condition = random.random() < probability
    neg_condition = condition ^ True
    return condition * augmented + neg_condition * original

def brightness_fn(img):
    brightness_scale = random_augmentation(0.5, random.uniform(0.7, 1.3), 1.0)
    print('Brightness scale: ', brightness_scale)
    return (img * brightness_scale).astype(np.uint8)  # Casting is needed since it will return fp64 outputs otherwise

def draw_patches(image, idx, layout="nchw", dtype="fp32", device="cpu"):
    # image is expected as a numpy array
    if layout == "nchw":
        image = image.transpose([1, 2, 0])
    if dtype in ["fp16", "fp32"]:
        image = image.astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_folder/python_function/" + str(idx) +
                "_" + "train" + ".png", image*255)


def main():
    if len(sys.argv) < 3:
        print("Please pass image_folder batch_size")
        exit(0)
    try:
        path = "output_folder/python_function/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    rocal_cpu = True  # Only supported for Host backend
    batch_size = int(sys.argv[2])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)

    local_rank = 0
    world_size = 1
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                                   seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=types.NHWC, tensor_dtype=types.FLOAT16)
    with pipe:
        jpegs, _ = fn.readers.file(file_root=data_path)
        decode = fn.decoders.image(jpegs, file_root=data_path, output_type=types.RGB, shard_id=local_rank, num_shards=world_size, random_shuffle=False)
        rand_brightness_output = fn.python_function(decode, function = brightness_fn, dtype=types.UINT8, layout=types.NHWC)
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmnp = fn.crop_mirror_normalize(rand_brightness_output,
                                        output_layout=types.NHWC,
                                        output_dtype=types.FLOAT16,
                                        crop=(224, 224),
                                        mirror=flip_coin,
                                        mean=[0.485 * 255, 0.456 *
                                              255, 0.406 * 255],
                                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        pipe.set_outputs(cmnp)
    pipe.build()
    
    # Dataloader
    data_loader = ROCALClassificationIterator(pipe, device="cpu", device_id=local_rank)
    cnt = 0

    # Enumerate over the Dataloader
    for epoch in range(3):
        print(
            "+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++", epoch)
        for i, it in enumerate(data_loader):
            print(
                "************************************** i *************************************", i)
            for img in it[0]:
                cnt += 1
                draw_patches(img[0], cnt, layout="nhwc",
                             dtype="fp16", device=rocal_cpu)
        data_loader.reset()
    print("*********************************************************************")


if __name__ == "__main__":
    main()
