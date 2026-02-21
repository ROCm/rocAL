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

import sys
import os
import gc
import time
from amd.rocal.pipeline import pipeline_def, Pipeline
from amd.rocal.plugin.generic import ROCALGenericIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import numpy as np

seed = 1549361629
image_dir = os.environ.get("ROCM_PATH", "/opt/rocm") + "/share/rocal/test/data/images/AMD-tinyDataSet"
batch_size = 1
gpu_id = 0

def get_image_names(pipe: Pipeline):
    batch_size = pipe._batch_size  # pylint: disable=protected-access
    name_lengths = np.zeros(batch_size, dtype=np.int32)
    total_length = pipe.get_image_name_length(name_lengths)
    if total_length == 0:
        return [""] * batch_size
    raw_bytes = pipe.get_image_name(total_length)
    names = []
    offset = 0
    for length in name_lengths:
        if length == 0:
            names.append("")
            continue
        chunk = raw_bytes[offset : offset + length]
        names.append(chunk.decode("utf-8", errors="replace"))
        offset += length
    return names

@pipeline_def(seed=seed)
def image_decoder_pipeline(device="cpu", path=image_dir):
    jpegs, labels = fn.readers.file(file_root=path)
    images = fn.decoders.image(
        jpegs,
        file_root=path,
        device=device,
        output_type=types.RGB,
        shard_id=0,
        num_shards=1,
        random_shuffle=True,
    )
    # Keep ops simple and deterministic for checkpoint debug
    return fn.brightness_fixed(images)


def create_and_checkpoint(bs, rocal_device, rocal_cpu, img_folder, ckpt_path=None):
    """Create a pipeline, advance a few iterations, dump checkpoint, and release it.

    Returns a tuple: (serialized_ckpt_bytes, ckpt_path_used or None)
    """
    pipe = image_decoder_pipeline(
        batch_size=bs,
        num_threads=1,
        device_id=gpu_id,
        rocal_cpu=rocal_cpu,
        tensor_layout=types.NHWC,
        reverse_channels=True,
        mean=[0, 0, 0],
        std=[255, 255, 255],
        device=rocal_device,
        path=img_folder,
        enable_checkpointing=True,
    )
    pipe.build()
    iterator = ROCALGenericIterator(pipe)

    print("Remaining images (initial):", pipe.get_remaining_images())
    for i in range(3):
        batch = iterator.next()
        [image], label = batch
        image_names = get_image_names(pipe)
        for idx in range(bs):
            print(image_names[idx], label[idx])

    # Save checkpoint bytes (and optionally to file)
    serialized_ckpt = pipe.checkpoint(filename=ckpt_path)
    if ckpt_path:
        try:
            size = os.path.getsize(ckpt_path)
            print(f"Checkpoint saved to: {ckpt_path} ({size} bytes)")
        except Exception as e:
            print("Warning: could not stat checkpoint file:", e)
    print("Remaining images at checkpoint:", pipe.get_remaining_images())

    for i in range(5):
        batch = iterator.next()
        [image], label = batch
        image_names = get_image_names(pipe)
        for idx in range(bs):
            print(image_names[idx], label[idx])

    return serialized_ckpt, ckpt_path

def main():
    print('Optional arguments: <cpu/gpu> <image_folder>')
    bs = batch_size
    rocal_device = "cpu"
    rocal_cpu = True
    img_folder = image_dir
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "gpu":
            rocal_device = "gpu"
            rocal_cpu = False
    if len(sys.argv) > 2:
        img_folder = sys.argv[2]

    # Use a unique file for convenience, but we primarily pass bytes to restore
    ckpt_path = os.path.join(
        os.path.dirname(__file__), f"checkpoint.bin"
    )

    print("\n========== Creating and Checkpointing Pipeline ==========")
    serialized_ckpt, used_path = create_and_checkpoint(
        bs, rocal_device, rocal_cpu, img_folder, ckpt_path=ckpt_path
    )

if __name__ == '__main__':
    main()
