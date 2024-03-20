# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
from amd.rocal.plugin.pytorch import ROCALAudioIterator
import amd.rocal.fn as fn
import random
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np

np.set_printoptions(threshold=1000, edgeitems=10000)

def plot_audio_wav(audio_tensor, idx, device):
    # audio is expected as a tensor
    audio_data = audio_tensor.detach().numpy()
    audio_data = audio_data.flatten()
    # Saving the array in a text file
    file = open("results/rocal_data_new" + str(idx) + ".txt", "w+")
    content = str(audio_data)
    file.write(content)
    file.close()
    plt.plot(audio_data)
    plt.savefig("results/rocal_data_new" + str(idx) + ".png")
    plt.close()

def main():
    if len(sys.argv) < 3:
        print("Please pass audio_folder file_list cpu/gpu batch_size")
        exit(0)
    try:
        path = "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "audio"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    file_list = ""
    rocal_cpu = True  # The GPU support for Audio is not given yet
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    print("*********************************************************************")
    audio_pipeline = Pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=random_seed,
        rocal_cpu=rocal_cpu,
    )
    with audio_pipeline:
        audio_decode = fn.decoders.audio(
            file_root=data_path,
            file_list_path=file_list,
            downmix=False,
            shard_id=0,
            num_shards=2,
            stick_to_shard=False,
        )
        audio_pipeline.set_outputs(audio_decode)
    audio_pipeline.build()
    audioIteratorPipeline = ROCALAudioIterator(audio_pipeline, auto_reset=True)
    cnt = 0
    for e in range(1):
        print("Epoch :: ", e)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i, it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************", i)
            for x in range(len(it[0])):
                for audio_tensor, label in zip(it[0][x], it[1]):
                    print("label", label)
                    print("cnt", cnt)
                    print("Audio", audio_tensor)
                    plot_audio_wav(audio_tensor, cnt, "cpu")
                    cnt += 1
        print("EPOCH DONE", e)


if __name__ == "__main__":
    main()
