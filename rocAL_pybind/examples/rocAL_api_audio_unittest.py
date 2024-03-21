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
from parse_config import parse_args

np.set_printoptions(threshold=1000, edgeitems=10000)

def plot_audio_wav(audio_tensor, idx):
    # audio is expected as a tensor
    audio_data = audio_tensor.detach().numpy()
    audio_data = audio_data.flatten()
    label = idx.cpu().detach().numpy()
    # Saving the array in a text file
    file = open("results/rocal_data_new" + str(label) + ".txt", "w+")
    content = str(audio_data)
    file.write(content)
    file.close()
    plt.plot(audio_data)
    plt.savefig("results/rocal_data_new" + str(label) + ".png")
    plt.close()

def verify_output(audio_tensor, roi_tensor, ref_path, test_results):
    data_array = np.fromfile(ref_path, dtype=np.float32)
    audio_data = audio_tensor.detach().numpy()
    audio_data = audio_data.flatten()
    roi_data = roi_tensor.detach().numpy()
    matched_indices = 0
    for j in range(roi_data[0]):
        ref_val = data_array[j]
        out_val = audio_data[j]
        invalid_comparison = (out_val == 0.0) and (ref_val != 0.0)
        if not invalid_comparison and np.abs(out_val - ref_val) < 1e-20:
            matched_indices += 1

    # Print results
    print("Results for Test case:")
    if matched_indices == roi_data[0] and matched_indices != 0:
        print("PASSED!")
        test_results.append("PASSED")
    else:
        print("FAILED!")
        test_results.append("FAILED")


def main():
    if len(sys.argv) < 3:
        print("Please pass audio_folder batch_size")
        exit(0)
    try:
        path = "results"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)

    args = parse_args()

    audio_path = args.audio_path
    file_list = args.file_list_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    test_case = args.test_case
    ref_path = args.ref_path
    qa_mode = args.qa_mode
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    if not rocal_cpu:
        print("The GPU support for Audio is not given yet. running on cpu")
        rocal_cpu = True
    if qa_mode:
        batch_size = 1

    print("*********************************************************************")
    audio_pipeline = Pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=random_seed,
        rocal_cpu=rocal_cpu,
    )
    with audio_pipeline:
        audio, label = fn.readers.file(file_root=audio_path, file_list=file_list)
        audio_decode = fn.decoders.audio(
            file_root=audio_path,
            file_list_path=file_list,
            downmix=False,
            shard_id=0,
            num_shards=1,
            stick_to_shard=False,
        )
        pre_emphasis_filter = fn.preemphasis_filter(audio_decode)
        spec = fn.spectrogram(
            pre_emphasis_filter,
            nfft=512,
            window_length=320,
            window_step=160,
            rocal_tensor_output_type = types.FLOAT)
        audio_pipeline.set_outputs(length)
    audio_pipeline.build()
    audioIteratorPipeline = ROCALAudioIterator(audio_pipeline, auto_reset=True)
    cnt = 0
    out_tensor = None
    out_roi = None
    test_results = []
    import timeit
    start = timeit.default_timer()
    # Enumerate over the Dataloader
    for e in range(int(args.num_epochs)):
        print("Epoch :: ", e)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i, it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************", i)
            for x in range(len(it[0])):
                for audio_tensor, label, roi in zip(it[0][x], it[1], it[2]):
                    if args.print_tensor:
                        print("label", label)
                        print("cnt", cnt)
                        print("Audio", audio_tensor)
                        print("Roi", roi)
                    if args.dump_output:
                        plot_audio_wav(audio_tensor, label)
                    out_tensor = audio_tensor
                    out_roi = roi
                    cnt+=1
        if qa_mode :
            verify_output(out_tensor, out_roi, ref_path, test_results)
            num_passed = test_results.count("PASSED")
            num_failed = test_results.count("FAILED")

            print("Number of PASSED tests:", num_passed)
            print("Number of FAILED tests:", num_failed)

        print("EPOCH DONE", e)


if __name__ == "__main__":
    main()
