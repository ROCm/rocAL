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

test_case_augmentation_map = {
    0: "audio_decoder",
}


def plot_audio_wav(audio_tensor, idx):
    # audio is expected as a tensor
    audio_data = audio_tensor.detach().numpy()
    audio_data = audio_data.flatten()
    plt.plot(audio_data)
    plt.savefig("OUTPUT_FOLDER/AUDIO_READER/" + str(idx) + ".png")
    plt.close()

def verify_output(audio_tensor, rocal_data_path, roi_tensor, test_results, case_name):
    ref_path = f'{rocal_data_path}/GoldenOutputsTensor/reference_outputs_audio/{case_name}_output.bin'
    data_array = np.fromfile(ref_path, dtype=np.float32)
    audio_data = audio_tensor.detach().numpy()
    audio_data = audio_data.flatten()
    roi_data = roi_tensor.detach().numpy()
    matched_indices = 0
    for j in range(roi_data[0]):
        ref_val = data_array[j]
        out_val = audio_data[j]
        # ensuring that out_val is not exactly zero while ref_val is non-zero.
        invalid_comparison = (out_val == 0.0) and (ref_val != 0.0)
        #comparing the absolute difference between the output value (out_val) and the reference value (ref_val) with a tolerance threshold of 1e-20.
        if not invalid_comparison and np.abs(out_val - ref_val) < 1e-20:
            matched_indices += 1

    # Print results
    print(f"Results for {case_name}:")
    if matched_indices == roi_data[0] and matched_indices != 0:
        print("PASSED!")
        test_results[case_name] = "PASSED"
    else:
        print("FAILED!")
        test_results[case_name] = "FAILED"


def main():
    args = parse_args()

    audio_path = args.audio_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    test_case = args.test_case
    qa_mode = args.qa_mode
    num_threads = 1
    device_id = 0
    case_name = test_case_augmentation_map.get(test_case)
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    rocal_data_path = os.environ.get("ROCAL_DATA_PATH")

    if args.display:
        try:
            path = "OUTPUT_FOLDER/AUDIO_READER"
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
        except OSError as error:
            print(error)

    if rocal_data_path is None:
        print("Need to export ROCAL_DATA_PATH")
        sys.exit()
    if not rocal_cpu:
        print("The GPU support for Audio is not given yet. running on cpu")
        rocal_cpu = True
    if audio_path == "":
        audio_path = f'{rocal_data_path}/audio/wav/'
    else:
        print("QA mode is disabled for custom audio data")
        qa_mode = 0
    if qa_mode and batch_size != 1:
        print("QA mode is enabled. Batch size is set to 1.")
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
        audio, labels = fn.readers.file(file_root=audio_path)
        audio_decode = fn.decoders.audio(
            audio,
            file_root=audio_path,
            downmix=False,
            shard_id=0,
            num_shards=1,
            stick_to_shard=False
        )
        audio_pipeline.set_outputs(audio_decode)
    audio_pipeline.build()
    audioIteratorPipeline = ROCALAudioIterator(audio_pipeline, auto_reset=True)
    cnt = 0
    test_results = {}
    import timeit
    start = timeit.default_timer()
    # Enumerate over the Dataloader
    for e in range(int(args.num_epochs)):
        print("Epoch :: ", e)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i, it in enumerate(audioIteratorPipeline):
            for x in range(len(it[0])):
                for audio_tensor, label, roi in zip(it[0][x], it[1], it[2]):
                    if args.print_tensor:
                        print("label", label)
                        print("Audio", audio_tensor)
                        print("Roi", roi)
                    if args.display:
                        plot_audio_wav(audio_tensor, cnt)
                    cnt+=1
        if qa_mode :
            verify_output(audio_tensor, rocal_data_path, roi, test_results, case_name)
            passed_cases = []
            failed_cases = []

            for augmentation_name, result in test_results.items():
                if result == "PASSED":
                    passed_cases.append(augmentation_name)
                else:
                    failed_cases.append(augmentation_name)

            print("Number of PASSED tests:", len(passed_cases))
            print(passed_cases)
            print("Number of FAILED tests:", len(failed_cases))
            print(failed_cases)

        print("EPOCH DONE", e)


if __name__ == "__main__":
    main()
