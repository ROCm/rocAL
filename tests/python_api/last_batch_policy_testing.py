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
from amd.rocal.pipeline import pipeline_def
from amd.rocal.plugin.pytorch import ROCALAudioIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import random
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
from parse_config import parse_args

np.set_printoptions(threshold=1000, edgeitems=10000)
seed = random.SystemRandom().randint(0, 2**32 - 1)

test_case_augmentation_map = {
    0: "last_batch_policy_FILL_last_batch_padded_True",
}

def plot_audio_wav(audio_tensor, idx):
    # audio is expected as a tensor
    audio_data = audio_tensor.detach().numpy()
    audio_data = audio_data.flatten()
    plt.plot(audio_data)
    plt.savefig("OUTPUT_FOLDER/AUDIO_READER/" + str(idx) + ".png")
    plt.close()

def verify_output(output_list, rocal_data_path, roi_tensor, test_results, case_name, dimensions):
    ref_path = f'{rocal_data_path}/GoldenOutputsTensor/reference_outputs_audio/{case_name}_output.bin'
    data_array = np.fromfile(ref_path, dtype=np.float32)
    audio_data = output_list[0].detach().numpy()
    audio_data = audio_data.flatten()
    roi_data = roi_tensor.detach().numpy()
    buffer_size = roi_data[0] * roi_data[1]
    matched_indices = 0
    for i in range(roi_data[0]):
        for j in range(roi_data[1]):
            ref_val = data_array[i * roi_data[1] + j]
            out_val = audio_data[i * dimensions[2] + j]
            # ensuring that out_val is not exactly zero while ref_val is non-zero.
            invalid_comparison = (out_val == 0.0) and (ref_val != 0.0)
            #comparing the absolute difference between the output value (out_val) and the reference value (ref_val) with a tolerance threshold of 1e-20.
            if not invalid_comparison and abs(out_val - ref_val) < 1e-20:
                matched_indices += 1

    # Print results
    print(f"Results for {case_name}:")
    if matched_indices == buffer_size and matched_indices != 0:
        print("PASSED!")
        test_results[case_name] = "PASSED"
    else:
        print("FAILED!")
        test_results[case_name] = "FAILED"

@pipeline_def(seed=seed)
def audio_decoder_pipeline(path, file_list, downmix=False):
    audio, labels = fn.readers.file(file_root=path, file_list=file_list)
    return fn.decoders.audio(
        audio,
        file_root=path,
        file_list_path=file_list,
        last_batch_policy=types.LAST_BATCH_DROP, pad_last_batch_repeated=False,
        downmix=downmix,
        shard_id=2,
        num_shards=3,
        stick_to_shard=True,
        shard_size=10*3) #shard_size for all the gpus should be passed here. - hence * 3


def main():
    args = parse_args()

    audio_path = args.audio_path
    file_list = args.file_list_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    test_case = args.test_case
    qa_mode = args.qa_mode
    num_threads = 1
    device_id = 0
    rocal_data_path = os.environ.get("ROCAL_DATA_PATH")

    case_list = list(test_case_augmentation_map.keys())

    if test_case is not None: 
        if test_case not in case_list:
            print(" Invalid Test Case! ")
            exit()
        else:
            case_list = [test_case]

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
    if audio_path == "" and file_list == "":
        audio_path = f'{rocal_data_path}/rocal_data/audio/'
        file_list = f'{rocal_data_path}/rocal_data/audio/wav_file_list.txt'
        downmix_audio_path = f'{rocal_data_path}/rocal_data/multi_channel_wav/'
    else:
        print("QA mode is disabled for custom audio data")
        qa_mode = 0
    if qa_mode and batch_size != 1:
        print("QA mode is enabled. Batch size is set to 1.")
        batch_size = 1

    print("*********************************************************************")
    test_results = {}
    for case in case_list:
        case_name = test_case_augmentation_map.get(case)
        if case_name == "last_batch_policy_FILL_last_batch_padded_True":
            audio_pipeline = audio_decoder_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, rocal_cpu=rocal_cpu, path=audio_path, file_list=file_list)
            audio_pipeline.build()
        audioIteratorPipeline = ROCALAudioIterator(audio_pipeline, auto_reset=True, size = 10)
        output_tensor_list = audio_pipeline.get_output_tensors()
        cnt = 0
        import timeit
        start = timeit.default_timer()
        # Enumerate over the Dataloader
        for e in range(int(args.num_epochs)):
            print("Epoch :: ", e)
            torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
            for i, output_list in enumerate(audioIteratorPipeline):
                
                for x in range(len(output_list[0])):
                    print("\n output_list[1]", output_list[1])
                    # for audio_tensor, label, roi in zip(output_list[0][x], output_list[1], output_list[2]):
                    #     if args.print_tensor:
                    #         print("label", label)
                    #     if args.display:
                    #         plot_audio_wav(audio_tensor, cnt)
                    #     cnt+=1

            print("EPOCH DONE", e)

        stop = timeit.default_timer()
        print('\nTime: ', stop - start)




if __name__ == "__main__":
    main()


