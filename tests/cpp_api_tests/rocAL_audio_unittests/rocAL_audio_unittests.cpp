/*
MIT License

Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "rocal_api.h"

using namespace std::chrono;

bool verify_output(float *dstPtr, long int frames, const char *ref_path)
{
    std::fstream refFile;
    bool pass_status = false;
    // read data from golden outputs
    long int oBufferSize = frames;
    std::vector<float> refOutput(oBufferSize);
    std::fstream fin(ref_path, std::ios::in | std::ios::binary);
    if(fin.is_open()) {
        for(long int i = 0; i < oBufferSize; i++) {
            if(!fin.eof()) {
                fin.read(reinterpret_cast<char*>(refOutput.data()), sizeof(float));
            } else {
                std::cout<<"\nUnable to read all data from golden outputs\n";
                return pass_status;
            }
        }
    } else {
        std::cout<<"\nCould not open the reference output. Please check the path specified\n";
        return pass_status;
    }

    int matchedIndices = 0;
    for (int j = 0; j < frames; j++) {
        float refVal, outVal;
        refVal = refOutput[j];
        outVal = dstPtr[j];
        bool invalidComparision = ((outVal == 0.0f) && (refVal != 0.0f));
        if (!invalidComparision && abs(outVal - refVal) < 1e-20)
            matchedIndices += 1;
    }

    std::cout << std::endl << "Results for Test case: " << std::endl;
    if ((matchedIndices == frames) && matchedIndices != 0) {
        pass_status = true;
    }

    return pass_status;
}

int test(int test_case, const char *path, const char *ref_path, int downmix, int gpu);
int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    printf("Usage: ./rocal_audio_unittests <audio-dataset-folder> <test_case> <downmix> <device-gpu=1/cpu=0> \n");
    if (argc < MIN_ARG_COUNT)
        return -1;

    int argIdx = 0;
    const char *path = argv[++argIdx];
    const char *ref_path = nullptr;
    unsigned test_case = 0;
    bool downmix = false;
    bool gpu = 0;

    if (argc >= argIdx + MIN_ARG_COUNT)
        ref_path = argv[++argIdx];

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        downmix = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);
    
    if (gpu) {  // TODO - Will be removed when GPU support is added for Audio pipeline
        std::cerr << "WRN : Currently Audio unit test supports only HOST backend\n";
        gpu = false;
    }

    int return_val = test(test_case, path, ref_path, downmix, gpu);
    return return_val;
}

int test(int test_case, const char *path, const char *ref_path, int downmix, int gpu) {
    int inputBatchSize = 1;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << std::endl;

    auto handle = rocalCreate(inputBatchSize,
                              gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                              1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    std::cout << ">>>>>>> Running AUDIO DECODER" << std::endl;
    rocalAudioFileSourceSingleShard(handle, path, 0, 1, true, false, false, downmix);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Audio source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    int iteration = 0;
    float *buffer = nullptr;
    int frames = 0;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    while (rocalGetRemainingImages(handle) >= static_cast<size_t>(inputBatchSize)) {
        std::cout << "\n Iteration:: " << iteration<<"\n";
        iteration++;
        if (rocalRun(handle) != 0) {
            break;
        }
        RocalTensorList output_tensor_list = rocalGetOutputTensors(handle);
        int image_name_length[inputBatchSize];
        int img_size = rocalGetImageNameLen(handle, image_name_length);
        char img_name[img_size];
        std::vector<int> roi(4 * inputBatchSize, 0);
        rocalGetImageName(handle, img_name);
        for (uint idx = 0; idx < output_tensor_list->size(); idx++) {
            buffer = static_cast<float*>(output_tensor_list->at(idx)->buffer());
            output_tensor_list->at(idx)->copy_roi(roi.data());
            frames = roi[idx * 4 + 2];
        }
    }

    if (ref_path) {
        std::cout << "\n *****************************Verifying Audio output**********************************\n";
        if (verify_output(buffer, frames, ref_path)) {
            std::cout << "PASSED!\n\n";
        } else {
            std::cout << "FAILED!\n\n";
        }
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalRelease(handle);
    return 0;
}
