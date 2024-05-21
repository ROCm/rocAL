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

bool verify_output(float *dst_ptr, long int frames, std::string case_name) {
    bool pass_status = false;
    // read data from golden outputs
    const char *rocal_data_path = std::getenv("ROCAL_DATA_PATH");
    if (strcmp(rocal_data_path, "") == 0) {
        std::cout << "\n ROCAL_DATA_PATH env variable has not been set. ";
        exit(0);
    }

    std::string ref_file_path = std::string(rocal_data_path) + "rocal_data/GoldenOutputsTensor/reference_outputs_audio/" + case_name + "_output.bin";
    long int out_buffer_size = frames;
    std::vector<float> ref_output(out_buffer_size);
    std::fstream fin(ref_file_path, std::ios::in | std::ios::binary);
    if (fin.is_open()) {
        for (long int i = 0; i < out_buffer_size; i++) {
            if (!fin.eof()) {
                fin.read(reinterpret_cast<char *>(ref_output.data()), sizeof(float));
            } else {
                std::cout << "\nUnable to read all data from golden outputs\n";
                return pass_status;
            }
        }
    } else {
        std::cout << "\nCould not open the reference output. Please check the path specified\n";
        return pass_status;
    }

    int matched_indices = 0;
    for (int j = 0; j < frames; j++) {
        float ref_val, out_val;
        ref_val = ref_output[j];
        out_val = dst_ptr[j];
        bool invalid_comparison = ((out_val == 0.0f) && (ref_val != 0.0f));
        if (!invalid_comparison && abs(out_val - ref_val) < 1e-20)
            matched_indices += 1;
    }

    std::cout << std::endl << "Results for Test case: " << std::endl;
    if ((matched_indices == frames) && matched_indices != 0) {
        pass_status = true;
    }

    return pass_status;
}

int test(int test_case, const char *path, int qa_mode, int downmix, int gpu);
int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT) {
        printf("Usage: ./audio_tests <audio-dataset-folder - required> <test_case> <downmix=0/1> <device-gpu=1/cpu=0> <qa_mode=0/1>\n");
        return -1;
    }

    int argIdx = 0;
    const char *path = argv[++argIdx];
    int qa_mode = 0;
    unsigned test_case = 0;
    bool downmix = false;
    bool gpu = 0;

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        downmix = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        qa_mode = atoi(argv[++argIdx]);

    if (gpu) {  // TODO - Will be removed when GPU support is added for Audio pipeline
        std::cout << "WRN : Currently Audio unit test supports only HOST backend\n";
        gpu = false;
    }

    int return_val = test(test_case, path, qa_mode, downmix, gpu);
    return return_val;
}

int test(int test_case, const char *path, int qa_mode, int downmix, int gpu) {
    int input_batch_size = 1;
    bool is_output_audio_decoder = false;
    std::cout << "test case " << test_case << std::endl;
    std::cout << "Running on " << (gpu ? "GPU" : "CPU") << std::endl;

    auto handle = rocalCreate(input_batch_size,
                              gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                              1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    std::string file_list_path;  // User can modify this with the file list path if required
    if (qa_mode) {                    // setting the default file list path from ROCAL_DATA_PATH
        file_list_path = std::string(std::getenv("ROCAL_DATA_PATH")) + "rocal_data/audio/wav_file_list.txt";
    }

    std::cout << ">>>>>>> Running LABEL READER" << std::endl;
    rocalCreateLabelReader(handle, path, file_list_path.c_str());

    if (test_case == 0)
        is_output_audio_decoder = true;
    RocalTensor decoded_output = rocalAudioFileSourceSingleShard(handle, path, file_list_path.c_str(), 0, 1, is_output_audio_decoder, false, false, downmix);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Audio source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    std::string case_name = "";
    switch (test_case) {
        case 0: {
            case_name = "audio_decoder";
            std::cout << "Running AUDIO DECODER" << std::endl;
        } break;
        case 1: {
            std::cout << "Running PREEMPHASIS" << std::endl;
            case_name = "preemphasis_filter";
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            RocalAudioBorderType preemph_border_type = RocalAudioBorderType::ROCAL_CLAMP;
            RocalFloatParam p_preemph_coeff = rocalCreateFloatParameter(0.97);
            rocalPreEmphasisFilter(handle, decoded_output, true, p_preemph_coeff, preemph_border_type, tensorOutputType);

        } break;
        default: {
            std::cout << "Not a valid test case ! Exiting!\n";
            return -1;
        }
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
    while (rocalGetRemainingImages(handle) >= static_cast<size_t>(input_batch_size)) {
        std::cerr << "\n Iteration:: " << iteration << "\n";
        iteration++;
        if (rocalRun(handle) != 0) {
            break;
        }
        RocalTensorList output_tensor_list = rocalGetOutputTensors(handle);
        int file_name_length[input_batch_size];
        int file_name_size = rocalGetImageNameLen(handle, file_name_length);
        char audio_file_name[file_name_size];
        std::vector<int> roi(4 * input_batch_size, 0);
        rocalGetImageName(handle, audio_file_name);
        RocalTensorList labels = rocalGetImageLabels(handle);
        int *label_id = reinterpret_cast<int *>(labels->at(0)->buffer());  // The labels are present contiguously in memory
        std::cerr << "Audio file : " << audio_file_name << "\n";
        std::cerr << "Label : " << *label_id << "\n";
        for (uint idx = 0; idx < output_tensor_list->size(); idx++) {
            buffer = static_cast<float *>(output_tensor_list->at(idx)->buffer());
            output_tensor_list->at(idx)->copy_roi(roi.data());
            frames = roi[idx * 4 + 2];
        }
    }

    if (qa_mode) {
        std::cout << "\n *****************************Verifying Audio output**********************************\n";
        if (verify_output(buffer, frames, case_name)) {
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
