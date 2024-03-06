/*
MIT License

Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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

#include <iostream>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <unistd.h>
#include <vector>

#include "rocal_api.h"

#include "opencv2/opencv.hpp"
using namespace cv;

#if USE_OPENCV_4
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define CV_GRAY2RGB COLOR_GRAY2RGB
#define CV_RGB2BGR COLOR_RGB2BGR
#define CV_FONT_HERSHEY_SIMPLEX FONT_HERSHEY_SIMPLEX
#define CV_FILLED FILLED
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
#endif

#define DISPLAY 1
#define METADATA 0 // Switch the meta-data part once the meta-data reader (file list reader) is introduced
using namespace std::chrono;

int test(int test_case, const char *path, float sample_rate, int downmix, unsigned max_frames, unsigned max_channels, int gpu);
int main(int argc, const char **argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    printf("Usage: image_augmentation <audio-dataset-folder> <test_case> <sample-rate> <downmix> <max_frames> <max_channels> gpu=1/cpu=0 \n");
    if (argc < MIN_ARG_COUNT)
        return -1;

    int argIdx = 0;
    const char *path = argv[++argIdx];
    unsigned test_case = 0;
    float sample_rate = 0.0;
    bool downmix = false;
    unsigned max_frames = 1;
    unsigned max_channels = 1;
    bool gpu = 0;

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        sample_rate = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        downmix = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        max_frames = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        max_channels = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    int return_val = test(test_case, path, sample_rate, downmix, max_frames, max_channels, gpu);
    return return_val;
}

int test(int test_case, const char *path, float sample_rate, int downmix, unsigned max_frames, unsigned max_channels, int gpu)
{
    int inputBatchSize = 10;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << std::endl;

    auto handle = rocalCreate(inputBatchSize,
                              gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                              1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    RocalMetaData metadata_output;
    // MetaData reader for input file_list which has file seperated by labels
    // if (METADATA) { // To uncomment later when meta-data reader for audio is added (PR4)
    //     std::cout << "META DATA READER";
    //     const char* file_list_path = "/workspace/rnnt/AMD/MIVisionX-data/rocal_data/audio_samples/audio_file_list.txt" ; // TODO: Add this as an arg in main() 
    //     metadata_output = rocalCreateFileListLabelReader(handle, path, file_list_path);
    // }

    //Decoder
    // RocalTensor input1; // Uncomment when augmentations are enabled
    // RocalTensorList non_silent_region_op; // Uncomment when NSR is introduced (PR5)
    // const char* file_list_path = "/media/MIVisionX-data/rocal_data/audio_samples/audio_file_list.txt" ; // Uncomment and use it when meta-data reader is introduced (PR4)
    rocalAudioFileSourceSingleShard(handle, path, 0, 1, true, false, false, false, max_frames, max_channels, 0);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Audio source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }
    /* The augmentation cases - To uncomment as each augmentation is introduced

    switch (test_case)
    {
        case 0:
        {
            RocalTensorLayout tensorLayout; // = RocalTensorLayout::None;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalToDecibels(handle, input1, tensorLayout, tensorOutputType, true);
            std::cout<<"\n Calls rocalToDecibels";
        }
        break;
        case 1:
        {
            RocalTensorLayout tensorLayout; // = RocalTensorLayout::None;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalPreEmphasisFilter(handle, input1, tensorOutputType, true);
            std::cout<<"\n Calls rocalPreEmphasisFilter ";
        }
        break;
        case 2:
        {
            RocalTensorLayout tensorLayout; // = RocalTensorLayout::None;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            // int nfftSize = 2048;
            std::vector<float> window_fn{};
            output = rocalSpectrogram(handle, input1, tensorOutputType, true, window_fn, true, true, RocalSpectrogramLayout(0), 2, 512, 512, 256);
            std::cout<<"\n Calls rocalSpectrogram ";
        }
        break;
        case 3:
        {
            auto non_silent_region = rocalNonSilentRegion(handle, input1, true, -60, 1, -1, 3);
            // RocalTensor begin = non_silent_region->at(0);
        }
        break;
        case 4:
        {
            std::cout<<"\n Mel Filter Bank";
            RocalTensorLayout tensorLayout; // = RocalTensorLayout::None;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            std::vector<float> window_fn{};
            RocalTensor temp_output = rocalSpectrogram(handle, input1, tensorOutputType, false, window_fn, true, true, RocalSpectrogramLayout(0), 2, 512, 512, 256);
            float sampleRate = 16000;
            float minFreq = 0.0;
            float maxFreq = sampleRate / 2;
            RocalMelScaleFormula melFormula = RocalMelScaleFormula::SLANEY;
            int numFilter = 128;
            bool normalize = true;

            output = rocalMelFilterBank(handle, temp_output, true, maxFreq, minFreq, melFormula, numFilter, normalize, sampleRate);
        }
        break;
        case 5:
        {
            RocalTensorLayout tensorLayout; // = RocalTensorLayout::None;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            const size_t num_values = 3;
            std::pair <RocalTensor,RocalTensor>  non_silent_region_output;
            non_silent_region_output = rocalNonSilentRegion(handle, input1, false, -60, 0.0, -1, 3);
            output = rocalSlice(handle, input1, tensorOutputType, true, non_silent_region_output.first, non_silent_region_output.second, {0.3f});
        }
        break;
        case 6:
        {
            std::cout<<"\n Normalize";
            RocalTensorLayout tensorLayout;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalNormalize(handle, input1, tensorOutputType, true, false, {1});
        }
        break;
        case 7:
        {
            std::cout<<"\nPad";
            RocalTensorLayout tensorLayout;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalPad(handle, input1, tensorOutputType, true, 4.0f);
        }
        break;

        default:
        {
            std::cout << "Not a valid pipeline type ! Exiting!\n";
            return -1;
        }

    }
    */
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    cv::Mat mat_output, mat_input, mat_color;
    int iteration = 0;
    RocalTensorList output_tensor_list;

    while (rocalGetRemainingImages(handle) >= static_cast<size_t>(inputBatchSize))
    {
        std::cout<<"\n rocalGetRemainingImages:: "<<rocalGetRemainingImages(handle)<<"\t inputBatchsize:: "<<inputBatchSize  ;
        std::cout<<"\n iteration:: "<<iteration;
        iteration++;
        if (rocalRun(handle) != 0) {
            break;
        }
        std::vector<float> audio_op;
        output_tensor_list = rocalGetOutputTensors(handle);
        std::cout << "\n *****************************Audio output**********************************\n";
        std::cout << "\n **************Printing the first 5 values of the Audio buffer**************\n";
        for(uint idx = 0; idx < output_tensor_list->size(); idx++) {
            float * buffer = (float *)output_tensor_list->at(idx)->buffer();
            for(int n = 0; n < 5; n++)
                std::cout << buffer[n] << "\n";
        }

        if (METADATA) {
            RocalTensorList labels = rocalGetImageLabels(handle);
            for(uint i = 0; i < labels->size(); i++) {
                int * labels_buffer = (int *)(labels->at(i)->buffer());
                std::cout << ">>>>> LABELS : " << labels_buffer[0] << "\t";
            }

        }
        std::cout<<"******************************************************************************\n";
    }
    rocalRelease(handle);
    return 0;
}
