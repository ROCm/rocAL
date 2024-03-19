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

#define DISPLAY 1
#define METADATA 0  // Switch the meta-data part once the meta-data reader (file list reader) is introduced
using namespace std::chrono;

void verify_output(float *dstPtr, long int frames, int inputBatchSize, int iteration)
{
    std::fstream refFile;
    int fileMatch = 0;

    // read data from golden outputs
    long int oBufferSize = inputBatchSize * frames;
    float *refOutput = static_cast<float *>(malloc(oBufferSize * sizeof(float)));
    std::string outFile = "/media/audio_reference_outputs/ref_output_" + std::to_string(iteration)+".bin";
    std::fstream fin(outFile, std::ios::in | std::ios::binary);
    if(fin.is_open())
    {
        for(long int i = 0; i < oBufferSize; i++)
        {
            if(!fin.eof())
                fin.read(reinterpret_cast<char*>(&refOutput[i]), sizeof(float));
            else
            {
                std::cout<<"\nUnable to read all data from golden outputs\n";
                return;
            }
        }
    }
    else
    {
        std::cout<<"\nCould not open the reference output. Please check the path specified\n";
        return;
    }

    // iterate over all samples in a batch and compare with reference outputs
    for (int batchCount = 0; batchCount < inputBatchSize; batchCount++)
    {
        float *dstPtrCurrent = dstPtr + batchCount * frames;
        float *refPtrCurrent = refOutput + batchCount * frames;
        float *dstPtrRow = dstPtrCurrent;
        float *refPtrRow = refPtrCurrent;
        int hStride = frames;

        int matchedIndices = 0;
        float *dstPtrTemp = dstPtrRow;
        std::cerr<<"\n "<<iteration<<inputBatchSize<<frames;
        float *refPtrTemp = refPtrRow ;
        for (int j = 0; j < frames; j++)
        {
            float refVal, outVal;
            refVal = refPtrTemp[j];
            outVal = dstPtrTemp[j];
            bool invalidComparision = ((outVal == 0.0f) && (refVal != 0.0f));
            if (!invalidComparision && abs(outVal - refVal) < 1e-20)
                matchedIndices += 1;
            else
            {
                std::cerr<<"\n mismatches : "<< j <<" "<<outVal<<" "<<refVal;
            }
        }
        if (matchedIndices == (frames) && matchedIndices !=0)
            fileMatch++;
    }

    std::cout << std::endl << "Results for Test case: " << std::endl;
    if (fileMatch == inputBatchSize)
    {
        std::cout << "PASSED!" << std::endl;
    }
    else
    {
        std::cout << "FAILED! " << fileMatch << "/" << inputBatchSize << " outputs are matching with reference outputs" << std::endl;
    }

    free(refOutput);
}

int test(int test_case, const char *path, float sample_rate, int downmix, unsigned max_frames, unsigned max_channels, int gpu);
int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    printf("Usage: audio_augmentation <audio-dataset-folder> <test_case> <sample-rate> <downmix> <max_frames> <max_channels> gpu=1/cpu=0 \n");
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

int test(int test_case, const char *path, float sample_rate, int downmix, unsigned max_frames, unsigned max_channels, int gpu) {
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

    rocalAudioFileSourceSingleShard(handle, path, 0, 1, true, false, false, false, max_frames, max_channels);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Audio source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Diplay Values<<<<<<<<<<<<<<<<<*/
    int iteration = 0;
    RocalTensorList output_tensor_list;

    while (rocalGetRemainingImages(handle) >= static_cast<size_t>(inputBatchSize)) {
        std::cout << "\n rocalGetRemainingAudios:: " << rocalGetRemainingImages(handle) << "\t inputBatchsize:: " << inputBatchSize;
        std::cout << "\n iteration:: " << iteration;
        iteration++;
        if (rocalRun(handle) != 0) {
            break;
        }
        output_tensor_list = rocalGetOutputTensors(handle);
        int image_name_length[inputBatchSize];
        int img_size = rocalGetImageNameLen(handle, image_name_length);
        char img_name[img_size];
        void* roi_buf = malloc(4 * sizeof(int)); // Allocate memory for four floats
        rocalGetImageName(handle, img_name);
        std::cerr << "\nPrinting image names of batch: " << img_name;
        std::cout << "\n *****************************Audio output**********************************\n";
        std::cout << "\n **************Printing the first 5 values of the Audio buffer**************\n";
        for (uint idx = 0; idx < output_tensor_list->size(); idx++) {
            float *buffer = (float *)output_tensor_list->at(idx)->buffer();
            output_tensor_list->at(idx)->copy_roi(roi_buf);
            int* int_buf = static_cast<int*>(roi_buf);
            long int frames = int_buf[2];
            int channels = output_tensor_list->at(idx)->dims().at(2);
            if(iteration == 10)
                verify_output(buffer, frames, inputBatchSize, iteration - 1);
            for (int n = 0; n < 5; n++)
                std::cout << buffer[n] << "\n";
        }

        if (METADATA) {
            RocalTensorList labels = rocalGetImageLabels(handle);
            for (uint i = 0; i < labels->size(); i++) {
                int *labels_buffer = (int *)(labels->at(i)->buffer());
                std::cout << ">>>>> LABELS : " << labels_buffer[0] << "\t";
            }
        }
        std::cout << "******************************************************************************\n";
    }
    rocalRelease(handle);
    return 0;
}
