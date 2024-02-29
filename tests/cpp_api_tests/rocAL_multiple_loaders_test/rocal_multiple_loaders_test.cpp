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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "rocal_api.h"
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

#define DISPLAY 0

using namespace std::chrono;

int test(const char *path, const char *outName, int rgb, int gpu, int width, int height, int display_all);
int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT) {
        printf("Usage: rocal_multiple_loaders_test <image-dataset-folder> output_image_name <width> <height> gpu=1/cpu=0 rgb=1/grayscale=0 display_all=0(display_last_only)1(display_all)\n");
        return -1;
    }

    int argIdx = 0;
    const char *path = argv[++argIdx];
    const char *outName = argv[++argIdx];
    int width = atoi(argv[++argIdx]);
    int height = atoi(argv[++argIdx]);
    int display_all = 0;
    int rgb = 1;  // process color images
    bool gpu = 1;

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        display_all = atoi(argv[++argIdx]);

    test(path, outName, rgb, gpu, width, height, display_all);

    return 0;
}

int test(const char *path, const char *outName, int rgb, int gpu, int width, int height, int display_all) {
    size_t num_threads = 1;
    unsigned int inputBatchSize = 2;
    int decode_max_width = width;
    int decode_max_height = height;

    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << " , " << (rgb ? " Color " : " Grayscale ") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24
                                              : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(inputBatchSize,
                              gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                              1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    rocalSetSeed(0);
    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/

    RocalTensor decoded_output, decoded_output1;
    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size

    std::cout << ">>>>>>> Running IMAGE READERS" << std::endl;
    rocalCreateLabelReader(handle, path);
    if (decode_max_height <= 0 || decode_max_width <= 0) {
        decoded_output = rocalJpegFileSource(handle, path, color_format, num_threads, false, true);
        decoded_output1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, true);
    } else {
        decoded_output = rocalJpegFileSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        decoded_output1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
    }

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    RocalTensor output;

    std::cout << ">>>>>>> Running rocalBrightness" << std::endl;
    output = rocalBrightness(handle, decoded_output, true);
    std::cout << ">>>>>>> Running rocalGamma" << std::endl;
    output = rocalGamma(handle, decoded_output1, true);

    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    std::vector<cv::Mat> mat_output, mat_input;
    cv::Mat mat_color;
    if (DISPLAY)
        cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    printf("Going to process images\n");
    printf("Remaining images %lu \n", rocalGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;
    bool first_run = true;
    while (rocalGetRemainingImages(handle) >= inputBatchSize) {
        index++;
        if (rocalRun(handle) != 0)
            break;

        RocalTensorList output_tensor_list = rocalGetOutputTensors(handle);

        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        for (unsigned idx = 0; idx < output_tensor_list->size(); idx++) {
            auto output_tensor = output_tensor_list->at(idx);
            int h = output_tensor->shape().at(1) * output_tensor->dims().at(0);
            int w = output_tensor->shape().at(0);
            if (first_run) {
                mat_input.emplace_back(cv::Mat(h, w, cv_color_format));
                mat_output.emplace_back(cv::Mat(h, w, cv_color_format));
            }

            output_tensor->copy_data(mat_input[idx].data, ROCAL_MEMCPY_HOST);
            mat_input[idx].copyTo(mat_output[idx](cv::Rect(0, 0, w, h)));

            std::string out_filename = std::string(outName) + ".png";  // in case the user specifies non png filename
            if (display_all)
                out_filename = std::string(outName) + std::to_string(index) + std::to_string(idx) + ".png";  // in case the user specifies non png filename

            if (color_format == RocalImageColor::ROCAL_COLOR_RGB24) {
                cv::cvtColor(mat_output[idx], mat_color, CV_RGB2BGR);
                if (DISPLAY)
                    cv::imshow("output", mat_output[idx]);
                else
                    cv::imwrite(out_filename, mat_color, compression_params);
            } else {
                if (DISPLAY)
                    cv::imshow("output", mat_output[idx]);
                else
                    cv::imwrite(out_filename, mat_output[idx], compression_params);
            }
        }
        first_run = false;
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    for (unsigned i = 0; i < mat_input.size(); i++) {
        mat_input[i].release();
        mat_output[i].release();
    }
    rocalRelease(handle);
    if (!output)
        return -1;
    return 0;
}