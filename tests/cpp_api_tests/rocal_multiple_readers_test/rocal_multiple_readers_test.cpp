/*
MIT License

Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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
// #define RANDOMBBOXCROP

using namespace std::chrono;

std::string get_interpolation_type(unsigned int val, RocalResizeInterpolationType &interpolation_type) {
    switch (val) {
        case 0: {
            interpolation_type = ROCAL_NEAREST_NEIGHBOR_INTERPOLATION;
            return "NearestNeighbor";
        }
        case 2: {
            interpolation_type = ROCAL_CUBIC_INTERPOLATION;
            return "Bicubic";
        }
        case 3: {
            interpolation_type = ROCAL_LANCZOS_INTERPOLATION;
            return "Lanczos";
        }
        case 4: {
            interpolation_type = ROCAL_GAUSSIAN_INTERPOLATION;
            return "Gaussian";
        }
        case 5: {
            interpolation_type = ROCAL_TRIANGULAR_INTERPOLATION;
            return "Triangular";
        }
        default: {
            interpolation_type = ROCAL_LINEAR_INTERPOLATION;
            return "Bilinear";
        }
    }
}

std::string get_scaling_mode(unsigned int val, RocalResizeScalingMode &scale_mode) {
    switch (val) {
        case 1: {
            scale_mode = ROCAL_SCALING_MODE_STRETCH;
            return "Stretch";
        }
        case 2: {
            scale_mode = ROCAL_SCALING_MODE_NOT_SMALLER;
            return "NotSmaller";
        }
        case 3: {
            scale_mode = ROCAL_SCALING_MODE_NOT_LARGER;
            return "Notlarger";
        }
        default: {
            scale_mode = ROCAL_SCALING_MODE_DEFAULT;
            return "Default";
        }
    }
}

int test(int test_case, int reader_type, const char *path, const char *outName, int rgb, int gpu, int width, int height, int num_of_classes, int display_all, int resize_interpolation_type, int resize_scaling_mode);
int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT) {
        printf("Usage: rocal_unittests reader-type <image-dataset-folder> output_image_name <width> <height> test_case gpu=1/cpu=0 rgb=1/grayscale=0 one_hot_labels=num_of_classes/0  display_all=0(display_last_only)1(display_all)\n");
        return -1;
    }

    int argIdx = 0;
    int reader_type = atoi(argv[++argIdx]);
    const char *path = argv[++argIdx];
    const char *outName = argv[++argIdx];
    int width = atoi(argv[++argIdx]);
    int height = atoi(argv[++argIdx]);
    int display_all = 0;

    int rgb = 1;  // process color images
    bool gpu = 1;
    int test_case = 3;  // For Rotate
    int num_of_classes = 0;
    int resize_interpolation_type = 1;  // For Bilinear interpolations
    int resize_scaling_mode = 0;        // For Default scaling mode

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        num_of_classes = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        display_all = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        resize_interpolation_type = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        resize_scaling_mode = atoi(argv[++argIdx]);

    test(test_case, reader_type, path, outName, rgb, gpu, width, height, num_of_classes, display_all, resize_interpolation_type, resize_scaling_mode);

    return 0;
}

int test(int test_case, int reader_type, const char *path, const char *outName, int rgb, int gpu, int width, int height, int num_of_classes, int display_all, int resize_interpolation_type, int resize_scaling_mode) {
    size_t num_threads = 1;
    unsigned int inputBatchSize = 2;
    int decode_max_width = width;
    int decode_max_height = height;
    int pipeline_type = -1;
    std::cout << ">>> test case " << test_case << std::endl;
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

    /*>>>>>>>>>>>>>>>> Getting the path for MIVisionX-data  <<<<<<<<<<<<<<<<*/

    std::string rocal_data_path;
    if (std::getenv("ROCAL_DATA_PATH"))
        rocal_data_path = std::getenv("ROCAL_DATA_PATH");

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalSetSeed(0);

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalIntParam color_temp_adj = rocalCreateIntParameter(-50);
    RocalIntParam mirror = rocalCreateIntParameter(1);

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/

#if defined RANDOMBBOXCROP
    bool all_boxes_overlap = true;
    bool no_crop = false;
#endif

    RocalTensor decoded_output, decoded_output1;
    RocalTensorLayout output_tensor_layout = (rgb != 0) ? RocalTensorLayout::ROCAL_NHWC : RocalTensorLayout::ROCAL_NCHW;
    RocalTensorOutputType output_tensor_dtype = RocalTensorOutputType::ROCAL_UINT8;
    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    switch (reader_type) {
        default: {
            std::cout << ">>>>>>> Running IMAGE READERS" << std::endl;
            pipeline_type = 1;
            rocalCreateLabelReader(handle, path);
            if (decode_max_height <= 0 || decode_max_width <= 0) {
                decoded_output = rocalJpegFileSource(handle, path, color_format, num_threads, false, true);
                decoded_output1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, true);
            } else {
                decoded_output = rocalJpegFileSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
                decoded_output1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
            }
        } break;
    }

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = width, resize_h = height;  // height and width

    RocalTensor input = decoded_output;
    RocalTensor input_1 = decoded_output1;
    // RocalTensor input = rocalResize(handle, decoded_output, resize_w, resize_h, false); // uncomment when processing images of different size
    RocalTensor output;

    if ((test_case == 48 || test_case == 49 || test_case == 50) && rgb == 0) {
        std::cout << "Not a valid option! Exiting!\n";
        return -1;
    }
    switch (test_case) {
        case 1: {
            std::cout << ">>>>>>> Running "
                      << "rocalBrightness" << std::endl;
            output = rocalBrightness(handle, input, true);
            output = rocalGamma(handle, input_1, true);
        } break;
        default:
            std::cout << "Not a valid option! Exiting!\n";
            return -1;
    }

    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    auto number_of_outputs = rocalGetAugmentationBranchCount(handle);
    std::cout << "\n\nAugmented copies count " << number_of_outputs << "\n";

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle);
    int w = rocalGetOutputWidth(handle);
    int p = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);
    const unsigned number_of_cols = 1; //1920 / w;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    std::vector<cv::Mat> mat_output, mat_input;
    cv::Mat mat_color;
    int col_counter = 0;
    if(DISPLAY)
        cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    printf("Going to process images\n");
    printf("Remaining images %lu \n", rocalGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;
    bool first_run = true;
    while (rocalGetRemainingImages(handle) >= inputBatchSize)
    {
        index++;
        if (rocalRun(handle) != 0)
            break;
        int numOfClasses = 0;
        int image_name_length[inputBatchSize];
        auto last_colot_temp = rocalGetIntValue(color_temp_adj);
        rocalUpdateIntParameter(last_colot_temp + 1, color_temp_adj);

        RocalTensorList output_tensor_list = rocalGetOutputTensors(handle);

        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        for(int idx = 0; idx < output_tensor_list->size(); idx++)
        {
            auto output_tensor = output_tensor_list->at(idx);
            int h = output_tensor->shape().at(1) * output_tensor->dims().at(0);
            int w = output_tensor->shape().at(0);
            if(first_run) {
                mat_input.emplace_back(cv::Mat(h, w, cv_color_format));
                mat_output.emplace_back(cv::Mat(h, w, cv_color_format));   
            }
            
            unsigned char *out_buffer = reinterpret_cast<unsigned char *>(malloc(output_tensor->data_size()));
            output_tensor->copy_data(out_buffer, ROCAL_MEMCPY_HOST);
            mat_input[idx].data = reinterpret_cast<unsigned char *>(out_buffer);
            mat_input[idx].copyTo(mat_output[idx](cv::Rect(0, 0, w, h)));

            std::string out_filename = std::string(outName) + ".png";   // in case the user specifies non png filename
            if (display_all)
                out_filename = std::string(outName) + std::to_string(index) + std::to_string(idx) + ".png";   // in case the user specifies non png filename

            if (color_format == RocalImageColor::ROCAL_COLOR_RGB24)
            {
                cv::cvtColor(mat_output[idx], mat_color, CV_RGB2BGR);
                if(DISPLAY)
                    cv::imshow("output", mat_output[idx]);
                else
                    cv::imwrite(out_filename, mat_color, compression_params);
            }
            else
            {
                if(DISPLAY)
                cv::imshow("output", mat_output[idx]);
                else
                cv::imwrite(out_filename, mat_output[idx], compression_params);
            }
            // col_counter = (col_counter + 1) % number_of_cols;
            if(out_buffer != nullptr) free(out_buffer);
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
    for (int i = 0; i < mat_input.size(); i++) {
        mat_input[i].release();
        mat_output[i].release();
    }
    rocalRelease(handle);
    if (!output)
        return -1;
    return 0;
}