/*
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <memory>

#include "rocal_api.h"
#include "opencv2/opencv.hpp"
using namespace cv;

int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if (argc < MIN_ARG_COUNT) {
        std::cout << "Usage: serialization_test <image_dataset_folder - required> <processing_device:gpu=1/cpu=0>\n";
        return -1;
    }

    int argIdx = 1;
    const char *folderPath = argv[argIdx++];
    int processing_device = 0;

    if (argc > argIdx)
        processing_device = atoi(argv[argIdx++]);

    const int inputBatchSize = 2;
    RocalImageColor color_format = RocalImageColor::ROCAL_COLOR_RGB24;
    std::cout << ">>> Running serialization test on " << (processing_device ? "GPU" : "CPU") << std::endl;

    // Create rocAL context
    auto handle = rocalCreate(inputBatchSize, 
                             processing_device ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 
                             0, 1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal context\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    // Create JPEG file source
    RocalTensor decoded_output = rocalJpegFileSource(handle, folderPath, color_format, 1, false, false);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }
    // Create label reader
    rocalCreateLabelReader(handle, folderPath);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Label reader could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }
    // Add brightness augmentation (mark as output)
    RocalTensor brightness_output = rocalBrightness(handle, decoded_output, true);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Brightness augmentation could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // Verify and build the augmentation graph
    if (rocalVerify(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph" << std::endl;
        return -1;
    }
    std::cout << "Pipeline created successfully!" << std::endl;
    std::cout << "Augmented copies count: " << rocalGetAugmentationBranchCount(handle) << std::endl;
    std::cout << "Output dimensions: " << rocalGetOutputWidth(handle) << "x" << rocalGetOutputHeight(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Serialization Test <<<<<<<<<<<<<<<<<<<*/
    std::cout << "\n=== Testing Pipeline Serialization ===" << std::endl;
    
    // Get the size of the serialized string
    size_t serialized_string_size = 0;
    RocalStatus serialize_status = rocalSerialize(handle, &serialized_string_size);
    if (serialize_status != ROCAL_OK) {
        std::cout << "Failed to serialize pipeline: " << rocalGetErrorMessage(handle) << std::endl;
        rocalRelease(handle);
        return -1;
    }
    std::cout << "Serialized string size: " << serialized_string_size << " bytes" << std::endl;

    // Allocate buffer for the serialized string
    std::string serialized_pipe_string(serialized_string_size, '\0');
    // Get the actual serialized string
    RocalStatus get_string_status = rocalGetSerializedString(handle, serialized_pipe_string.data());
    if (get_string_status != ROCAL_OK) {
        std::cout << "Failed to get serialized string: " << rocalGetErrorMessage(handle) << std::endl;
        rocalRelease(handle);
        return -1;
    }
    std::cout << "\n=== Serialized Pipeline String ===" << std::endl;
    std::cout << serialized_pipe_string << std::endl;
    std::cout << "=== End of Serialized String ===" << std::endl;
    
    /*>>>>>>>>>>>>>>>>>>> Test Pipeline Execution <<<<<<<<<<<<<<<<<<<*/
    std::cout << "\n=== Testing Pipeline Execution ===" << std::endl;
    std::cout << "Available images: " << rocalGetRemainingImages(handle) << std::endl;
    
    // Run a few iterations to test the pipeline
    const int test_iterations = 2;
    int ImageNameLen[inputBatchSize];
    std::vector<std::string> names;
    names.resize(inputBatchSize);
    /*>>>>>>>>>>>>>>>>>>> Display using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle) * inputBatchSize;
    int w = rocalGetOutputWidth(handle);
    int p = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;

    // Variables for OpenCV display
    int col_counter = 0;
    int number_of_cols = 1;
    bool display_all = true;
    const char* outName = "serialization_test_output";

    for (int iter = 0; iter < test_iterations && !rocalIsEmpty(handle); iter++) {
        std::cout << "\nIteration " << (iter + 1) << ":" << std::endl;
        
        if (rocalRun(handle) != 0) {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            rocalRelease(handle);
            return -1;
        }
        
        // Get labels
        RocalTensorList labels = rocalGetImageLabels(handle);
        // Get image names
        unsigned imagename_size = rocalGetImageNameLen(handle, ImageNameLen);
        std::vector<char> imageNames(imagename_size);
        rocalGetImageName(handle, imageNames.data());
        std::string imageNamesStr(imageNames.data());
        int pos = 0;
        int *labels_buffer = reinterpret_cast<int *>(labels->at(0)->buffer());
        for (int i = 0; i < inputBatchSize; i++) {
            names[i] = imageNamesStr.substr(pos, ImageNameLen[i]);
            pos += ImageNameLen[i];
            std::cout << "  Image: " << names[i] << " | Label: " << labels_buffer[i] << std::endl;
        }

        // Copy Image data from handle
        rocalCopyToOutput(handle, mat_input.data, h * w * p);
        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        std::string out_filename = std::string(outName) + ".png";  // in case the user specifies non png filename
        if (display_all)
            out_filename = std::string(outName) + std::to_string(iter) + ".png";  // in case the user specifies non png filename

        if (color_format == RocalImageColor::ROCAL_COLOR_RGB24) {
            cv::cvtColor(mat_output, mat_color, cv::COLOR_RGB2BGR);
            cv::imwrite(out_filename, mat_color);
        } else {
            cv::imwrite(out_filename, mat_output);
        }
        col_counter = (col_counter + 1) % number_of_cols;
    }

    std::cout << "\n=== Serialization Test Completed Successfully ===" << std::endl;
    rocalRelease(handle);
    return 0;
}
