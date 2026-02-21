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
  
    /*>>>>>>>>>>>>>>>>>>> Deserialization Test <<<<<<<<<<<<<<<<<<<*/
    std::cout << "\n=== Testing Pipeline Deserialization ===" << std::endl;
    // Set up pipeline parameters for deserialization
    // These parameters define how the deserialized pipeline should be configured
    RocalPipelineParams pipe_params;

    // Deserialize the pipeline from the serialized string
    // This creates a new pipeline context from the previously serialized pipeline configuration
    RocalContext second_handle = rocalDeserialize(serialized_pipe_string.c_str(), 
                                                  serialized_string_size, 
                                                  &pipe_params);
    
    // Check if deserialization was successful
    if (second_handle != nullptr && rocalGetStatus(second_handle) != ROCAL_OK) {
        std::cout << "Failed to deserialize pipeline: " << rocalGetErrorMessage(second_handle) << std::endl;
        rocalRelease(handle);
        rocalRelease(second_handle);
        return -1;
    } else if (second_handle == nullptr) {
        std::cout << "Failed to deserialize pipeline: Received null context" << std::endl;
        rocalRelease(handle);
        return -1;
    }
    // Set seed if provided in pipeline parameters
    if (pipe_params.seed.has_value()) {
        rocalSetSeed(pipe_params.seed.value());
        std::cout << "Seed set to: " << pipe_params.seed.value() << std::endl;
    }
    // Verify and build the deserialized augmentation graph
    // This step validates the pipeline configuration and prepares it for execution
    if (rocalVerify(second_handle) != ROCAL_OK) {
        std::cout << "Could not verify the deserialized augmentation graph: " << rocalGetErrorMessage(second_handle) << std::endl;
        rocalRelease(handle);
        rocalRelease(second_handle);
        return -1;
    }

    // Get and validate the number of output branches in the deserialized pipeline
    auto number_of_output = rocalGetAugmentationBranchCount(second_handle);
    std::cout << "Deserialized pipeline - Augmented copies count: " << number_of_output << std::endl;
    std::cout << "Deserialized pipeline - Output dimensions: " << rocalGetOutputWidth(second_handle) << "x" << rocalGetOutputHeight(second_handle) << std::endl;

    // Validate that the deserialized pipeline has the expected number of outputs
    if (number_of_output != 1) {
        std::cout << "Error: Deserialized pipeline has " << number_of_output << " outputs, expected 1" << std::endl;
        rocalRelease(handle);
        rocalRelease(second_handle);
        return -1;
    }
    
    std::cout << "Pipeline deserialized and verified successfully!" << std::endl;
    /*>>>>>>>>>>>>>>>>>>> Test Original Pipeline Execution <<<<<<<<<<<<<<<<<<<*/
    std::cout << "\n=== Testing Original Pipeline Execution ===" << std::endl;
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
        std::cout << "\nOriginal Pipeline - Iteration " << (iter + 1) << ":" << std::endl;
        
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
            std::cout << "  Original - Image: " << names[i] << " | Label: " << labels_buffer[i] << std::endl;
        }

        // Copy Image data from handle
        rocalCopyToOutput(handle, mat_input.data, h * w * p);
        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        std::string out_filename = std::string("original_") + std::string(outName);
        if (display_all)
            out_filename += std::to_string(iter);
        out_filename += ".png";

        if (color_format == RocalImageColor::ROCAL_COLOR_RGB24) {
            cv::cvtColor(mat_output, mat_color, cv::COLOR_RGB2BGR);
            cv::imwrite(out_filename, mat_color);
        } else {
            cv::imwrite(out_filename, mat_output);
        }
        std::cout << "  Original output saved as: " << out_filename << std::endl;
        col_counter = (col_counter + 1) % number_of_cols;
    }
    
    /*>>>>>>>>>>>>>>>>>>> Test Deserialized Pipeline Execution <<<<<<<<<<<<<<<<<<<*/
    std::cout << "\n=== Testing Deserialized Pipeline Execution ===" << std::endl;
    std::cout << "Available images in deserialized pipeline: " << rocalGetRemainingImages(second_handle) << std::endl;
    
    // Prepare OpenCV matrices for deserialized pipeline output
    // Using the same dimensions as the original pipeline
    cv::Mat mat_deserialized_output(h, w, cv_color_format);
    cv::Mat mat_deserialized_input(h, w, cv_color_format);
    cv::Mat mat_deserialized_color;
    
    // Reset column counter for deserialized pipeline output
    col_counter = 0;
    
    // Run the same number of iterations on the deserialized pipeline to compare outputs
    for (int iter = 0; iter < test_iterations && !rocalIsEmpty(second_handle); iter++) {
        std::cout << "\nDeserialized Pipeline - Iteration " << (iter + 1) << ":" << std::endl;
        
        // Execute one iteration of the deserialized pipeline
        if (rocalRun(second_handle) != 0) {
            std::cout << "rocalRun Failed with runtime error on deserialized pipeline" << std::endl;
            rocalRelease(handle);
            rocalRelease(second_handle);
            return -1;
        }
        
        // Get labels from deserialized pipeline
        RocalTensorList deserialized_labels = rocalGetImageLabels(second_handle);
        
        // Get image names from deserialized pipeline
        int image_name_len_deserialized[inputBatchSize];
        unsigned imagename_size_deserialized = rocalGetImageNameLen(second_handle, image_name_len_deserialized);
        std::vector<char> image_name_deserialized(imagename_size_deserialized);
        rocalGetImageName(second_handle, image_name_deserialized.data());
        std::string image_name_str_deserialized(image_name_deserialized.data());
        
        // Parse and display image names and labels from deserialized pipeline
        int pos = 0;
        int *labels_buffer_deserialized = reinterpret_cast<int *>(deserialized_labels->at(0)->buffer());
        std::vector<std::string> names_deserialized;
        names_deserialized.resize(inputBatchSize);
        
        for (int i = 0; i < inputBatchSize; i++) {
            names_deserialized[i] = image_name_str_deserialized.substr(pos, image_name_len_deserialized[i]);
            pos += image_name_len_deserialized[i];
            std::cout << "  Deserialized - Image: " << names_deserialized[i] << " | Label: " << labels_buffer_deserialized[i] << std::endl;
        }
        
        // Copy processed image data from deserialized pipeline
        rocalCopyToOutput(second_handle, mat_deserialized_input.data, h * w * p);
        // Prepare output image with deserialized prefix for comparison
        mat_deserialized_input.copyTo(mat_deserialized_output(cv::Rect(col_counter * w, 0, w, h)));
        std::string out_filename_deserialized = std::string("deserialized_") + std::string(outName);
        if (display_all)
            out_filename_deserialized += std::to_string(iter);
        out_filename_deserialized += ".png";
        
        // Save the output image with proper color conversion if needed
        if (color_format == RocalImageColor::ROCAL_COLOR_RGB24) {
            cv::cvtColor(mat_deserialized_output, mat_deserialized_color, cv::COLOR_RGB2BGR);
            cv::imwrite(out_filename_deserialized, mat_deserialized_color);
        } else {
            cv::imwrite(out_filename_deserialized, mat_deserialized_output);
        }
        std::cout << "  Deserialized output saved as: " << out_filename_deserialized << std::endl;
        col_counter = (col_counter + 1) % number_of_cols;
    }
    
    std::cout << "\n=== Deserialization Test Completed Successfully ===" << std::endl;
    std::cout << "Both original and deserialized pipelines executed successfully!" << std::endl;

    // Clean up both pipelines
    rocalRelease(handle);
    rocalRelease(second_handle);
    
    return 0;
}
