/*
MIT License

Copyright (c) 2018 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <fstream>
#include <vector>

#include "rocal_api.h"
#include <opencv2/opencv.hpp>

// Checkpointing smoke test: run a few iterations, capture a checkpoint, then finish.
int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 3;  // Minimum CLI args: dataset path, labels path, device flag.
    if (argc < MIN_ARG_COUNT) {
        std::cout << "Usage: checkpoint_test <image_dataset_folder - required> <label_text_file_path> <processing_device=1/cpu=0>  decode_width decode_height <gray_scale:0/rgb:1> decode_shard_counts decoder_type \n";
        return -1;
    }
    int argIdx = 1;                           // Cursor for optional CLI args.
    const char *folderPath1 = argv[argIdx++]; // Dataset folder path.
    const char *label_text_file_path = "";    // Optional label file path.
    int rgb = 1;                              // Process color images by default.
    int decode_width = 0;                     // Optional decode width override.
    int decode_height = 0;                    // Optional decode height override.
    bool processing_device = 0;              // 0=CPU, 1=GPU.
    size_t decode_shard_counts = 1;           // Number of reader shards.
    int decoder_type = 0;                     // Default TurboJpeg decoder.

    if (argc > argIdx)
        label_text_file_path = argv[argIdx++];
    
    if (argc > argIdx)
        processing_device = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decode_width = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decode_height = atoi(argv[argIdx++]);

    if (argc > argIdx)
        rgb = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decode_shard_counts = atoi(argv[argIdx++]);

    if (argc > argIdx)
        decoder_type = atoi(argv[argIdx++]);

    const int inputBatchSize = 1;  // Batch size for the test pipeline.

    // Set the rocAL decoder type
    RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG;  // Selected decoder backend.
    if (decoder_type == 1) {
        rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_OPENCV;
    } else if (decoder_type == 2) {
        rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_ROCJPEG;
        processing_device = 1;  // Requires GPU backend for rocJpeg decoder
    }

    std::cout << ">>> Running on " << (processing_device ? "GPU" : "CPU") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;  // Output color format.

    // Create a pipeline context with checkpointing enabled.
    auto handle = rocalCreate(inputBatchSize, processing_device ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0, 1, 3, ROCAL_FP32, true);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal context\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RocalTensor decoded_output;  // Loader output tensor.

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if (decode_height <= 0 || decode_width <= 0)
        decoded_output = rocalJpegFileSource(handle, folderPath1, color_format, decode_shard_counts, false, true);
    else
        decoded_output = rocalJpegFileSource(handle, folderPath1, color_format, decode_shard_counts, false, true, false,
                                             ROCAL_USE_USER_GIVEN_SIZE, decode_width, decode_height, rocal_decoder_type);
    if (strcmp(label_text_file_path, "") == 0)
        rocalCreateLabelReader(handle, folderPath1);
    else
        rocalCreateTextFileBasedLabelReader(handle, label_text_file_path);
    rocalBrightness(handle, decoded_output, true);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = rocalGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
    }
    // Calling the API to verify and build the augmentation graph
    if (rocalVerify(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph" << std::endl;
        return -1;
    }

    std::cout << "Augmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle) * inputBatchSize;  // Output height in pixels.
    int w = rocalGetOutputWidth(handle);                                                             // Output width in pixels.
    int p = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);                           // Output channel count.
    std::cout << "output width " << w << " output height " << h << " color planes " << p << std::endl;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ? CV_8UC3 : CV_8UC1);  // OpenCV matrix type.

    int ImageNameLen[inputBatchSize];  // Per-sample image name lengths.

    std::vector<std::string> names;  // Decoded image names for the current batch.
    names.resize(inputBatchSize);

    std::cout << "Available images = " << rocalGetRemainingImages(handle) << std::endl;
    int process_image_count = rocalGetRemainingImages(handle);  // Total images to process.
    std::cout << "Process " << process_image_count << " images" << std::endl;

    int counter = 0;                                   // Processed image counter.
    cv::Mat mat_input(h, w, cv_color_format);          // Output buffer for rocalCopyToOutput.

    while (counter < 15 && !rocalIsEmpty(handle)) {
        if (rocalRun(handle) != 0) {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            rocalRelease(handle);
            return -1;
        }

        rocalCopyToOutput(handle, mat_input.data, h * w * p);

        counter += inputBatchSize;
        RocalTensorList labels = rocalGetImageLabels(handle);  // Label tensor list for this batch.

        unsigned imagename_size = rocalGetImageNameLen(handle, ImageNameLen);  // Total name bytes for this batch.
        std::vector<char> imageNames(imagename_size);                          // Name buffer.
        rocalGetImageName(handle, imageNames.data());
        std::string imageNamesStr(imageNames.data());                          // Concatenated names string.

        int pos = 0;                                                           // Offset into names string.
        int *labels_buffer = reinterpret_cast<int *>(labels->at(0)->buffer()); // Pointer to label data.
        for (int i = 0; i < inputBatchSize; i++) {  // Batch index.
            names[i] = imageNamesStr.substr(pos, ImageNameLen[i]);
            pos += ImageNameLen[i];
            std::cout << "name: " << names[i] << " label: " << labels_buffer[i] << std::endl;
        }
        std::cout << std::endl;
    }
    // Capture a checkpoint mid-run (after 15 iterations) and then continue running to completion.
    size_t size_ckpt;  // Serialized checkpoint size in bytes.
    rocalCheckpoint(handle, &size_ckpt);
    std::string serialized_ckpt(size_ckpt, '\0');  // Checkpoint payload buffer.
    rocalGetSerializedCheckpointString(handle, &serialized_ckpt[0]);

    // Save to file
    std::ofstream file("checkpoint.bin", std::ios::binary);  // Output file stream.
    file.write(serialized_ckpt.data(), size_ckpt);
    file.close();

    std::cerr << "The checkpoint size: " << size_ckpt << " bytes\n";
    std::cerr << "The checkpoint -> " << serialized_ckpt << "\n";

    std::cout << "Saved checkpoint\n";
    counter = 0;

    while (!rocalIsEmpty(handle)) {
        if (rocalRun(handle) != 0) {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            break;
        }

        rocalCopyToOutput(handle, mat_input.data, h * w * p);

        counter += inputBatchSize;
        RocalTensorList labels = rocalGetImageLabels(handle);  // Label tensor list for this batch.

        unsigned imagename_size = rocalGetImageNameLen(handle, ImageNameLen);  // Total name bytes for this batch.
        std::vector<char> imageNames(imagename_size);                          // Name buffer.
        rocalGetImageName(handle, imageNames.data());
        std::string imageNamesStr(imageNames.data());                          // Concatenated names string.

        int pos = 0;                                                           // Offset into names string.
        int *labels_buffer = reinterpret_cast<int *>(labels->at(0)->buffer()); // Pointer to label data.
        for (int i = 0; i < inputBatchSize; i++) {  // Batch index.
            names[i] = imageNamesStr.substr(pos, ImageNameLen[i]);
            pos += ImageNameLen[i];
            std::cout << "name: " << names[i] << " label: " << labels_buffer[i] << std::endl;
        }
        std::cout << std::endl;
    }
    
    rocalRelease(handle);

    return 0;
}
