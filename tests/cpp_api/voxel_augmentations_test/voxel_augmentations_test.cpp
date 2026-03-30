/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "rocal_api.h"

using namespace std::chrono;

// Voxel augmentations pipeline test: exercises the new APIs introduced for
// 3D / volumetric (numpy-based) pipelines.
//
// Pipeline:
//   NumpyReader -> RandomObjectBbox -> ROIRandomCrop -> SliceFixed -> Log ->
//   Flip (with depth) -> FlipFixed (with depth) ->
//   Brightness (conditional) -> BrightnessFixed (conditional) ->
//   GaussianNoise (conditional) -> GaussianNoiseFixed (conditional)
//
// Usage:
//   voxel_augmentations_test [gpu=1|cpu=0] [batch_size]
//   Requires ROCAL_DATA_PATH env variable to be set. Reads npy files from $ROCAL_DATA_PATH/rocal_data/voxel_numpy/

int main(int argc, const char **argv) {
    // Usage: ./voxel_augmentations_test <gpu=1/cpu=0> <batch_size>
    // Requires ROCAL_DATA_PATH env variable pointing to the test data directory.
    // Reads npy files from $ROCAL_DATA_PATH/rocal_data/voxel_numpy/
    int argIdx = 1;
    int processing_device = 0;  // processing_device: 0 = CPU, 1 = GPU (default: 0)
    int batch_size = 2;         // batch_size: number of samples per batch (default: 2)

    if (argc > argIdx)
        processing_device = atoi(argv[argIdx++]);
    if (argc > argIdx)
        batch_size = atoi(argv[argIdx++]);

    /*>>>>>>>>>>>>>>>> Getting the path for data  <<<<<<<<<<<<<<<<*/
    std::string rocal_data_path;
    if (std::getenv("ROCAL_DATA_PATH"))
        rocal_data_path = std::getenv("ROCAL_DATA_PATH");
    if (rocal_data_path.empty()) {
        std::cout << "\n ROCAL_DATA_PATH env variable has not been set. " << std::endl;
        return -1;
    }
    std::string npy_folder_str = rocal_data_path + "/rocal_data/voxel_numpy/";
    const char *npy_folder = npy_folder_str.c_str();

    std::cout << ">>> Running voxel augmentations test on " << (processing_device ? "GPU" : "CPU") << std::endl;

    auto processing_mode = processing_device ? RocalProcessMode::ROCAL_PROCESS_GPU
                                             : RocalProcessMode::ROCAL_PROCESS_CPU;

    /*>>>>>>>>>>>>>>>>>>> Create the rocAL context <<<<<<<<<<<<<<<<<<<*/
    auto handle = rocalCreate(batch_size, processing_mode, 0, 1, 3, ROCAL_FP32);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal context" << std::endl;
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    // Numpy file source with NCDHW layout (volumetric), last_batch_policy = DROP
    RocalShardingInfo sharding_info(RocalLastBatchPolicy::ROCAL_LAST_BATCH_DROP,
                                    false, true, -1);
    RocalTensor input = rocalNumpyFileSource(handle, npy_folder, 1,
                                             RocalTensorLayout::ROCAL_NCDHW,
                                             {}, false, true, false, 42,
                                             sharding_info);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Numpy source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // RandomObjectBbox with "start_end" output format
    RocalTensorList bbox_tensors = rocalRandomObjectBbox(handle, input,
                                                         "start_end", 2, 0.4, false);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding rocalRandomObjectBbox (start_end) : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }
    if (bbox_tensors == nullptr || bbox_tensors->size() != 2) {
        std::cout << "rocalRandomObjectBbox expected 2 output tensors (start_end format)" << std::endl;
        return -1;
    }
    RocalTensor roi_start = bbox_tensors->at(0);
    RocalTensor roi_end = bbox_tensors->at(1);

    // RandomObjectBbox with "anchor_shape" format and caching
    RocalTensorList bbox_as = rocalRandomObjectBbox(handle, input,
                                                     "anchor_shape", -1, 1.0, true);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding rocalRandomObjectBbox (anchor_shape) : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }
    if (bbox_as == nullptr || bbox_as->size() != 2) {
        std::cout << "rocalRandomObjectBbox expected 2 output tensors (anchor_shape format)" << std::endl;
        return -1;
    }

    // ROIRandomCrop - random anchor within ROI
    std::vector<int> crop_shape = {1, 128, 128, 128};
    RocalTensor anchor = rocalROIRandomCrop(handle, input, roi_start,
                                            roi_end, crop_shape);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding rocalROIRandomCrop : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // SliceFixed
    std::vector<int> slice_shape = {1, 128, 128, 128};
    std::vector<float> fill_values = {0.0f};
    RocalTensor sliced = rocalSliceFixed(handle, input, false, anchor,
                                         slice_shape, fill_values,
                                         RocalOutOfBoundsPolicy::ROCAL_PAD,
                                         ROCAL_UINT8);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding rocalSliceFixed : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // Log - convert sliced output to float
    RocalTensor log_output = rocalLog(handle, sliced, false);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding rocalLog : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // Flip with depth flag (3D flip)
    RocalIntParam hflip_param = rocalCreateIntUniformRand(0, 1);
    RocalIntParam vflip_param = rocalCreateIntUniformRand(0, 1);
    RocalIntParam dflip_param = rocalCreateIntUniformRand(0, 1);

    RocalTensor flipped = rocalFlip(handle, log_output, false,
                                     hflip_param, vflip_param, dflip_param,
                                     RocalTensorLayout::ROCAL_NCDHW, ROCAL_FP32);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding rocalFlip : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // FlipFixed with depth flag
    RocalTensor flip_fixed = rocalFlipFixed(handle, flipped, 1, 0, false, 1,
                                            RocalTensorLayout::ROCAL_NCDHW, ROCAL_FP32);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding rocalFlipFixed : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // Brightness with conditional execution
    RocalFloatParam brightness_param = rocalCreateFloatUniformRand(0.7f, 1.3f);
    RocalIntParam cond_exec_param = rocalCreateIntUniformRand(0, 1);

    RocalTensor brightness_out = rocalBrightness(handle, flip_fixed, false,
                                                  brightness_param, NULL, cond_exec_param,
                                                  RocalTensorLayout::ROCAL_NCDHW, ROCAL_FP32);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding rocalBrightness : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // BrightnessFixed with conditional execution
    RocalTensor brightness_fixed_out = rocalBrightnessFixed(handle, brightness_out,
                                                            1.0f, 0.0f, false, 1,
                                                            RocalTensorLayout::ROCAL_NCDHW,
                                                            ROCAL_FP32);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding rocalBrightnessFixed : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // GaussianNoise with conditional execution
    RocalIntParam noise_cond_param = rocalCreateIntUniformRand(0, 1);
    RocalTensor noise_out = rocalGaussianNoise(handle, brightness_fixed_out, false,
                                               NULL, NULL, 0, noise_cond_param,
                                               RocalTensorLayout::ROCAL_NCDHW, ROCAL_FP32);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding rocalGaussianNoise : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // GaussianNoiseFixed with conditional execution
    RocalTensor noise_fixed_out = rocalGaussianNoiseFixed(handle, noise_out, true,
                                                          0.0f, 0.1f, 0, 1,
                                                          RocalTensorLayout::ROCAL_NCDHW,
                                                          ROCAL_FP32);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Error while adding rocalGaussianNoiseFixed : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    // Calling the API to verify and build the augmentation graph
    if (rocalVerify(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph" << std::endl;
        return -1;
    }

    std::cout << "\n\nAugmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    /*>>>>>>>>>>>>>>>>>>> Pipeline execution <<<<<<<<<<<<<<<<<<<*/
    printf("Remaining images %lu \n", rocalGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;

    while (rocalGetRemainingImages(handle) >= (size_t)batch_size) {
        index++;
        if (rocalRun(handle) != 0) {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            rocalRelease(handle);
            return -1;
        }

        RocalTensorList output_tensors = rocalGetOutputTensors(handle);
        for (uint64_t t = 0; t < output_tensors->size(); t++) {
            auto tensor = output_tensors->at(t);
            auto dims = tensor->dims();
            std::cout << "Output tensor dims=(";
            for (size_t d = 0; d < dims.size(); d++) {
                if (d > 0) std::cout << ",";
                std::cout << dims[d];
            }
            std::cout << ")"<< std::endl;
        }
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << "Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalResetLoaders(handle);
    rocalRelease(handle);
    return 0;
}
