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

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "rocal_api.h"
using namespace cv;

#if ENABLE_HIP
#include <half/half.hpp>
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#endif

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
#define RANDOMBBOXCROP

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

// Introduce function to generate Bbox anchors for Box IOU matcher
int get_anchors(std::vector<float>& anchors, std::string anchors_file_path) {
    std::ifstream fin(anchors_file_path, std::ios::binary);  // Open the binary file for reading

    if (!fin.is_open()) {
        std::cout << "Error: Unable to open the input binary file\n";
        return -1;
    }

    // Get the size of the file
    fin.seekg(0, std::ios::end);
    std::streampos fileSize = fin.tellg();
    fin.seekg(0, std::ios::beg);

    std::size_t numFloats = fileSize / sizeof(float);

    anchors.resize(numFloats);

    // Read the floats from the file
    fin.read(reinterpret_cast<char *>(anchors.data()), fileSize);

    if (fin.fail()) {
        std::cout << "Error: Failed to read from the input binary file\n";
        return -1;
    }

    fin.close();

    return 0;
}

int test(int test_case, int reader_type, const char *path, const char *outName, int rgb, int gpu, int width, int height, int num_of_classes, int display_all, int resize_interpolation_type, int resize_scaling_mode, int memcpy_backend, int output_layout, int reverse_channels);
int main(int argc, const char **argv) {
    // check command-line usage
    const int MIN_ARG_COUNT = 6;
    if (argc < MIN_ARG_COUNT) {
        printf("Usage: unit_tests reader-type <image-dataset-folder> output_image_name <width> <height> test_case gpu=1/cpu=0 rgb=1/grayscale=0 one_hot_labels=num_of_classes/0  display_all=0(display_last_only)1(display_all) <memcpy_host=0/memcpy_gpu=1> <nhwc=0/nchw=1> <reverse_channels=0/1> \n");
        return -1;
    }

    int argIdx = 1;
    int reader_type = atoi(argv[argIdx++]);
    const char *path = argv[argIdx++];
    const char *outName = argv[argIdx++];
    int width = atoi(argv[argIdx++]);
    int height = atoi(argv[argIdx++]);
    int display_all = 0;

    int rgb = 1;  // process color images
    bool gpu = 1;
    int test_case = 3;  // For Rotate
    int num_of_classes = 0;
    int resize_interpolation_type = 1;  // For Bilinear interpolations
    int resize_scaling_mode = 0;        // For Default scaling mode
    int memcpy_backend = 0;              // For MEMCPY_HOST
    int output_layout = 0;              // For NHWC tensor layout
    int reverse_channels = 0;

    if (argc > argIdx)
        test_case = atoi(argv[argIdx++]);

    if (argc > argIdx)
        gpu = atoi(argv[argIdx++]);

    if (argc > argIdx)
        rgb = atoi(argv[argIdx++]);

    if (argc > argIdx)
        num_of_classes = atoi(argv[argIdx++]);

    if (argc > argIdx)
        display_all = atoi(argv[argIdx++]);

    if (argc > argIdx)
        resize_interpolation_type = atoi(argv[argIdx++]);

    if (argc > argIdx)
        resize_scaling_mode = atoi(argv[argIdx++]);
    
    if (argc > argIdx)
        memcpy_backend = atoi(argv[argIdx++]);

    if (argc > argIdx)
        output_layout = atoi(argv[argIdx++]);

    if (argc > argIdx)
        reverse_channels = atoi(argv[argIdx++]);

    return test(test_case, reader_type, path, outName, rgb, gpu, width, height, num_of_classes, display_all, resize_interpolation_type, resize_scaling_mode, memcpy_backend, output_layout, reverse_channels);
}

int test(int test_case, int reader_type, const char *path, const char *outName, int rgb, int gpu, int width, int height, int num_of_classes, int display_all, int resize_interpolation_type, int resize_scaling_mode, int memcpy_backend, int output_layout, int reverse_channels) {
    size_t num_threads = 1;
    const unsigned int input_batch_size = 2;
    int decode_max_width = width;
    int decode_max_height = height;
    int pipeline_type = -1;
    bool use_pixelwise_masks = false;
    std::cout << "Test case " << test_case << std::endl;
    std::cout << "Running on " << (gpu ? "GPU" : "CPU") << " , " << (rgb ? " Color " : " Grayscale ") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24
                                              : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(input_batch_size,
                              gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                              1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    // Some image augmentations add "meta nodes" which currently propagate bbox/polygon metadata but not pixelwise masks.
    // For the pixelwise mask reader test (reader_type == 16), keep the graph metadata-only w.r.t masks to validate the reader + mask APIs.
    if (reader_type == 16 && test_case != 61) {
        std::cout << "INFO: Overriding test_case " << test_case << " -> 61 (rocalNop) for pixelwise mask validation\n";
        test_case = 61;
    }

    /*>>>>>>>>>>>>>>>> Getting the path for data  <<<<<<<<<<<<<<<<*/

    std::string rocal_data_path;
    if (std::getenv("ROCAL_DATA_PATH"))
        rocal_data_path = std::getenv("ROCAL_DATA_PATH");

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalSetSeed(0);
    auto seed = rocalGetSeed();
    std::cout << "Seed set for rocAL pipeline: " << seed << "\n";

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalIntParam color_temp_adj = rocalCreateIntParameter(-50);
    RocalIntParam mirror = rocalCreateIntParameter(1);
    std::vector<int> values = {0, 1};
    std::vector<double> frequencies = {0.5, 0.5};
    RocalIntParam rand_prob = rocalCreateIntRand(values.data(), frequencies.data(), values.size());
    rocalUpdateIntRand(values.data(), frequencies.data(), values.size(), rand_prob);

    RocalFloatParam float_param = rocalCreateFloatParameter(1.0f);
    rocalUpdateFloatParameter(2.0f, float_param);
    rocalGetFloatValue(float_param);

    RocalIntParam uniform_int_param = rocalCreateIntUniformRand(0, 1);
    rocalUpdateIntUniformRand(0, 2, uniform_int_param);
    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/

#if defined RANDOMBBOXCROP
    bool all_boxes_overlap = true;
    bool no_crop = false;
#endif
    bool enable_iou_matcher = false;

    RocalTensor decoded_output;
    RocalTensorLayout output_tensor_layout = (rgb != 0) ? RocalTensorLayout::ROCAL_NHWC : RocalTensorLayout::ROCAL_NCHW;
    RocalTensorOutputType output_tensor_dtype = RocalTensorOutputType::ROCAL_UINT8;
    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    switch (reader_type) {
        case 1:  // image_partial decode
        {
            std::cout << "Running PARTIAL DECODE" << std::endl;
            pipeline_type = 1;
            rocalCreateLabelReader(handle, path);
            std::vector<float> area = {0.08, 1};
            std::vector<float> aspect_ratio = {3.0f / 4, 4.0f / 3};
            decoded_output = rocalFusedJpegCrop(handle, path, color_format, num_threads, false, area, aspect_ratio, 10, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 2:  // coco detection
        {
            std::cout << "Running COCO READER" << std::endl;
            pipeline_type = 2;
            if (strcmp(rocal_data_path.c_str(), "") == 0) {
                std::cout << "\n ROCAL_DATA_PATH env variable has not been set. ";
                exit(0);
            }
            // setting the default json path to ROCAL_DATA_PATH coco sample train annotation
            std::string json_path = rocal_data_path + "/rocal_data/coco/coco_10_img/annotations/coco_data.json";
            rocalCreateCOCOReader(handle, json_path.c_str(), true);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                decoded_output = rocalJpegCOCOFileSource(handle, path, json_path.c_str(), color_format, num_threads, false, true, false);
            else
                decoded_output = rocalJpegCOCOFileSource(handle, path, json_path.c_str(), color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 3:  // coco detection partial
        {
            std::cout << "Running COCO READER PARTIAL" << std::endl;
            pipeline_type = 2;
            if (strcmp(rocal_data_path.c_str(), "") == 0) {
                std::cout << "\n ROCAL_DATA_PATH env variable has not been set. ";
                exit(0);
            }
            // setting the default json path to ROCAL_DATA_PATH coco sample train annotation
            std::string json_path = rocal_data_path + "/rocal_data/coco/coco_10_img/annotations/coco_data.json";
            rocalCreateCOCOReader(handle, json_path.c_str(), true);
#if defined RANDOMBBOXCROP
            rocalRandomBBoxCrop(handle, all_boxes_overlap, no_crop);
#endif
            std::vector<float> area = {0.08, 1};
            std::vector<float> aspect_ratio = {3.0f / 4, 4.0f / 3};
            decoded_output = rocalJpegCOCOFileSourcePartial(handle, path, json_path.c_str(), color_format, num_threads, false, area, aspect_ratio, 10, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 4:  // tf classification
        {
            std::cout << "Running TF CLASSIFICATION READER" << std::endl;
            pipeline_type = 1;
            char key1[25] = "image/encoded";
            char key2[25] = "image/class/label";
            char key8[25] = "image/filename";
            rocalCreateTFReader(handle, path, true, key2, key8);
            decoded_output = rocalJpegTFRecordSource(handle, path, color_format, num_threads, false, key1, key8, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 5:  // tf detection
        {
            std::cout << "Running TF DETECTION READER" << std::endl;
            pipeline_type = 2;
            char key1[25] = "image/encoded";
            char key2[25] = "image/object/class/label";
            char key3[25] = "image/object/class/text";
            char key4[25] = "image/object/bbox/xmin";
            char key5[25] = "image/object/bbox/ymin";
            char key6[25] = "image/object/bbox/xmax";
            char key7[25] = "image/object/bbox/ymax";
            char key8[25] = "image/filename";
            rocalCreateTFReaderDetection(handle, path, true, key2, key3, key4, key5, key6, key7, key8);
            decoded_output = rocalJpegTFRecordSource(handle, path, color_format, num_threads, false, key1, key8, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 6:  // caffe classification
        {
            std::cout << "Running CAFFE CLASSIFICATION READER" << std::endl;
            pipeline_type = 1;
            rocalCreateCaffeLMDBLabelReader(handle, path);
            decoded_output = rocalJpegCaffeLMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 7:  // caffe detection
        {
            std::cout << "Running CAFFE DETECTION READER" << std::endl;
            pipeline_type = 2;
            rocalCreateCaffeLMDBReaderDetection(handle, path);
            decoded_output = rocalJpegCaffeLMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 8:  // caffe2 classification
        {
            std::cout << "Running CAFFE2 CLASSIFICATION READER" << std::endl;
            pipeline_type = 1;
            rocalCreateCaffe2LMDBLabelReader(handle, path, true);
            decoded_output = rocalJpegCaffe2LMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 9:  // caffe2 detection
        {
            std::cout << "Running CAFFE2 DETECTION READER" << std::endl;
            pipeline_type = 2;
            rocalCreateCaffe2LMDBReaderDetection(handle, path, true);
            decoded_output = rocalJpegCaffe2LMDBRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 10:  // coco reader keypoints
        {
            std::cout << "Running COCO KEYPOINTS READER" << std::endl;
            pipeline_type = 3;
            if (strcmp(rocal_data_path.c_str(), "") == 0) {
                std::cout << "\n ROCAL_DATA_PATH env variable has not been set. ";
                exit(0);
            }
            // setting the default json path to ROCAL_DATA_PATH coco sample train annotation
            std::string json_path = rocal_data_path + "/rocal_data/coco/coco_10_img_keypoints/annotations/person_keypoints_val2017.json";
            float sigma = 3.0;
            rocalCreateCOCOReaderKeyPoints(handle, json_path.c_str(), true, sigma, (unsigned)width, (unsigned)height);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                decoded_output = rocalJpegCOCOFileSource(handle, path, json_path.c_str(), color_format, num_threads, false, true, false);
            else
                decoded_output = rocalJpegCOCOFileSource(handle, path, json_path.c_str(), color_format, num_threads, false, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 11:  // mxnet reader
        {
            std::cout << "Running MXNET READER" << std::endl;
            pipeline_type = 1;
            rocalCreateMXNetReader(handle, path, true);
            decoded_output = rocalMXNetRecordSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 12:  // web_dataset reader
        {
            std::cout << "Running WEB DATASET READER" << std::endl;
            pipeline_type = 4;
            std::vector<std::set<std::string>> extensions = {
                {"JPEG", "cls"},
            };
            rocalCreateWebDatasetReader(handle, path, "", extensions, RocalMissingComponentsBehaviour::ROCAL_MISSING_COMPONENT_ERROR, true);
            decoded_output = rocalWebDatasetSourceSingleShard(handle, path, "", color_format, 0, 1, false, false, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);
        } break;
        case 13:  // Numpy reader
        {
            std::cout << "Running Numpy reader" << std::endl;
            pipeline_type = 5;
            decoded_output = rocalNumpyFileSource(handle, path, num_threads, RocalTensorLayout::ROCAL_NHWC);
        } break;
        case 14:  // image_partial decode
        {
            std::cout << "Running PARTIAL DECODE - SINGLE SHARD" << std::endl;
            pipeline_type = 1;
            rocalCreateLabelReader(handle, path);
            std::vector<float> area = {0.08, 1};
            std::vector<float> aspect_ratio = {3.0f / 4, 4.0f / 3};
            RocalDecoderType dec_type = gpu ? RocalDecoderType::ROCAL_DECODER_ROCJPEG : RocalDecoderType::ROCAL_DECODER_TJPEG;
            decoded_output = rocalFusedJpegCropSingleShard(handle, path, color_format, 0, 1, false, area, aspect_ratio, 10, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height, RocalShardingInfo(), dec_type);
        } break;
        case 15:  // coco detection with Box IOU matcher
        {
            std::cout << "Running COCO READER - SINGLE SHARD" << std::endl;
            pipeline_type = 2;
            if (strcmp(rocal_data_path.c_str(), "") == 0) {
                std::cout << "\n ROCAL_DATA_PATH env variable has not been set. ";
                exit(0);
            }
            // setting the default json path to ROCAL_DATA_PATH coco sample train annotation
            std::string json_path = rocal_data_path + "/rocal_data/coco/coco_10_img/annotations/coco_data.json";
            rocalCreateCOCOReader(handle, json_path.c_str(), true, false, true, false, false, true, true);  // Enable Box IOU matcher
            if (decode_max_height <= 0 || decode_max_width <= 0)
                decoded_output = rocalJpegCOCOFileSourceSingleShard(handle, path, json_path.c_str(), color_format, 0, 1, false, true, false);
            else
                decoded_output = rocalJpegCOCOFileSourceSingleShard(handle, path, json_path.c_str(), color_format, 0, 1, false, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);

            // Box IOU matcher - used for Retinanet training
            std::vector<float> coco_anchors;
            std::string anchors_path = rocal_data_path + "/rocal_data/coco/coco_anchors/retinanet_anchors.bin";
            if (get_anchors(coco_anchors, anchors_path) != 0)
                return -1;
            enable_iou_matcher = true;
            rocalBoxIouMatcher(handle, coco_anchors, 0.5, 0.4, true);
        } break;
        case 16:  // coco segmentation
        {
            std::cout << "Running COCO SEGMENTATION READER - SINGLE SHARD" << std::endl;
            pipeline_type = 6;
            use_pixelwise_masks = true;
            if (strcmp(rocal_data_path.c_str(), "") == 0) {
                std::cout << "\n ROCAL_DATA_PATH env variable has not been set. ";
                exit(0);
            }
            std::string json_path = rocal_data_path + "/rocal_data/coco/coco_10_img_keypoints/annotations/person_keypoints_val2017.json";
            rocalCreateCOCOReader(handle, json_path.c_str(), true, false, true);
            // Configure the random pixel selector: pick from foreground (value > 0)
            rocalSetRandomPixelMaskConfig(handle, true, 0, true);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                decoded_output = rocalJpegCOCOFileSourceSingleShard(handle, path, json_path.c_str(), color_format, 0, 1, false, true, false);
            else
                decoded_output = rocalJpegCOCOFileSourceSingleShard(handle, path, json_path.c_str(), color_format, 0, 1, false, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 17:  // caffe classification
        {
            std::cout << "Running CAFFE CLASSIFICATION READER - SINGLE SHARD" << std::endl;
            pipeline_type = 1;
            rocalCreateCaffeLMDBLabelReader(handle, path);
            RocalShardingInfo sharding_info = RocalShardingInfo(RocalLastBatchPolicy::ROCAL_LAST_BATCH_DROP, true, false, -1);
            decoded_output = rocalJpegCaffeLMDBRecordSourceSingleShard(handle, path, color_format, 0, 1, false, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height, RocalDecoderType::ROCAL_DECODER_TJPEG, sharding_info);
        } break;
        case 18:  // caffe2 classification
        {
            std::cout << "Running CAFFE2 CLASSIFICATION READER - SINGLE SHARD" << std::endl;
            pipeline_type = 1;
            rocalCreateCaffe2LMDBLabelReader(handle, path, true);
            RocalShardingInfo sharding_info = RocalShardingInfo(RocalLastBatchPolicy::ROCAL_LAST_BATCH_DROP, true, false, -1);
            decoded_output = rocalJpegCaffe2LMDBRecordSourceSingleShard(handle, path, color_format, 0, 1, false, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height, RocalDecoderType::ROCAL_DECODER_TJPEG, sharding_info);
        } break;
        case 19:  // mxnet reader
        {
            std::cout << "Running MXNET READER - SINGLE SHARD" << std::endl;
            pipeline_type = 1;
            rocalCreateMXNetReader(handle, path, true);
            RocalShardingInfo sharding_info = RocalShardingInfo(RocalLastBatchPolicy::ROCAL_LAST_BATCH_DROP, true, false, -1);
            decoded_output = rocalMXNetRecordSourceSingleShard(handle, path, color_format, 0, 1, false, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height, RocalDecoderType::ROCAL_DECODER_TJPEG, sharding_info);
        } break;
        case 20:  // Numpy reader
        {
            std::cout << "Running Numpy reader - SINGLE SHARD" << std::endl;
            pipeline_type = 5;
            RocalShardingInfo sharding_info = RocalShardingInfo(RocalLastBatchPolicy::ROCAL_LAST_BATCH_DROP, true, false, -1);
            decoded_output = rocalNumpyFileSourceSingleShard(handle, path, RocalTensorLayout::ROCAL_NHWC, {}, false, true, false, 0, 1, seed, sharding_info);
        } break;
        case 21: {
            std::cout << "Running IMAGE READER - SINGLE SHARD" << std::endl;
            pipeline_type = 1;
            rocalCreateLabelReader(handle, path);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                decoded_output = rocalJpegFileSourceSingleShard(handle, path, color_format, 0, 1, false, true);
            else
                decoded_output = rocalJpegFileSourceSingleShard(handle, path, color_format, 0, 1, false, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 22:  // caffe classification
        {
            std::cout << "Running CAFFE CLASSIFICATION PARTIAL READER - SINGLE SHARD" << std::endl;
            pipeline_type = 1;
            rocalCreateCaffeLMDBLabelReader(handle, path);
            std::vector<float> area = {0.08, 1};
            std::vector<float> aspect_ratio = {3.0f / 4, 4.0f / 3};
            decoded_output = rocalJpegCaffeLMDBRecordSourcePartialSingleShard(handle, path, color_format, 0, 1, false, area, aspect_ratio, 10, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 23:  // caffe2 classification
        {
            std::cout << "Running CAFFE2 CLASSIFICATION PARTIAL READER - SINGLE SHARD" << std::endl;
            pipeline_type = 1;
            rocalCreateCaffe2LMDBLabelReader(handle, path, true);
            std::vector<float> area = {0.08, 1};
            std::vector<float> aspect_ratio = {3.0f / 4, 4.0f / 3};
            decoded_output = rocalJpegCaffe2LMDBRecordSourcePartialSingleShard(handle, path, color_format, 0, 1, false, area, aspect_ratio, 10, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;
        case 24:  // tf classification
        {
            std::cout << "Running TF CLASSIFICATION READER - SINGLE SHARD" << std::endl;
            pipeline_type = 1;
            char key1[25] = "image/encoded";
            char key2[25] = "image/class/label";
            char key8[25] = "image/filename";
            rocalCreateTFReader(handle, path, true, key2, key8);
            RocalShardingInfo sharding_info = RocalShardingInfo(RocalLastBatchPolicy::ROCAL_LAST_BATCH_DROP, true, false, -1);
            decoded_output = rocalJpegTFRecordSourceSingleShard(handle, path, color_format, 0, 1, false, key1, key8, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height, RocalDecoderType::ROCAL_DECODER_TJPEG, sharding_info);
        } break;
        case 25:  // web_dataset reader
        {
            std::cout << "Running WEB DATASET READER - WITH IDX FILE" << std::endl;
            pipeline_type = 4;
            std::vector<std::set<std::string>> extensions = {
                {"JPEG", "cls"},
            };
            std::string idx_file_path = rocal_data_path + "/rocal_data/web_dataset/idx_file/";
            rocalCreateWebDatasetReader(handle, path, idx_file_path.c_str(), extensions, RocalMissingComponentsBehaviour::ROCAL_MISSING_COMPONENT_ERROR, true);
            RocalShardingInfo sharding_info = RocalShardingInfo(RocalLastBatchPolicy::ROCAL_LAST_BATCH_DROP, true, false, -1);
            decoded_output = rocalWebDatasetSourceSingleShard(handle, path, idx_file_path.c_str(), color_format, 0, 1, false, true, false, ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height, RocalDecoderType::ROCAL_DECODER_TJPEG, sharding_info);
        } break;
        case 26:  // coco detection with Box encoder
        {
            std::cout << "Running COCO READER - SINGLE SHARD" << std::endl;
            pipeline_type = 7;
            if (strcmp(rocal_data_path.c_str(), "") == 0) {
                std::cout << "\n ROCAL_DATA_PATH env variable has not been set. ";
                exit(0);
            }
            // setting the default json path to ROCAL_DATA_PATH coco sample train annotation
            std::string json_path = rocal_data_path + "/rocal_data/coco/coco_10_img/annotations/coco_data.json";
            rocalCreateCOCOReader(handle, json_path.c_str(), true, false, true, true);  // Enable IOU matcher
            if (decode_max_height <= 0 || decode_max_width <= 0)
                decoded_output = rocalJpegCOCOFileSourceSingleShard(handle, path, json_path.c_str(), color_format, 0, 1, false, true, false);
            else
                decoded_output = rocalJpegCOCOFileSourceSingleShard(handle, path, json_path.c_str(), color_format, 0, 1, false, true, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
            
            // Box Encoder - used for SSD training
            std::vector<float> coco_anchors;
            std::vector<float> mean = {0.0, 0.0, 0.0, 0.0};
            std::vector<float> stddev = {1.0, 1.0, 1.0, 1.0};
            std::string anchors_path = rocal_data_path + "/rocal_data/coco/coco_anchors/coco_anchors.bin";
            if (get_anchors(coco_anchors, anchors_path) != 0)
                return -1;
            rocalBoxEncoder(handle, coco_anchors, 0.5, mean, stddev);
        } break;
        case 27:  // raw tfrecord reader
        {
            std::cout << "Running RAW TFRECORD READER" << std::endl;
            pipeline_type = 1;
            char key1[25] = "image/decoded";
            char key2[25] = "image/class/label";
            char key8[25] = "image/filename";
            rocalCreateTFReader(handle, path, true, key2, key8);
            decoded_output = rocalRawTFRecordSource(handle, path, key1, key8, color_format, false, false, false, decode_max_width, decode_max_height);
        } break;
        case 28:  // raw tfrecord reader
        {
            std::cout << "Running RAW TFRECORD READER - SINGLE SHARD" << std::endl;
            pipeline_type = 1;
            char key1[25] = "image/decoded";
            char key2[25] = "image/class/label";
            char key8[25] = "image/filename";
            rocalCreateTFReader(handle, path, true, key2, key8);
            decoded_output = rocalRawTFRecordSourceSingleShard(handle, path, key1, key8, color_format, 0, 1, false, true, false, decode_max_width, decode_max_height);
        } break;
        default: {
            std::cout << "Running IMAGE READER" << std::endl;
            pipeline_type = 1;
            rocalCreateLabelReader(handle, path);
            if (decode_max_height <= 0 || decode_max_width <= 0)
                decoded_output = rocalJpegFileSource(handle, path, color_format, num_threads, false, true);
            else
                decoded_output = rocalJpegFileSource(handle, path, color_format, num_threads, false, false, false, ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED, decode_max_width, decode_max_height);
        } break;

    }

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    int resize_w = width, resize_h = height;  // height and width

    RocalTensor input = decoded_output;
    // RocalTensor input = rocalResize(handle, decoded_output, resize_w, resize_h, false); // uncomment when processing images of different size
    RocalTensor output;

    if ((test_case == 48 || test_case == 49 || test_case == 50 || test_case == 21 || test_case == 22 || test_case == 24 || test_case == 16 || test_case == 43 || test_case == 71 || test_case == 72 || test_case == 64 || test_case == 75 || test_case == 83 || reader_type == 13 || reader_type == 21 || reader_type == 27 || reader_type == 28) && rgb == 0) {
        std::cout << "Not a valid option! Exiting!\n";
        rocalRelease(handle);
        return -1;
    }
    if ((test_case == 71 || test_case == 72) && gpu == 1) {
        std::cout << "Not a valid option! Exiting!\n";
        rocalRelease(handle);
        return -1;
    }
    switch (test_case) {
        case 0: {
            std::cout << "Running rocalResize" << std::endl;
            resize_w = 400;
            resize_h = 400;
            std::string interpolation_type_name, scaling_node_name;
            RocalResizeInterpolationType interpolation_type;
            RocalResizeScalingMode scale_mode;
            interpolation_type_name = get_interpolation_type(resize_interpolation_type, interpolation_type);
            scaling_node_name = get_scaling_mode(resize_scaling_mode, scale_mode);
            std::cerr << " \n Interpolation_type_name " << interpolation_type_name;
            std::cerr << " \n Scaling_node_name " << scaling_node_name << std::endl;
            if (scale_mode != ROCAL_SCALING_MODE_DEFAULT && interpolation_type != ROCAL_LINEAR_INTERPOLATION) {  // (Reference output available for bilinear interpolation for this
                std::cerr << " \n Running " << scaling_node_name << " scaling mode with Bilinear interpolation for comparison \n";
                interpolation_type = ROCAL_LINEAR_INTERPOLATION;
            }
            if (scale_mode == ROCAL_SCALING_MODE_STRETCH)  // For reference Output comparison
                output = rocalResize(handle, input, resize_w, 0, true, scale_mode, {}, 0, 0, interpolation_type);
            else
                output = rocalResize(handle, input, resize_w, resize_h, true, scale_mode, {}, 0, 0, interpolation_type);
        } break;
        case 1: {
            std::cout << "Running rocalCropResize" << std::endl;
            output = rocalCropResize(handle, input, resize_w, resize_h, true);
        } break;
        case 2: {
            std::cout << "Running rocalRotate" << std::endl;
            output = rocalRotate(handle, input, true);
        } break;
        case 3: {
            std::cout << "Running rocalBrightness" << std::endl;
            output = rocalBrightness(handle, input, true);
        } break;
        case 4: {
            std::cout << "Running rocalGamma" << std::endl;
            output = rocalGamma(handle, input, true);
        } break;
        case 5: {
            std::cout << "Running rocalContrast" << std::endl;
            output = rocalContrast(handle, input, true);
        } break;
        case 6: {
            std::cout << "Running rocalFlip" << std::endl;
            output = rocalFlip(handle, input, true);
        } break;
        case 7: {
            std::cout << "Running rocalBlur" << std::endl;
            output = rocalBlur(handle, input, true);
        } break;
        case 8: {
            std::cout << "Running rocalBlend" << std::endl;
            RocalTensor output_1 = rocalRotate(handle, input, false);
            output = rocalBlend(handle, input, output_1, true);
        } break;
        case 9: {
            std::cout << "Running rocalWarpAffine" << std::endl;
            output = rocalWarpAffine(handle, input, true);
        } break;
        case 10: {
            std::cout << "Running rocalFishEye" << std::endl;
            output = rocalFishEye(handle, input, true);
        } break;
        case 11: {
            std::cout << "Running rocalVignette" << std::endl;
            output = rocalVignette(handle, input, true);
        } break;
        case 12: {
            std::cout << "Running rocalJitter" << std::endl;
            output = rocalJitter(handle, input, true);
        } break;
        case 13: {
            std::cout << "Running rocalSnPNoise" << std::endl;
            output = rocalSnPNoise(handle, input, true);
        } break;
        case 14: {
            std::cout << "Running rocalSnow" << std::endl;
            output = rocalSnow(handle, input, true);
        } break;
        case 15: {
            std::cout << "Running rocalRain" << std::endl;
            output = rocalRain(handle, input, true);
        } break;
        case 16: {
            std::cout << "Running rocalColorTemp" << std::endl;
            output = rocalColorTemp(handle, input, true);
        } break;
        case 17: {
            std::cout << "Running rocalFog" << std::endl;
            output = rocalFog(handle, input, true);
        } break;
        case 18: {
            std::cout << "Running rocalLensCorrection" << std::endl;
            CameraMatrix sampleCameraMatrix = {534.07088364, 341.53407554, 534.11914595, 232.94565259};
            DistortionCoeffs sampleDistortionCoeffs = {-0.29297164, 0.10770696, 0.00131038, -0.0000311, 0.0434798};
            std::vector<CameraMatrix> cameraMatrixVector;
            std::vector<DistortionCoeffs> distortionCoeffsVector;
            for (unsigned i = 0; i < input_batch_size; i++) {
                cameraMatrixVector.push_back(sampleCameraMatrix);
                distortionCoeffsVector.push_back(sampleDistortionCoeffs);
            }
            output = rocalLensCorrection(handle, input, cameraMatrixVector, distortionCoeffsVector, true);
        } break;
        case 19: {
            std::cout << "Running rocalPixelate" << std::endl;
            output = rocalPixelate(handle, input, true);
        } break;
        case 20: {
            std::cout << "Running rocalExposure" << std::endl;
            output = rocalExposure(handle, input, true);
        } break;
        case 21: {
            std::cout << "Running rocalHue" << std::endl;
            output = rocalHue(handle, input, true);
        } break;
        case 22: {
            std::cout << "Running rocalSaturation" << std::endl;
            output = rocalSaturation(handle, input, true);
        } break;
        case 23: {
            std::cout << "Running rocalCopy" << std::endl;
            output = rocalCopy(handle, input, true);
        } break;
        case 24: {
            std::cout << "Running rocalColorTwist" << std::endl;
            output = rocalColorTwist(handle, input, true);
        } break;
        case 25: {
            std::cout << "Running rocalCropMirrorNormalize" << std::endl;
            std::vector<float> mean = {128, 128, 128};
            std::vector<float> std_dev = {1.2, 1.2, 1.2};
            output = rocalCropMirrorNormalize(handle, input, 224, 224, 0, 0, mean, std_dev, true, mirror, output_tensor_layout, output_tensor_dtype);
        } break;
        case 26: {
            std::cout << "Running rocalCrop" << std::endl;
            output = rocalCrop(handle, input, true);
        } break;
        case 27: {
            std::cout << "Running rocalResizeCropMirror" << std::endl;
            output = rocalResizeCropMirror(handle, input, resize_w, resize_h, true);
        } break;

        case 30: {
            std::cout << "Running rocalCropResizeFixed" << std::endl;
            output = rocalCropResizeFixed(handle, input, resize_w, resize_h, true, 0.25, 1.2, 0.6, 0.4);
        } break;
        case 31: {
            std::cout << "Running rocalRotateFixed" << std::endl;
            output = rocalRotateFixed(handle, input, 45, true);
        } break;
        case 32: {
            std::cout << "Running rocalBrightnessFixed" << std::endl;
            output = rocalBrightnessFixed(handle, input, 1.90, 20, true);
        } break;
        case 33: {
            std::cout << "Running rocalGammaFixed" << std::endl;
            output = rocalGammaFixed(handle, input, 0.5, true);
        } break;
        case 34: {
            std::cout << "Running rocalContrastFixed" << std::endl;
            output = rocalContrastFixed(handle, input, 30, 80, true);
        } break;
        case 36: {
            std::cout << "Running rocalBlendFixed" << std::endl;
            RocalTensor output_1 = rocalRotateFixed(handle, input, 45, false);
            output = rocalBlendFixed(handle, input, output_1, 0.5, true);
        } break;
        case 37: {
            std::cout << "Running rocalWarpAffineFixed" << std::endl;
            output = rocalWarpAffineFixed(handle, input, 1, 1, 0.5, 0.5, 7, 7, true);
        } break;
        case 38: {
            std::cout << "Running rocalVignetteFixed" << std::endl;
            output = rocalVignetteFixed(handle, input, 50, true);
        } break;
        case 39: {
            std::cout << "Running rocalJitterFixed" << std::endl;
            output = rocalJitterFixed(handle, input, 3, true);
        } break;
        case 40: {
            std::cout << "Running rocalSnPNoiseFixed" << std::endl;
            output = rocalSnPNoiseFixed(handle, input, 0.2, 0.2, 0.2, 0.5, true, 0);
        } break;
        case 41: {
            std::cout << "Running rocalSnowFixed" << std::endl;
            output = rocalSnowFixed(handle, input, 0.2, true);
        } break;
        case 42: {
            std::cout << "Running rocalRainFixed" << std::endl;
            output = rocalRainFixed(handle, input, true, 7, 1, 6, 0, 0.4);
        } break;
        case 43: {
            std::cout << "Running rocalColorTempFixed" << std::endl;
            output = rocalColorTempFixed(handle, input, 70, true);
        } break;
        case 44: {
            std::cout << "Running rocalFogFixed" << std::endl;
            output = rocalFogFixed(handle, input, 0.1, 0.3, true);
        } break;
        case 46: {
            std::cout << "Running rocalExposureFixed" << std::endl;
            output = rocalExposureFixed(handle, input, 1, true);
        } break;
        case 47: {
            std::cout << "Running rocalFlipFixed" << std::endl;
            output = rocalFlipFixed(handle, input, 1, 0, true);
        } break;
        case 48: {
            std::cout << "Running rocalHueFixed" << std::endl;
            output = rocalHueFixed(handle, input, 150, true);
        } break;
        case 49: {
            std::cout << "Running rocalSaturationFixed" << std::endl;
            output = rocalSaturationFixed(handle, input, 0.3, true);
        } break;
        case 50: {
            std::cout << "Running rocalColorTwistFixed" << std::endl;
            output = rocalColorTwistFixed(handle, input, 0.2, 10.0, 100.0, 0.25, true);
        } break;
        case 51: {
            std::cout << "Running rocalCropFixed" << std::endl;
            output = rocalCropFixed(handle, input, 224, 224, 1, true, 0, 0, 0);
        } break;
        case 52: {
            std::cout << "Running rocalCropCenterFixed" << std::endl;
            output = rocalCropCenterFixed(handle, input, 224, 224, 2, true);
        } break;
        case 53: {
            std::cout << "Running rocalResizeCropMirrorFixed" << std::endl;
            output = rocalResizeCropMirrorFixed(handle, input, 400, 400, true, 200, 200, mirror);
        } break;
        case 54: {
            std::cout << "Running rocalSSDRandomCrop" << std::endl;
            output = rocalSSDRandomCrop(handle, input, true);
        } break;
        case 55: {
            std::cout << "Running rocalCropMirrorNormalizeFixed_center crop" << std::endl;
            std::vector<float> mean = {128, 128, 128};
            std::vector<float> std_dev = {1.2, 1.2, 1.2};
            output = rocalCropMirrorNormalize(handle, input, 224, 224, 0.5, 0.5, mean, std_dev, true, mirror);
        } break;
        case 56: {
            std::vector<float> mean = {128, 128, 128};
            std::vector<float> std_dev = {1.2, 1.2, 1.2};
            std::cout << "Running  Resize Mirror Normalize " << std::endl;
            output = rocalResizeMirrorNormalize(handle, input, 400, 400, mean, std_dev, true, ROCAL_SCALING_MODE_DEFAULT,
                                                {}, 0, 0, ROCAL_LINEAR_INTERPOLATION, mirror);
        } break;
        case 57: {
            std::vector<float> mean = {128, 128, 128};
            std::vector<float> std_dev = {1.2, 1.2, 1.2};
            std::vector<unsigned int> axes = {0, 1};
            std::cout << "Running Normalize" << std::endl;
            output = rocalNormalize(handle, input, axes, mean, std_dev, true, 1.0f, 0.0f, ROCAL_UINT8);
        } break;
        case 58: {
            std::cout << "Running Transpose" << std::endl;
            // Transpose permutation needs to be changed according to input layout
            std::vector<unsigned> perm = rgb ? std::vector<unsigned>{1, 0, 2} : std::vector<unsigned>{0, 2, 1};
            output = rocalTranspose(handle, input, perm, true, output_tensor_layout);
        } break;
        case 59: {
            std::cout << "Running rocalRandomCrop" << std::endl;
            output = rocalRandomCrop(handle, input, true);
        } break;
        case 60: {
            std::cout << "Running rocalROIResize" << std::endl;
            output = rocalROIResize(handle, input, 384, 384, true, 416, 416);
        } break;
        case 61: {
            std::cout << "Running rocalNop" << std::endl;
            output = rocalNop(handle, input, true);
        } break;
        case 62: {
            std::cout << "Running rocalLog1p" << std::endl;
            output = rocalLog1p(handle, input, true);
        } break;
        case 63: {
            std::cout << "Running rocalRandomResizedCrop" << std::endl;
            std::vector<float> area_factor = {0.08, 1};
            std::vector<float> aspect_ratio = {3.0f / 4, 4.0f / 3};
            output = rocalRandomResizedCrop(handle, input, resize_w, resize_h, true, area_factor, aspect_ratio);
        } break;
        case 64: {
            std::cout << "Running rocalGaussianNoise" << std::endl;
            output = rocalGaussianNoise(handle, input, true);
        } break;
        case 65: {
            std::cout << "Running rocalGaussianNoiseFixed" << std::endl;
            output = rocalGaussianNoiseFixed(handle, input, true, 0.0f, 0.2f, 1255459);
        } break;
        case 66: {
            std::cout << "Running rocalShotNoise" << std::endl;
            output = rocalShotNoise(handle, input, true);
        } break;
        case 67: {
            std::cout << "Running rocalShotNoiseFixed" << std::endl;
            output = rocalShotNoiseFixed(handle, input, 80.0f, true, 1255459);
        } break;
        case 68: {
            std::cout << "Running rocalSpatter" << std::endl;
            output = rocalSpatter(handle, input, true);
        } break;
        case 69: {
            std::cout << "Running rocalSpatterFixed" << std::endl;
            output = rocalSpatterFixed(handle, input, 65, 50, 23, true);
        } break;
        case 70: {
            std::cout << "Running rocalLog" << std::endl;
            output = rocalLog(handle, input, true);
        } break;
        case 71: {
            std::cout << "Running rocalColorJitter" << std::endl;
            output = rocalColorJitter(handle, input, true);
        } break;
        case 72: {
            std::cout << "Running rocalColorJitterFixed" << std::endl;
            output = rocalColorJitterFixed(handle, input, 1.02f, 1.1f, 0.02f, 1.3f, true);
        } break;
        case 73: {
            std::cout << "Running rocalWater" << std::endl;
            output = rocalWater(handle, input, true);
        } break;
        case 74: {
            std::cout << "Running rocalWaterFixed" << std::endl;
            output = rocalWaterFixed(handle, input, 2.0f, 5.0f, 5.8f, 1.2f, 10.0f, 15.0f, true);
        } break;
        case 75: {
            std::cout << "Running rocalChannelPermute" << std::endl;
            std::vector<unsigned> permutation_order = {2, 1, 0};  // RGB to BGR
            output = rocalChannelPermute(handle, input, permutation_order, true);
        } break;
        case 76: {
            std::cout << "Running rocalJpegCompressionDistortion" << std::endl;
            output = rocalJpegCompressionDistortion(handle, input, true);
        } break;
        case 77: {
            std::cout << "Running rocalJpegCompressionDistortionFixed" << std::endl;
            output = rocalJpegCompressionDistortionFixed(handle, input, 50, true);
        } break;
        case 78: {
            std::cout << "Running rocalLUT" << std::endl;
            output = rocalLUT(handle, input, true);
        } break;
        case 79: {
            std::cout << "Running rocalPosterize" << std::endl;
            output = rocalPosterize(handle, input, true);
        } break;
        case 80: {
            std::cout << "Running rocalPosterizeFixed" << std::endl;
            output = rocalPosterizeFixed(handle, input, 3, true);
        } break;
        case 81: {
            std::cout << "Running rocalSolarize" << std::endl;
            output = rocalSolarize(handle, input, true);
        } break;
        case 82: {
            std::cout << "Running rocalSolarizeFixed" << std::endl;
            output = rocalSolarizeFixed(handle, input, 0.5f, true);
        } break;
        case 83: {
            std::cout << "Running rocalColorToGreyscale" << std::endl;
            output = rocalColorToGreyscale(handle, input, true);
        } break;
        case 84: {
            std::cout << "Running tensor reduction augmentations" << std::endl;
            auto tensor_sum = rocalTensorSum(handle, input, false, ROCAL_NONE, ROCAL_FP32);
            auto tensor_min = rocalTensorMin(handle, input, false, ROCAL_NONE, ROCAL_UINT8);
            auto tensor_max = rocalTensorMax(handle, input, false, ROCAL_NONE, ROCAL_UINT8);
            auto tensor_mean = rocalTensorMean(handle, input, false, ROCAL_NONE, ROCAL_FP32);
            auto tensor_stddev = rocalTensorStdDev(handle, input, tensor_mean, false, ROCAL_NONE, ROCAL_FP32);
            output = rocalCopy(handle, input, true);
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

    if (number_of_outputs != 1) {
        std::cout << "More than 1 output set in the pipeline";
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle) * input_batch_size;
    int w = rocalGetOutputWidth(handle);
    int output_color_format = rocalGetOutputColorFormat(handle);
    auto last_batch_padded_size = rocalGetLastBatchPaddedSize(handle);
    // Use output_color_format to determine channels: 0=RGB24(3ch), 1=BGR24(3ch), 2=U8(1ch), 3=RGB_PLANAR(3ch)
    int p = ((output_color_format == 0 || output_color_format == 1 || output_color_format == 3) ? 3 : 1);
    const unsigned number_of_cols = 1;  // 1920 / w;
    auto cv_color_format = ((output_color_format == 0 || output_color_format == 1 || output_color_format == 3) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_output(h, w, cv_color_format);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color;
    int col_counter = 0;
    if (DISPLAY)
        cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    printf("Remaining images %lu \n", rocalGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;

    while (rocalGetRemainingImages(handle) >= input_batch_size) {
        index++;
        if (rocalRun(handle) != 0) {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            rocalRelease(handle);
            return -1;
        }
        int image_name_length[input_batch_size];
        switch (pipeline_type) {
            case 1: {   // classification pipeline
                RocalTensorList labels = rocalGetImageLabels(handle);
                int *label_id = reinterpret_cast<int *>(labels->at(0)->buffer());  // The labels are present contiguously in memory
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                std::vector<char> img_name(img_size);
                std::vector<int> label_one_hot_encoded(input_batch_size * num_of_classes);
                rocalGetImageName(handle, img_name.data());
                if (num_of_classes != 0) {
                    rocalGetOneHotImageLabels(handle, label_one_hot_encoded.data(), num_of_classes, RocalOutputMemType::ROCAL_MEMCPY_HOST);
                }
                std::cerr << "\nImage name:" << img_name.data() << "\n";
                for (unsigned int i = 0; i < input_batch_size; i++) {
                    std::cerr << "Label id: " << label_id[i] << std::endl;
                    if(num_of_classes != 0)
                    {
                        std::cout << "One Hot Encoded labels:"<<"\t";
                        for (int j = 0; j < num_of_classes; j++)
                        {
                            int idx_value = label_one_hot_encoded[(i*num_of_classes)+j];
                            if(idx_value == 0)
                                std::cout << idx_value << "\t";
                            else
                            {
                                std::cout << idx_value << "\t";
                            }
                        }
                    }
                    std::cout << "\n";
                }
            } break;
            case 2: {   // detection pipeline
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                std::vector<char> img_name(img_size);
                rocalGetImageName(handle, img_name.data());
                std::cerr << "\nImage name:" << img_name.data();
                RocalTensorList bbox_labels = rocalGetBoundingBoxLabel(handle);
                RocalTensorList bbox_coords = rocalGetBoundingBoxCords(handle);
                for (unsigned i = 0; i < bbox_labels->size(); i++) {
                    int *labels_buffer = reinterpret_cast<int *>(bbox_labels->at(i)->buffer());
                    float *bbox_buffer = reinterpret_cast<float *>(bbox_coords->at(i)->buffer());
                    std::cerr << "\nBBOX Labels : ";
                    for (unsigned j = 0; j < bbox_labels->at(i)->dims().at(0); j++)
                        std::cerr << labels_buffer[j] << " ";
                    std::cerr << "\nBBOX Count: " << bbox_coords->at(i)->dims().at(0) << "\n";
                    for (unsigned j = 0, j4 = 0; j < bbox_coords->at(i)->dims().at(0); j++, j4 = j * 4)
                        std::cerr << bbox_buffer[j4] << " " << bbox_buffer[j4 + 1] << " " << bbox_buffer[j4 + 2] << " " << bbox_buffer[j4 + 3] << "\n";
                }
                int img_sizes_batch[input_batch_size * 2];
                rocalGetImageSizes(handle, img_sizes_batch);
                for (int i = 0; i < (int)input_batch_size; i++) {
                    std::cout << "\nwidth:" << img_sizes_batch[i * 2];
                    std::cout << "\nHeight:" << img_sizes_batch[(i * 2) + 1];
                }

                // Get output matched indices
                if (enable_iou_matcher) {
                    rocalGetMatchedIndices(handle); // TODO - To verify the output
                }
            } break;
            case 3: {   // keypoints pipeline
                int size = input_batch_size;
                RocalJointsData *joints_data;
                rocalGetJointsDataPtr(handle, &joints_data);
                for (int i = 0; i < size; i++) {
                    std::cout << "ImageID: " << joints_data->image_id_batch[i] << std::endl;
                    std::cout << "AnnotationID: " << joints_data->annotation_id_batch[i] << std::endl;
                    std::cout << "ImagePath: " << joints_data->image_path_batch[i] << std::endl;
                    std::cout << "Center: " << joints_data->center_batch[i][0] << " " << joints_data->center_batch[i][1] << std::endl;
                    std::cout << "Scale: " << joints_data->scale_batch[i][0] << " " << joints_data->scale_batch[i][1] << std::endl;
                    std::cout << "Score: " << joints_data->score_batch[i] << std::endl;
                    std::cout << "Rotation: " << joints_data->rotation_batch[i] << std::endl;

                    for (int k = 0; k < 17; k++) {
                        std::cout << "x : " << joints_data->joints_batch[i][k][0] << " , y : " << joints_data->joints_batch[i][k][1] << " , v : " << joints_data->joints_visibility_batch[i][k][0] << std::endl;
                    }
                }
            } break;
            case 4: {   // webdataset pipeline
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                std::vector<char> img_name(img_size);
                rocalGetImageName(handle, img_name.data());
                std::cout << "\n Image name: " << img_name.data() << "\n \n";
                RocalMetaData ascii_sample_contents = rocalGetAsciiDatas(handle);
                std::vector<std::vector<std::vector<uint8_t>>> ext_componenet_list;
                for(uint ext = 0; ext < ascii_sample_contents->size(); ext++) {
                    RocalTensorList ext_ascii_values_batch = ascii_sample_contents->at(ext);
                    std::vector<std::vector<uint8_t>> component_list;
                    std::vector<uint8_t> ascii_components_array;
                    for (uint i = 0; i < ext_ascii_values_batch->size(); i++) {
                        if (ext_ascii_values_batch->at(i)->buffer() !=  nullptr) {
                            uint8_t* buffer = reinterpret_cast<uint8_t*>(ext_ascii_values_batch->at(i)->buffer());
                            size_t length = ext_ascii_values_batch->at(i)->dims().at(0);
                            ascii_components_array.assign(buffer, buffer + length);
                        } else {
                            ascii_components_array = std::vector<uint8_t>{};
                        }
                        component_list.push_back(ascii_components_array);
                    }
                    ext_componenet_list.push_back(component_list);
                }
                for (size_t i = 0; i < ext_componenet_list.size(); ++i) {
                    std::cout << " Meta Data Component " << i + 1 << ":" << std::endl;
                    for (size_t j = 0; j < ext_componenet_list[i].size(); ++j) {
                        std::cout << "  Value " << j + 1 << ": ";
                        for (const auto& value : ext_componenet_list[i][j]) {
                            std::cout << static_cast<uint8_t>(value) << " ";
                        }
                        std::cout << std::endl;
                    }
                }
            } break;
            case 5: {  // numpy reader pipeline
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                std::vector<char> img_name(img_size);
                rocalGetImageName(handle, img_name.data());
                std::cerr << "\nNumpy array name:" << img_name.data() << "\n";
            } break;
            case 6: {   // segmentation pipeline
                std::vector<int> img_id_batch(input_batch_size);
                rocalGetImageId(handle, img_id_batch.data());
                RocalTensorList bbox_labels = rocalGetBoundingBoxLabel(handle);
                RocalTensorList bbox_coords = rocalGetBoundingBoxCords(handle);
                std::vector<int> img_sizes(input_batch_size * 2);
                rocalGetImageSizes(handle, img_sizes.data());
                std::vector<int> roi_img_sizes(input_batch_size * 2);
                rocalGetROIImageSizes(handle, roi_img_sizes.data());
                for (unsigned i = 0; i < input_batch_size; i++) {
                    std::cout << "\nImage ID:" << img_id_batch[i];
                    std::cout << "\timg:" << img_sizes[i * 2] << "x" << img_sizes[(i * 2) + 1];
                    std::cout << "\troi:" << roi_img_sizes[i * 2] << "x" << roi_img_sizes[(i * 2) + 1];
                }

                if (use_pixelwise_masks) {
                    RocalTensorList pixelwise_masks = rocalGetPixelwiseMaskLabels(handle);
                    RocalTensorList random_mask_pixels = rocalRandomMaskPixel(handle);
                    RocalTensorList random_object_bboxes = RocalRandomObjectBBox(handle, ROCAL_OUT_BOX, -1, 1.0f, false);

                    if (!pixelwise_masks || !random_mask_pixels || !random_object_bboxes) {
                        std::cerr << "\nSegmentation metadata API returned null";
                        return -1;
                    }

                    if (pixelwise_masks->size() != bbox_labels->size() ||
                        random_mask_pixels->size() != bbox_labels->size() ||
                        random_object_bboxes->size() != bbox_labels->size()) {
                        std::cerr << "\nSegmentation metadata batch size mismatch";
                        return -1;
                    }

                    std::cout << "\nPixelwise labels (summary)";
                    for (int i = 0; i < bbox_labels->size(); i++) {
                        int *mask_buffer = static_cast<int *>(pixelwise_masks->at(i)->buffer());
                        auto mask_w = pixelwise_masks->at(i)->dims().at(0);
                        auto mask_h = pixelwise_masks->at(i)->dims().at(1);

                        if (!mask_buffer || mask_w == 0 || mask_h == 0) {
                            std::cerr << "\nInvalid pixelwise mask tensor for image " << i;
                            return -1;
                        }

                        if ((int)mask_w != img_sizes[i * 2] || (int)mask_h != img_sizes[i * 2 + 1]) {
                            std::cerr << "\nPixelwise mask dims (" << mask_w << "x" << mask_h
                                      << ") do not match image dims (" << img_sizes[i * 2] << "x" << img_sizes[i * 2 + 1] << ")";
                            return -1;
                        }

                        size_t total = mask_w * mask_h;
                        size_t nonzero = 0;
                        int max_label = 0;
                        for (size_t j = 0; j < total; j++) {
                            int v = mask_buffer[j];
                            if (v > 0) nonzero++;
                            if (v > max_label) max_label = v;
                        }

                        std::cout << "\nImage " << i << " (" << mask_w << "x" << mask_h << "): nonzero=" << nonzero << " max_label=" << max_label;

                        int *coords = static_cast<int *>(random_mask_pixels->at(i)->buffer());
                        if (!coords) {
                            std::cerr << "\nInvalid random_mask_pixel output for image " << i;
                            return -1;
                        }
                        int row = coords[0];
                        int col = coords[1];
                        if (row < 0 || col < 0 || row >= (int)mask_h || col >= (int)mask_w) {
                            std::cerr << "\nrandom_mask_pixel out of bounds for image " << i << " -> (" << row << "," << col
                                      << ") for mask (" << mask_w << "x" << mask_h << ")";
                            return -1;
                        }

                        if (nonzero > 0) {
                            int picked = mask_buffer[(size_t)row * (size_t)mask_w + (size_t)col];
                            std::cout << "\nImage " << i << " random_mask_pixel -> (row=" << row << ", col=" << col << "), label=" << picked;
                            if (picked <= 0) {
                                std::cerr << "\nrandom_mask_pixel did not select foreground for image " << i << " (picked=" << picked << ")";
                                return -1;
                            }
                        } else {
                            std::cout << "\nImage " << i << " random_mask_pixel -> (row=" << row << ", col=" << col << "), label=0 (no foreground in mask)";
                        }

                        unsigned int *bbox = static_cast<unsigned int *>(random_object_bboxes->at(i)->buffer());
                        if (!bbox) {
                            std::cerr << "\nInvalid random_object_bbox output for image " << i;
                            return -1;
                        }
                        unsigned y0 = bbox[0], x0 = bbox[1], y1 = bbox[2], x1 = bbox[3];  // OUT_BOX returns (y0, x0, y1, x1)
                        std::cout << "\nImage " << i << " random_object_bbox -> (y0=" << y0 << ", x0=" << x0 << ", y1=" << y1 << ", x1=" << x1 << ")";
                        if (y0 >= y1 || x0 >= x1 || y1 > (unsigned)mask_h || x1 > (unsigned)mask_w) {
                            std::cerr << "\nrandom_object_bbox out of bounds/degenerate for image " << i
                                      << " -> (" << y0 << "," << x0 << "," << y1 << "," << x1 << ") for mask (" << mask_w << "x" << mask_h << ")";
                            return -1;
                        }

                        if (nonzero > 0) {
                            bool found_fg = false;
                            for (unsigned yy = y0; yy < y1 && !found_fg; yy++) {
                                size_t base = (size_t)yy * (size_t)mask_w;
                                for (unsigned xx = x0; xx < x1; xx++) {
                                    if (mask_buffer[base + xx] > 0) {
                                        found_fg = true;
                                        break;
                                    }
                                }
                            }
                            if (!found_fg) {
                                std::cerr << "\nrandom_object_bbox does not contain foreground for image " << i;
                                return -1;
                            }
                        }
                    }
                } else {
                    int bbox_total = rocalGetBoundingBoxCount(handle);
                    std::vector<int> mask_count(bbox_total);
                    int mask_total = rocalGetMaskCount(handle, mask_count.data());
                    std::vector<int> polygon_size(mask_total);
                    RocalTensorList polygon_masks = rocalGetMaskCoordinates(handle, polygon_size.data());

                    std::cout << "\nPolygon mask metadata";
                    for (int i = 0; i < bbox_total; i++)
                        std::cout << "\n Number of polygons per object: " << mask_count[i];
                    std::cout << "\nMask Size:: " << mask_total;
                    for (int i = 0; i < mask_total; i++)
                        std::cout << "\nPolygon size : " << polygon_size[i];
                    int poly_cnt = 0;
                    int prev_object_cnt = 0;
                    for (int i = 0; i < bbox_labels->size(); i++) {
                        float *mask_buffer = static_cast<float *>(polygon_masks->at(i)->buffer());
                        for (unsigned j = prev_object_cnt; j < bbox_labels->at(i)->dims().at(0) + prev_object_cnt; j++) {
                            for (int k = 0; k < mask_count[j]; k++) {
                                std::cout << "\nPolygon Values (Image " << i << ", Object " << j - prev_object_cnt << "): ";
                                for (int l = 0; l < polygon_size[poly_cnt]; l++)
                                    std::cout << mask_buffer[l] << " ";
                                mask_buffer += polygon_size[poly_cnt++];
                            }
                        }
                        prev_object_cnt += bbox_labels->at(i)->dims().at(0);
                    }
                }

                for (int i = 0; i < bbox_labels->size(); i++) {
                    int *labels_buffer = static_cast<int *>(bbox_labels->at(i)->buffer());
                    float *bbox_buffer = static_cast<float *>(bbox_coords->at(i)->buffer());
                    std::cout << "\nBBOX labels for image " << i << ": ";
                    for (unsigned j = 0; j < bbox_labels->at(i)->dims().at(0); j++)
                        std::cout << labels_buffer[j] << " ";
                    std::cout << "\nBBOX coords:";
                    for (unsigned j = 0, idx = 0; j < bbox_coords->at(i)->dims().at(0); j++, idx = j * 4)
                        std::cout << " (" << bbox_buffer[idx] << ", " << bbox_buffer[idx + 1] << ", " << bbox_buffer[idx + 2] << ", " << bbox_buffer[idx + 3] << ")";
                }
            } break;
            case 7: // Box encoder
            {
                int img_size = rocalGetImageNameLen(handle, image_name_length);
                std::vector<char> img_name(img_size);
                rocalGetImageName(handle, img_name.data());
                std::cerr << "\nImage name:" << img_name.data();
                auto num_anchors = 8732;
                rocalGetEncodedBoxesAndLables(handle, input_batch_size * num_anchors);
                float *boxes_buffer = new float[input_batch_size * num_anchors * 4];
                int *labels_buffer = new int[input_batch_size * num_anchors];
                rocalCopyEncodedBoxesAndLables(handle, boxes_buffer, labels_buffer);
                int img_sizes_batch[input_batch_size * 2];
                rocalGetImageSizes(handle, img_sizes_batch);
                for (int i = 0; i < (int)input_batch_size; i++) {
                    std::cout << "\nwidth:" << img_sizes_batch[i * 2];
                    std::cout << "\nHeight:" << img_sizes_batch[(i * 2) + 1];
                }
                delete[] boxes_buffer;
                delete[] labels_buffer;
            } break;
            default: {
                std::cout << "Not a valid pipeline type ! Exiting!\n";
                return -1;
            }
        }
        auto last_colot_temp = rocalGetIntValue(color_temp_adj);
        rocalUpdateIntParameter(last_colot_temp + 1, color_temp_adj);

        rocalCopyToOutput(handle, mat_input.data, h * w * p);
        
        // Testing the rocalToTensor API for copy augmentation
        // Memory allocated here is freed after used in rocalToTensor API for memcopy
        if (test_case == 23) {
            if (gpu == 0) {
                if (memcpy_backend) {
#if ENABLE_HIP
                    float *d_f32_batch_output;
                    hipMalloc(&d_f32_batch_output, 256 * ((input_batch_size * h * w * p * sizeof(float)) / 256 + 1));
                    rocalToTensor(handle, d_f32_batch_output, (RocalTensorLayout)output_layout, RocalTensorOutputType::ROCAL_FP32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, reverse_channels, RocalOutputMemType::ROCAL_MEMCPY_GPU);
                    hipFree(d_f32_batch_output);

                    half *d_f16_batch_output;
                    hipMalloc(&d_f16_batch_output, 256 * ((input_batch_size * h * w * p * sizeof(half)) / 256 + 1));
                    rocalToTensor(handle, d_f16_batch_output, (RocalTensorLayout)output_layout, RocalTensorOutputType::ROCAL_FP16, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, reverse_channels, RocalOutputMemType::ROCAL_MEMCPY_GPU);
                    hipFree(d_f16_batch_output);
#endif
                } else {
                    float *f32_batch_output = (float *)aligned_alloc(256, 256 * ((input_batch_size * h * w * p * sizeof(float)) / 256 + 1));
                    rocalToTensor(handle, f32_batch_output, (RocalTensorLayout)output_layout, RocalTensorOutputType::ROCAL_FP32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, reverse_channels, RocalOutputMemType::ROCAL_MEMCPY_HOST);
                    free(f32_batch_output);

                    half *f16_batch_output = (half *)aligned_alloc(256, 256 * ((input_batch_size * h * w * p * sizeof(half)) / 256 + 1));
                    rocalToTensor(handle, f16_batch_output, (RocalTensorLayout)output_layout, RocalTensorOutputType::ROCAL_FP16, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, reverse_channels, RocalOutputMemType::ROCAL_MEMCPY_HOST);
                    free(f16_batch_output);
                }
            } else {
#if ENABLE_HIP
                float *d_f32_batch_output;
                hipMalloc(&d_f32_batch_output, 256 * ((input_batch_size * h * w * p * sizeof(float)) / 256 + 1));
                rocalToTensor(handle, d_f32_batch_output, (RocalTensorLayout)output_layout, RocalTensorOutputType::ROCAL_FP32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, reverse_channels, RocalOutputMemType::ROCAL_MEMCPY_GPU);
                hipFree(d_f32_batch_output);

                half *d_f16_batch_output;
                hipMalloc(&d_f16_batch_output, 256 * ((input_batch_size * h * w * p * sizeof(half)) / 256 + 1));
                rocalToTensor(handle, d_f16_batch_output, (RocalTensorLayout)output_layout, RocalTensorOutputType::ROCAL_FP16, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, reverse_channels, RocalOutputMemType::ROCAL_MEMCPY_GPU);
                hipFree(d_f16_batch_output);
#endif
            }
        }

        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        mat_input.copyTo(mat_output(cv::Rect(col_counter * w, 0, w, h)));
        std::string out_filename = std::string(outName) + ".png";  // in case the user specifies non png filename
        if (display_all)
            out_filename = std::string(outName) + std::to_string(index) + ".png";  // in case the user specifies non png filename

        if (output_color_format == 0) {  // RGB24
            cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
            if (DISPLAY)
                cv::imshow("output", mat_output);
            else
                cv::imwrite(out_filename, mat_color, compression_params);
        } else if (output_color_format == 1) {  // BGR24
            if (DISPLAY)
                cv::imshow("output", mat_output);
            else
                cv::imwrite(out_filename, mat_output, compression_params);
        } else {
            if (DISPLAY)
                cv::imshow("output", mat_output);
            else
                cv::imwrite(out_filename, mat_output, compression_params);
        }
        col_counter = (col_counter + 1) % number_of_cols;
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
    mat_input.release();
    mat_output.release();
    if (!output)
        return -1;
    return 0;
}
