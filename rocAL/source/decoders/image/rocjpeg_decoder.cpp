/*
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


#include "pipeline/commons.h"
#include <stdio.h>
#include <string.h>
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "decoders/image/rocjpeg_decoder.h"

#if ENABLE_HIP

#define CHECK_HIP(call) {                                             \
    hipError_t hip_status = (call);                                   \
    if (hip_status != hipSuccess) {                                   \
        std::cerr << "HIP failure: 'status: " << hipGetErrorName(hip_status) << "' at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                      \
    }                                                                 \
}

#define CHECK_ROCJPEG(call) {                                             \
    RocJpegStatus rocjpeg_status = (call);                                \
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {                       \
        std::cerr << #call << " returned " << rocJpegGetErrorName(rocjpeg_status) << " at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                        \
    }                                                                     \
}

RocJpegDecoder::RocJpegDecoder() {
};

/**
 * @brief Gets the channel pitch and sizes.
 *
 * This function gets the channel pitch and sizes based on the specified output format, chroma subsampling,
 * output image, and channel sizes.
 *
 * @param output_format The output format.
 * @param subsampling The chroma subsampling.
 * @param widths The array to store the channel widths.
 * @param heights The array to store the channel heights.
 * @param num_channels The number of channels.
 * @param output_image The output image.
 * @param channel_sizes The array to store the channel sizes.
 * @return The channel pitch.
 */
int GetChannelPitchAndSizes(RocJpegOutputFormat output_format, RocJpegChromaSubsampling subsampling, uint32_t *widths, uint32_t *heights,
                            uint32_t &num_channels, RocJpegImage &output_image, uint32_t *channel_sizes) {
    switch (output_format) {
        case ROCJPEG_OUTPUT_NATIVE:
            switch (subsampling) {
                case ROCJPEG_CSS_444:
                    num_channels = 3;
                    output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = widths[0];
                    channel_sizes[2] = channel_sizes[1] = channel_sizes[0] = output_image.pitch[0] * heights[0];
                    break;
                case ROCJPEG_CSS_422:
                    num_channels = 1;
                    output_image.pitch[0] = widths[0] * 2;
                    channel_sizes[0] = output_image.pitch[0] * heights[0];
                    break;
                case ROCJPEG_CSS_420:
                    num_channels = 2;
                    output_image.pitch[1] = output_image.pitch[0] = widths[0];
                    channel_sizes[0] = output_image.pitch[0] * heights[0];
                    channel_sizes[1] = output_image.pitch[1] * (heights[0] >> 1);
                    break;
                case ROCJPEG_CSS_400:
                    num_channels = 1;
                    output_image.pitch[0] = widths[0];
                    channel_sizes[0] = output_image.pitch[0] * heights[0];
                    break;
                default:
                    std::cout << "Unknown chroma subsampling!" << std::endl;
                    return EXIT_FAILURE;
            }
            break;
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            if (subsampling == ROCJPEG_CSS_400) {
                num_channels = 1;
                output_image.pitch[0] = widths[0];
                channel_sizes[0] = output_image.pitch[0] * heights[0];
            } else {
                num_channels = 3;
                output_image.pitch[0] = widths[0];
                output_image.pitch[1] = widths[1];
                output_image.pitch[2] = widths[2];
                channel_sizes[0] = output_image.pitch[0] * heights[0];
                channel_sizes[1] = output_image.pitch[1] * heights[1];
                channel_sizes[2] = output_image.pitch[2] * heights[2];
            }
            break;
        case ROCJPEG_OUTPUT_Y:
            num_channels = 1;
            output_image.pitch[0] = widths[0];
            channel_sizes[0] = output_image.pitch[0] * heights[0];
            break;
        case ROCJPEG_OUTPUT_RGB:
            num_channels = 1;
            output_image.pitch[0] = widths[0] * 3;
            channel_sizes[0] = output_image.pitch[0] * heights[0];
            break;
        case ROCJPEG_OUTPUT_RGB_PLANAR:
            num_channels = 3;
            output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = widths[0];
            channel_sizes[2] = channel_sizes[1] = channel_sizes[0] = output_image.pitch[0] * heights[0];
            break;
        default:
            std::cout << "Unknown output format!" << std::endl;
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

/**
 * @brief Gets the chroma subsampling string.
 *
 * This function gets the chroma subsampling string based on the specified subsampling value.
 *
 * @param subsampling The chroma subsampling value.
 * @param chroma_sub_sampling The string to store the chroma subsampling.
 */
void GetChromaSubsamplingStr(RocJpegChromaSubsampling subsampling, std::string &chroma_sub_sampling) {
    switch (subsampling) {
        case ROCJPEG_CSS_444:
            chroma_sub_sampling = "YUV 4:4:4";
            break;
        case ROCJPEG_CSS_440:
            chroma_sub_sampling = "YUV 4:4:0";
            break;
        case ROCJPEG_CSS_422:
            chroma_sub_sampling = "YUV 4:2:2";
            break;
        case ROCJPEG_CSS_420:
            chroma_sub_sampling = "YUV 4:2:0";
            break;
        case ROCJPEG_CSS_411:
            chroma_sub_sampling = "YUV 4:1:1";
            break;
        case ROCJPEG_CSS_400:
            chroma_sub_sampling = "YUV 4:0:0";
            break;
        case ROCJPEG_CSS_UNKNOWN:
            chroma_sub_sampling = "UNKNOWN";
            break;
        default:
            chroma_sub_sampling = "";
            break;
    }
}

void RocJpegDecoder::initialize(int device_id) {
    int num_devices;
    hipDeviceProp_t hip_dev_prop;
    CHECK_HIP(hipGetDeviceCount(&num_devices));
    if (num_devices < 1) {
        std::cerr << "ERROR: didn't find any GPU!" << std::endl;
        return;
    }
    if (device_id >= num_devices) {
        std::cerr << "ERROR: the requested device_id is not found!" << std::endl;
        return;
    }
    CHECK_HIP(hipSetDevice(device_id));
    CHECK_HIP(hipGetDeviceProperties(&hip_dev_prop, device_id));

    std::cout << "Using GPU device " << device_id << ": " << hip_dev_prop.name << "[" << hip_dev_prop.gcnArchName << "] on PCI bus " <<
    std::setfill('0') << std::setw(2) << std::right << std::hex << hip_dev_prop.pciBusID << ":" << std::setfill('0') << std::setw(2) <<
    std::right << std::hex << hip_dev_prop.pciDomainID << "." << hip_dev_prop.pciDeviceID << std::dec << std::endl;

    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    // Create stream and handle
    CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &_rocjpeg_handle));
    CHECK_ROCJPEG(rocJpegStreamCreate(&_rocjpeg_stream));
}

Decoder::Status RocJpegDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *color_comps) {
    RocJpegChromaSubsampling subsampling;
    uint8_t num_components;
    uint32_t widths[4] = {};
    uint32_t heights[4] = {};

    CHECK_ROCJPEG(rocJpegStreamParse(reinterpret_cast<uint8_t*>(input_buffer), input_size, _rocjpeg_stream));
    std::cerr << "Stream parse is done.............\n";
    CHECK_ROCJPEG(rocJpegGetImageInfo(_rocjpeg_handle, _rocjpeg_stream, &num_components, &subsampling, widths, heights));
    std::cerr << "Get Image Info ----------.............\n";
    *width = widths[0];
    *height = heights[0];
    std::cerr << "WIDTH : " << widths[0] << " HEIGHT : " << heights[0] << "\n";
    return Status::OK;
}

Decoder::Status RocJpegDecoder::decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                           size_t max_decoded_width, size_t max_decoded_height,
                                           size_t original_image_width, size_t original_image_height,
                                           size_t &actual_decoded_width, size_t &actual_decoded_height,
                                           Decoder::ColorFormat desired_decoded_color_format, DecoderConfig decoder_config, bool keep_original_size) {

    RocJpegChromaSubsampling subsampling;
    uint8_t num_components;
    uint32_t num_channels = 0;
    uint32_t widths[4] = {};
    uint32_t heights[4] = {};
    RocJpegDecodeParams decode_params = {};
    RocJpegImage output_image = {};
    uint32_t channel_sizes[4] = {};

    switch(desired_decoded_color_format) {
        case Decoder::ColorFormat::GRAY:
            decode_params.output_format = ROCJPEG_OUTPUT_Y; // TODO - Need to check the correct color format
            break;
        case Decoder::ColorFormat::RGB:
        case Decoder::ColorFormat::BGR:
            decode_params.output_format = ROCJPEG_OUTPUT_RGB;
            break;
    };

    // if (selected_output_format == "native") {
    //     decode_params.output_format = ROCJPEG_OUTPUT_NATIVE;
    // } else if (selected_output_format == "yuv") {
    //     decode_params.output_format = ROCJPEG_OUTPUT_YUV_PLANAR;
    // } else if (selected_output_format == "y") {
    //     decode_params.output_format = ROCJPEG_OUTPUT_Y;
    // } else if (selected_output_format == "rgb") {
    //     decode_params.output_format = ROCJPEG_OUTPUT_RGB;
    // } else if (selected_output_format == "rgb_planar") {
    //     decode_params.output_format = ROCJPEG_OUTPUT_RGB_PLANAR;
    // } else {
    //     ShowHelpAndExit(argv[i], num_threads != nullptr);
    // }

    CHECK_ROCJPEG(rocJpegStreamParse(reinterpret_cast<uint8_t*>(input_buffer), input_size, _rocjpeg_stream));
    CHECK_ROCJPEG(rocJpegGetImageInfo(_rocjpeg_handle, _rocjpeg_stream, &num_components, &subsampling, widths, heights));

    std::string chroma_sub_sampling = "";
    GetChromaSubsamplingStr(subsampling, chroma_sub_sampling);
    std::cout << "Input image resolution: " << widths[0] << "x" << heights[0] << std::endl;
    std::cout << "Chroma subsampling: " + chroma_sub_sampling  << std::endl;
    if (subsampling == ROCJPEG_CSS_440 || subsampling == ROCJPEG_CSS_411) {
        std::cerr << "The chroma sub-sampling is not supported by VCN Hardware" << std::endl;
        return Status::UNSUPPORTED;
    }

    if (GetChannelPitchAndSizes(decode_params.output_format, subsampling, widths, heights, num_channels, output_image, channel_sizes)) {
        std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
        return Status::HEADER_DECODE_FAILED;
    }

    // allocate memory for each channel and reuse them if the sizes remain unchanged for a new image.
    // for (int i = 0; i < num_channels; i++) {
    //     std::cerr << "Num channels : " << num_channels << " Channels size : " << channel_sizes[i] << "\n";
    //     output_image.channel[i] = output_buffer;
    //     output_buffer += channel_sizes[i];
    // }

    // allocate memory for each channel
    for (int i = 0; i < num_channels; i++) {
        if (output_image.channel[i] != nullptr) {
            CHECK_HIP(hipFree((void*)output_image.channel[i]));
            output_image.channel[i] = nullptr;
        }
        CHECK_HIP(hipMalloc(&output_image.channel[i], channel_sizes[i]));
    }

    std::cout << "Decoding started, please wait! ... " << std::endl;
    // auto start_time = std::chrono::high_resolution_clock::now();
    CHECK_ROCJPEG(rocJpegDecode(_rocjpeg_handle, _rocjpeg_stream, &decode_params, &output_image));
    // auto end_time = std::chrono::high_resolution_clock::now();
    // double time_per_image_in_milli_sec = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    // double image_size_in_mpixels = (static_cast<double>(widths[0]) * static_cast<double>(heights[0]) / 1000000);
    // image_count++;

    uint32_t channel0_size = output_image.pitch[0] * heights[0];
    uint32_t channel1_size = output_image.pitch[1] * heights[1];
    uint32_t channel2_size = output_image.pitch[2] * heights[2];
    CHECK_HIP(hipMemcpyDtoD((void *)output_buffer, output_image.channel[0], output_image.pitch[0] * heights[0]));
    actual_decoded_width = widths[0];
    actual_decoded_height = heights[0];
    return Status::OK;
}

RocJpegDecoder::~RocJpegDecoder() {
}
#endif
