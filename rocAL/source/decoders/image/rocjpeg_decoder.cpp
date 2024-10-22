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
#include "decoders/image/rocjpeg_decoder.h"

#if ENABLE_HIP
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "rocal_hip_kernels.h"

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

void RocJpegDecoder::initialize_batch(int device_id, unsigned batch_size) {
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
    for (int i = 0; i < batch_size; i++) {
        RocJpegStreamHandle rocjpeg_stream;
        CHECK_ROCJPEG(rocJpegStreamCreate(&rocjpeg_stream));
        _rocjpeg_streams.push_back(rocjpeg_stream);
    }

    // hipError_t err = hipStreamCreate(&_hip_stream);
    // if (err != hipSuccess) {
    //     THROW("init_hip::hipStreamCreate failed " + TOSTR(err))
    // }
    _batch_size = batch_size;

    // Allocate mem for width and height arrays for src and dst
    if (!_dev_src_width) CHECK_HIP(hipMalloc((void **)&_dev_src_width, _batch_size * sizeof(size_t)));
    if (!_dev_src_height) CHECK_HIP(hipMalloc((void **)&_dev_src_height, _batch_size * sizeof(size_t)));
    if (!_dev_dst_width) CHECK_HIP(hipMalloc((void **)&_dev_dst_width, _batch_size * sizeof(size_t)));
    if (!_dev_dst_height) CHECK_HIP(hipMalloc((void **)&_dev_dst_height, _batch_size * sizeof(size_t)));
    if (!_dev_src_hstride) CHECK_HIP(hipMalloc((void **)&_dev_src_hstride, _batch_size * sizeof(size_t)));
    if (!_dev_src_img_offset) CHECK_HIP(hipMalloc((void **)&_dev_src_img_offset, _batch_size * sizeof(size_t)));

}

Decoder::Status RocJpegDecoder::decode_info_batch(std::vector<std::vector<unsigned char>> &input_buffer, std::vector<size_t> &input_size, std::vector<size_t> &width, std::vector<size_t> &height, int *color_comps) {
    RocJpegChromaSubsampling subsampling;
    uint8_t num_components;
    uint32_t widths[4] = {};
    uint32_t heights[4] = {};
    unsigned max_buffer_size = 0;
    for (int i = 0; i < _batch_size; i++) {
        CHECK_ROCJPEG(rocJpegStreamParse(reinterpret_cast<uint8_t*>(input_buffer[i].data()), input_size[i], _rocjpeg_streams[i]));
        CHECK_ROCJPEG(rocJpegGetImageInfo(_rocjpeg_handle, _rocjpeg_streams[i], &num_components, &subsampling, widths, heights));
        width[i] = widths[0];
        height[i] = heights[0];
        max_buffer_size += (((widths[0] + 8) &~ 7) * ((heights[0] + 8) &~ 7));
    }
    _rocjpeg_image_buff_size = max_buffer_size;
    return Status::OK;
}

Decoder::Status RocJpegDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *color_comps) {
    RocJpegChromaSubsampling subsampling;
    uint8_t num_components;
    uint32_t widths[4] = {};
    uint32_t heights[4] = {};
    if (rocJpegStreamParse(reinterpret_cast<uint8_t*>(input_buffer), input_size, _rocjpeg_streams[0]) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    if (rocJpegGetImageInfo(_rocjpeg_handle, _rocjpeg_streams[0], &num_components, &subsampling, widths, heights) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    *width = widths[0];
    *height = heights[0];
    _rocjpeg_image_buff_size += (((widths[0] + 8) &~ 7) * ((heights[0] + 8) &~ 7));

    if (widths[0] < 64 || heights[0] < 64) {
        return Status::CONTENT_DECODE_FAILED;
    }

    return Status::OK;
}

Decoder::Status RocJpegDecoder::decode_batch(std::vector<std::vector<unsigned char>> &input_buffer, std::vector<size_t> &input_size,
                                       std::vector<unsigned char *> &output_buffer,
                                       size_t max_decoded_width, size_t max_decoded_height,
                                       std::vector<size_t> original_image_width, std::vector<size_t> original_image_height,
                                       std::vector<size_t> &actual_decoded_width, std::vector<size_t> &actual_decoded_height,
                                       Decoder::ColorFormat desired_decoded_color_format, DecoderConfig decoder_config, bool keep_original_size) {

    RocJpegChromaSubsampling subsampling;
    uint8_t num_components;
    uint32_t num_channels = 0, channels_size = 0;
    uint32_t widths[4] = {};
    uint32_t heights[4] = {};
    RocJpegDecodeParams decode_params = {};
    std::vector<RocJpegImage> output_images = {};
    output_images.resize(_batch_size);
    uint32_t channel_sizes[4] = {};

    switch(desired_decoded_color_format) {
        case Decoder::ColorFormat::GRAY:
            decode_params.output_format = ROCJPEG_OUTPUT_Y; // TODO - Need to check the correct color format
            num_channels = 1;
            break;
        case Decoder::ColorFormat::RGB:
        case Decoder::ColorFormat::BGR:
            decode_params.output_format = ROCJPEG_OUTPUT_RGB;
            num_channels = 3;
            break;
    };

    // Allocate memory for the itermediate decoded output
    _rocjpeg_image_buff_size *= num_channels;
    if (!_rocjpeg_image_buff) {
        CHECK_HIP(hipMalloc((void **)&_rocjpeg_image_buff, _rocjpeg_image_buff_size));
        _prev_image_buff_size = _rocjpeg_image_buff_size;
    } else if (_rocjpeg_image_buff_size > _prev_image_buff_size) {  // Reallocate if the intermediate output exceeds the allocated memory
        CHECK_HIP(hipFree((void *)_rocjpeg_image_buff));
        CHECK_HIP(hipMalloc((void **)&_rocjpeg_image_buff, _rocjpeg_image_buff_size));
        _prev_image_buff_size = _rocjpeg_image_buff_size;
    }

    uint8_t *img_buff = reinterpret_cast<uint8_t*>(_rocjpeg_image_buff);
    std::vector<size_t> src_hstride(_batch_size);
    std::vector<size_t> src_img_offset(_batch_size);
    size_t src_offset = 0;
    bool resize_batch = false;
    uint32_t max_widths[4] = {max_decoded_width, 0, 0, 0};
    uint32_t max_heights[4] = {max_decoded_height, 0, 0, 0};

    for (int i = 0; i < _batch_size; i++) {
        CHECK_ROCJPEG(rocJpegStreamParse(reinterpret_cast<uint8_t*>(input_buffer[i].data()), input_size[i], _rocjpeg_streams[i]));
        CHECK_ROCJPEG(rocJpegGetImageInfo(_rocjpeg_handle, _rocjpeg_streams[i], &num_components, &subsampling, widths, heights));
        std::string chroma_sub_sampling = "";
        GetChromaSubsamplingStr(subsampling, chroma_sub_sampling);
        // std::cout << "Input image resolution: " << widths[0] << "x" << heights[0] << std::endl;
        // std::cout << "Chroma subsampling: " + chroma_sub_sampling  << std::endl;
        if (subsampling == ROCJPEG_CSS_440 || subsampling == ROCJPEG_CSS_411) {
            std::cerr << "The chroma sub-sampling is not supported by VCN Hardware" << std::endl;
            return Status::UNSUPPORTED;
        }

        // std::cout << "Decoding started, please wait! ... "<<  num_channels<< std::endl;

        uint scaledw = original_image_width[i], scaledh = original_image_height[i];
        if (original_image_width[i] > max_decoded_width || original_image_height[i] > max_decoded_height) {
            for (int j=0; j < _num_scaling_factors; j++) {
                scaledw = (((original_image_width[i]) * _scaling_factors[j].num + _scaling_factors[j].denom - 1) / _scaling_factors[j].denom);
                scaledh = (((original_image_height[i]) * _scaling_factors[j].num + _scaling_factors[j].denom - 1) / _scaling_factors[j].denom);
                if (scaledw <= max_decoded_width && scaledh <= max_decoded_height)
                    break;
            }
        }

        if (scaledw != original_image_width[i] || scaledh != original_image_height[i]) {
            max_widths[0] = (widths[0] + 8) &~ 7;
            max_heights[0] = (heights[0] + 8) &~ 7;
        }
        if (GetChannelPitchAndSizes(decode_params.output_format, subsampling, max_widths, max_heights, channels_size, output_images[i], channel_sizes)) {
            std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
            return Status::HEADER_DECODE_FAILED;
        }

        if (scaledw != original_image_width[i] || scaledh != original_image_height[i]) {
            resize_batch = true;
            output_images[i].channel[0] = static_cast<uint8_t *>(img_buff);    // For RGB
            src_img_offset[i] = src_offset;
            src_offset += (max_widths[0] * max_heights[0] * num_channels);
            img_buff += (max_widths[0] * max_heights[0] * num_channels);
            src_hstride[i] = max_widths[0] * num_channels;
        } else {
            output_images[i].channel[0] = static_cast<uint8_t *>(output_buffer[i]);    // For RGB
        }

        actual_decoded_width[i] = scaledw;
        actual_decoded_height[i] = scaledh;
    }

    if (resize_batch) {
        // Copy width and height args to HIP memory
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_src_width, original_image_width.data(), _batch_size * sizeof(size_t)));
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_src_height, original_image_height.data(), _batch_size * sizeof(size_t)));
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_dst_width, actual_decoded_width.data(), _batch_size * sizeof(size_t)));
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_dst_height, actual_decoded_height.data(), _batch_size * sizeof(size_t)));
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_src_hstride, src_hstride.data(), _batch_size * sizeof(size_t)));
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_src_img_offset, src_img_offset.data(), _batch_size * sizeof(size_t)));
    }

    CHECK_ROCJPEG(rocJpegDecodeBatched(_rocjpeg_handle, _rocjpeg_streams.data(), _batch_size, &decode_params, output_images.data()));

    if (resize_batch) {
        HipExecResizeTensor(_hip_stream, (void *)_rocjpeg_image_buff, (void *)output_buffer[0], 
                            _batch_size, _dev_src_width, _dev_src_height, 
                        _dev_dst_width, _dev_dst_height, _dev_src_hstride, _dev_src_img_offset, num_channels,
                            max_decoded_width, max_decoded_height, max_decoded_width, max_decoded_height);
    }

    return Status::OK;
}

RocJpegDecoder::~RocJpegDecoder() {
    if (_dev_src_width) CHECK_HIP(hipFree(_dev_src_width));
    if (_dev_src_height) CHECK_HIP(hipFree(_dev_src_height));
    if (_dev_dst_width) CHECK_HIP(hipFree(_dev_dst_width));
    if (_dev_dst_height) CHECK_HIP(hipFree(_dev_dst_height));
    if (_dev_src_hstride) CHECK_HIP(hipFree(_dev_src_hstride));
    if (_dev_src_img_offset) CHECK_HIP(hipFree(_dev_src_img_offset));
}
#endif
