/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#if ENABLE_ROCJPEG

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

HWRocJpegDecoder::HWRocJpegDecoder() {
};

void HWRocJpegDecoder::initialize(int device_id, unsigned batch_size) {
    int num_devices;
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
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    // Create stream and handle
    CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &_rocjpeg_handle));
    _rocjpeg_streams.resize(batch_size);
    for (unsigned i = 0; i < batch_size; i++) {
        CHECK_ROCJPEG(rocJpegStreamCreate(&_rocjpeg_streams[i]));
    }

    _device_id = device_id;
    _batch_size = batch_size;
    _output_images.resize(_batch_size);
    _decode_params.resize(_batch_size);
    _image_needs_rescaling.resize(_batch_size);

    // Allocate pinned memory for width and height arrays for src and dst.
    if (!_src_width) CHECK_HIP(hipHostMalloc((void **)&_src_width, _batch_size * sizeof(size_t)));
    if (!_src_height) CHECK_HIP(hipHostMalloc((void **)&_src_height, _batch_size * sizeof(size_t)));
    if (!_dst_width) CHECK_HIP(hipHostMalloc((void **)&_dst_width, _batch_size * sizeof(size_t)));
    if (!_dst_height) CHECK_HIP(hipHostMalloc((void **)&_dst_height, _batch_size * sizeof(size_t)));
    if (!_src_hstride) CHECK_HIP(hipHostMalloc((void **)&_src_hstride, _batch_size * sizeof(size_t)));
    if (!_src_img_offset) CHECK_HIP(hipHostMalloc((void **)&_src_img_offset, _batch_size * sizeof(size_t)));
    if (!_dst_img_idx) CHECK_HIP(hipHostMalloc((void **)&_dst_img_idx, _batch_size * sizeof(uint32_t)));

}

// Obtains the decode info of the image, and modifies width and height based on the max decode params after scaling
Decoder::Status HWRocJpegDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *actual_width, 
                                              int *actual_height, int max_decoded_width, int max_decoded_height, Decoder::ColorFormat desired_decoded_color_format, int index) {
    RocJpegChromaSubsampling subsampling;
    uint8_t num_components;
    uint32_t widths[4] = {};
    uint32_t heights[4] = {};
    uint32_t channels_size = 0;
    uint32_t channel_sizes[4] = {};

    uint32_t max_widths[4] = {static_cast<uint32_t>(max_decoded_width), 0, 0, 0};
    uint32_t max_heights[4] = {static_cast<uint32_t>(max_decoded_height), 0, 0, 0};

    switch(desired_decoded_color_format) {
        case Decoder::ColorFormat::GRAY:
            _decode_params[index].output_format = ROCJPEG_OUTPUT_Y;
            _num_channels = 1;
            break;
        case Decoder::ColorFormat::RGB:
        case Decoder::ColorFormat::BGR:
            _decode_params[index].output_format = ROCJPEG_OUTPUT_RGB;
            _num_channels = 3;
            break;
    };

    if (rocJpegStreamParse(reinterpret_cast<uint8_t*>(input_buffer), input_size, _rocjpeg_streams[index]) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    if (rocJpegGetImageInfo(_rocjpeg_handle, _rocjpeg_streams[index], &num_components, &subsampling, widths, heights) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }

    if (widths[0] < 64 || heights[0] < 64) {
        return Status::CONTENT_DECODE_FAILED;
    }

    std::string chroma_sub_sampling = "";
    GetChromaSubsamplingStr(subsampling, chroma_sub_sampling);
    if (subsampling == ROCJPEG_CSS_411 || subsampling == ROCJPEG_CSS_UNKNOWN) {
        return Status::UNSUPPORTED;
    }

    if (width) *width = widths[0];
    if (height) *height = heights[0];
    // Reset per-image flag for this batch (it is used later for intermediate buffer layout).
    _image_needs_rescaling[index] = false;

    uint scaledw = widths[0], scaledh = heights[0];
    // If original dims exceed max decode dims, compute output dims that fit within max while preserving aspect ratio.
    // Pick the scale based on the dimension that would be downscaled the most (largest original/max ratio).
    bool has_max_dims = max_decoded_width > 0 && max_decoded_height > 0;
    bool exceeds_max_dims = (widths[0] > static_cast<uint32_t>(max_decoded_width)) || (heights[0] > static_cast<uint32_t>(max_decoded_height));
    if (has_max_dims && exceeds_max_dims) {
        const uint64_t in_w = widths[0];
        const uint64_t in_h = heights[0];
        const float scale_factor_w = static_cast<float>(in_w) / max_decoded_width;   // How much width exceeds its max
        const float scale_factor_h = static_cast<float>(in_h) / max_decoded_height;  // How much height exceeds its max
    
        if (scale_factor_w >= scale_factor_h) {                                   // Width is the limiting dimension
            scaledw = static_cast<uint32_t>(max_decoded_width);
            scaledh = static_cast<uint32_t>(std::max(1.0f, in_h / scale_factor_w));
        } else {                                                    // Height is the limiting dimension
            scaledh = static_cast<uint32_t>(max_decoded_height);
            scaledw = static_cast<uint32_t>(std::max(1.0f, in_w / scale_factor_h));
        }
    }
    // If scaled width is different than original width and height, update max dims with the original width and height, to be used for decoding
    if (scaledw != widths[0] || scaledh != heights[0]) {
        _enable_resize = true;   // If the size of any image in the batch is greater than max size, resize the complete batch
        max_widths[0] = (widths[0] + 8) &~ 7;
        max_heights[0] = (heights[0] + 8) &~ 7;
        _image_needs_rescaling[index] = true;
    }

    if (GetChannelPitchAndSizes(_decode_params[index], subsampling, max_widths, max_heights, channels_size, _output_images[index], channel_sizes)) {
        return Status::HEADER_DECODE_FAILED;
    }
    if (actual_width) *actual_width = scaledw;
    if (actual_height) *actual_height = scaledh;

    // Only images that are decoded into the intermediate buffer contribute to its size.
    // Non-resized images decode directly into the output tensor to avoid extra copies/branches.
    if (_image_needs_rescaling[index]) {
        _rocjpeg_image_buff_size += max_widths[0] * max_heights[0];
    }

    return Status::OK;
}

Decoder::Status HWRocJpegDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *color_comps) {
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

    std::string chroma_sub_sampling = "";
    GetChromaSubsamplingStr(subsampling, chroma_sub_sampling);
    if (subsampling == ROCJPEG_CSS_411 || subsampling == ROCJPEG_CSS_UNKNOWN) {
        return Status::UNSUPPORTED;
    }
    return Status::OK;
}

Decoder::Status HWRocJpegDecoder::decode_batch(std::vector<unsigned char *> &output_buffer,
                                               size_t max_decoded_width, size_t max_decoded_height,
                                               std::vector<size_t> original_image_width, std::vector<size_t> original_image_height,
                                               std::vector<size_t> &actual_decoded_width, std::vector<size_t> &actual_decoded_height) {
    unsigned resize_count = 0;    // A count of how many images in the batch need resizing, used for launching the resize kernel.
    if (_enable_resize && _rocjpeg_image_buff_size > 0) {
        // Allocate memory for the intermediate decoded output for only the images that need resizing.
        const size_t resize_image_buff_bytes = (size_t)_rocjpeg_image_buff_size * (size_t)_num_channels;
        if (!_rocjpeg_image_buff) {
            CHECK_HIP(hipMalloc((void **)&_rocjpeg_image_buff, resize_image_buff_bytes));
            _prev_image_buff_size = resize_image_buff_bytes;
        } else if (resize_image_buff_bytes > _prev_image_buff_size) {  // Reallocate if intermediate output exceeds allocated memory
            CHECK_HIP(hipFree((void *)_rocjpeg_image_buff));
            CHECK_HIP(hipMalloc((void **)&_rocjpeg_image_buff, resize_image_buff_bytes));
            _prev_image_buff_size = resize_image_buff_bytes;
        }

        uint8_t *img_buff = reinterpret_cast<uint8_t*>(_rocjpeg_image_buff);
        size_t src_offset = 0;

        // Decode directly into the output tensor for images that don't need resizing.
        // For images that need resizing, decode into an intermediate buffer (original size) and then resize into output.
        // Update RocJpegImage pointers: resized images -> intermediate buffer; non-resized images -> output buffer.
        for (unsigned i = 0; i < _batch_size; i++) {
            if (_image_needs_rescaling[i]) {
                _output_images[i].channel[0] = static_cast<uint8_t *>(img_buff);    // For RGB

                const unsigned pitch_width = (original_image_width[i] + 8) & ~7;
                const unsigned pitch_height = (original_image_height[i] + 8) & ~7;
                const size_t img_bytes = static_cast<size_t>(pitch_width) * static_cast<size_t>(pitch_height) * static_cast<size_t>(_num_channels);

                _src_width[resize_count] = original_image_width[i];
                _src_height[resize_count] = original_image_height[i];
                // NOTE: dst sizes come from decode_info() (aspect-ratio fit). If these ever come in as 0 due to a
                // caller-side issue, the resize kernel will early-exit and leave output unchanged.
                _dst_width[resize_count] = actual_decoded_width[i] ? actual_decoded_width[i] : max_decoded_width;
                _dst_height[resize_count] = actual_decoded_height[i] ? actual_decoded_height[i] : max_decoded_height;
                _src_hstride[resize_count] = static_cast<size_t>(pitch_width * _num_channels);
                _src_img_offset[resize_count] = src_offset;
                _dst_img_idx[resize_count] = i;
                src_offset += img_bytes;
                img_buff += img_bytes;
                resize_count++;
            } else {
                _output_images[i].channel[0] = static_cast<uint8_t *>(output_buffer[i]);  // Direct decode into output tensor
            }
        }
    } else {
        for (unsigned i = 0; i < _batch_size; i++) {
            _output_images[i].channel[0] = static_cast<uint8_t *>(output_buffer[i]);    // For RGB
        }
    }

    CHECK_ROCJPEG(rocJpegDecodeBatched(_rocjpeg_handle, _rocjpeg_streams.data(), _batch_size, _decode_params.data(), _output_images.data()));

    if (_enable_resize && resize_count) {
        HipExecResizeTensor(_hip_stream, (void *)_rocjpeg_image_buff, (void *)output_buffer[0],
                            resize_count, _src_width, _src_height,
                            _dst_width, _dst_height, _src_hstride, _src_img_offset, _dst_img_idx, _num_channels,
                            max_decoded_width, max_decoded_height, max_decoded_width, max_decoded_height);
        // The circular buffer hands off `output_buffer` to the consumer immediately after `decode_batch()` returns.
        // Ensure the resize kernel has finished writing into the output tensor before we release the batch.
        CHECK_HIP(hipStreamSynchronize(_hip_stream));
    }
    _enable_resize = false;  // Need to reset this value for every batch
    _rocjpeg_image_buff_size = 0;
    std::fill(_image_needs_rescaling.begin(), _image_needs_rescaling.end(), false); // Reset per-image flags for this batch

    return Status::OK;
}

HWRocJpegDecoder::~HWRocJpegDecoder() {
    if (_rocjpeg_handle) {
        CHECK_ROCJPEG(rocJpegDestroy(_rocjpeg_handle));
        _rocjpeg_handle = nullptr;
    }
    for (size_t j = 0; j < _rocjpeg_streams.size(); j++) {
        CHECK_ROCJPEG(rocJpegStreamDestroy(_rocjpeg_streams[j]));
    }
    if (_rocjpeg_image_buff) CHECK_HIP(hipFree(_rocjpeg_image_buff));
    if (_src_width) CHECK_HIP(hipHostFree(_src_width));
    if (_src_height) CHECK_HIP(hipHostFree(_src_height));
    if (_dst_width) CHECK_HIP(hipHostFree(_dst_width));
    if (_dst_height) CHECK_HIP(hipHostFree(_dst_height));
    if (_src_hstride) CHECK_HIP(hipHostFree(_src_hstride));
    if (_src_img_offset) CHECK_HIP(hipHostFree(_src_img_offset));
    if (_dst_img_idx) CHECK_HIP(hipHostFree(_dst_img_idx));
}
#endif
