/*
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

#include "decoders/image/rocjpeg_fused_crop_decoder.h"

#include <stdio.h>
#include <string.h>

#include "pipeline/commons.h"

#include "decoders/image/rocjpeg_decoder.h"
#include "pipeline/commons.h"

#if ENABLE_ROCJPEG

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "rocal_hip_kernels.h"

#define CHECK_HIP(call)                                                                                                        \
    {                                                                                                                          \
        hipError_t hip_status = (call);                                                                                        \
        if (hip_status != hipSuccess) {                                                                                        \
            std::cerr << "HIP failure: 'status: " << hipGetErrorName(hip_status) << "' at " << __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                      \
        }                                                                                                                      \
    }

#define CHECK_ROCJPEG(call)                                                                                                \
    {                                                                                                                      \
        RocJpegStatus rocjpeg_status = (call);                                                                             \
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {                                                                    \
            std::cerr << #call << " returned " << rocJpegGetErrorName(rocjpeg_status) << " at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                        \
        }                                                                     \
    }

void FusedCropRocJpegDecoder::set_bbox_coords(std::vector<float> bbox_coord) {
    _bbox_coord = bbox_coord;
    _crop_window.x = std::lround(_bbox_coord[0] * _original_image_width);
    _crop_window.y = std::lround(_bbox_coord[1] * _original_image_height);
    _crop_window.W = std::lround((_bbox_coord[2]) * _original_image_width);
    _crop_window.H = std::lround((_bbox_coord[3]) * _original_image_height);

    // Clamp crop dimensions to max decoded dimensions to ensure ROI consistency with output constraints.
    _crop_window.W = std::min(_crop_window.W, static_cast<unsigned int>(_max_decoded_width));
    _crop_window.H = std::min(_crop_window.H, static_cast<unsigned int>(_max_decoded_height));

    _decode_params->crop_rectangle.left = _crop_window.x;
    _decode_params->crop_rectangle.top = _crop_window.y;
    _decode_params->crop_rectangle.right = _crop_window.W + _crop_window.x - 1;
    _decode_params->crop_rectangle.bottom = _crop_window.H + _crop_window.y - 1;

    if (static_cast<size_t>(_current_index) < _batch_size) _roi_width[_current_index] = _crop_window.W;
    if (static_cast<size_t>(_current_index) < _batch_size) _roi_height[_current_index] = _crop_window.H;
}

void FusedCropRocJpegDecoder::set_crop_window(CropWindow &crop_window) {
    _crop_window = crop_window;
    _crop_window.W = std::min(_crop_window.W, static_cast<unsigned int>(_max_decoded_width));
    _crop_window.H = std::min(_crop_window.H, static_cast<unsigned int>(_max_decoded_height));

    _decode_params->crop_rectangle.left = _crop_window.x;
    _decode_params->crop_rectangle.top = _crop_window.y;
    _decode_params->crop_rectangle.right = _crop_window.W + _crop_window.x - 1;
    _decode_params->crop_rectangle.bottom = _crop_window.H + _crop_window.y - 1;

    if (static_cast<size_t>(_current_index) < _batch_size) _roi_width[_current_index] = _crop_window.W;
    if (static_cast<size_t>(_current_index) < _batch_size) _roi_height[_current_index] = _crop_window.H;
}

void FusedCropRocJpegDecoder::initialize(int device_id, unsigned batch_size) {
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

    _batch_size = batch_size;
    _output_images.resize(_batch_size);
    _decode_params_batch.resize(_batch_size);
    _roi_width.assign(_batch_size, 0);
    _roi_height.assign(_batch_size, 0);
}

// Obtains the decode info of the image, and modifies width and height based on the max decode params after scaling
Decoder::Status FusedCropRocJpegDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *actual_width,
                                                     int *actual_height, int max_decoded_width, int max_decoded_height, Decoder::ColorFormat desired_decoded_color_format, int index) {
    uint8_t num_components;
    uint32_t widths[4] = {};
    uint32_t heights[4] = {};
    uint32_t channels_size = 0;
    uint32_t channel_sizes[4] = {};

    uint32_t max_widths[4] = {static_cast<uint32_t>(max_decoded_width), 0, 0, 0};
    uint32_t max_heights[4] = {static_cast<uint32_t>(max_decoded_height), 0, 0, 0};

    RocJpegChromaSubsampling subsampling;
    switch (desired_decoded_color_format) {
        case Decoder::ColorFormat::GRAY:
            _decode_params_batch[index].output_format = ROCJPEG_OUTPUT_Y;
            break;
        case Decoder::ColorFormat::RGB:
        case Decoder::ColorFormat::BGR:
            _decode_params_batch[index].output_format = ROCJPEG_OUTPUT_RGB;
            break;
    };

    if (rocJpegStreamParse(reinterpret_cast<uint8_t *>(input_buffer), input_size, _rocjpeg_streams[index]) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    if (rocJpegGetImageInfo(_rocjpeg_handle, _rocjpeg_streams[index], &num_components, &subsampling, widths, heights) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    if (widths[0] < 64 || heights[0] < 64) {
        return Status::CONTENT_DECODE_FAILED;
    }
    if (subsampling == ROCJPEG_CSS_411 || subsampling == ROCJPEG_CSS_UNKNOWN) {
        return Status::UNSUPPORTED;
    }

    *width = widths[0];
    *height = heights[0];
    _max_decoded_width = max_decoded_width;
    _max_decoded_height = max_decoded_height;
    _current_index = index;
    _decode_params = &_decode_params_batch[index];

    if (GetChannelPitchAndSizes(_decode_params_batch[index], subsampling, max_widths, max_heights, channels_size, _output_images[index], channel_sizes)) {
        return Status::HEADER_DECODE_FAILED;
    }
    _original_image_width = *actual_width = widths[0];
    _original_image_height = *actual_height = heights[0];
    _roi_width[index] = widths[0];
    _roi_height[index] = heights[0];

    return Status::OK;
}

// Obtains only the decode info of the image used for image source evaluation
Decoder::Status FusedCropRocJpegDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *color_comps) {
    RocJpegChromaSubsampling subsampling;
    uint8_t num_components;
    uint32_t widths[4] = {};
    uint32_t heights[4] = {};
    if (rocJpegStreamParse(reinterpret_cast<uint8_t *>(input_buffer), input_size, _rocjpeg_streams[0]) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    if (rocJpegGetImageInfo(_rocjpeg_handle, _rocjpeg_streams[0], &num_components, &subsampling, widths, heights) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    *width = widths[0];
    *height = heights[0];

    if (widths[0] < 64 || heights[0] < 64) {
        return Status::CONTENT_DECODE_FAILED;
    }
    if (subsampling == ROCJPEG_CSS_411 || subsampling == ROCJPEG_CSS_UNKNOWN) {
        return Status::UNSUPPORTED;
    }
    return Status::OK;
}

Decoder::Status FusedCropRocJpegDecoder::decode_batch(std::vector<unsigned char *> &output_buffer,
                                                      size_t max_decoded_width, size_t max_decoded_height,
                                                      std::vector<size_t> original_image_width, std::vector<size_t> original_image_height,
                                                      std::vector<size_t> &actual_decoded_width, std::vector<size_t> &actual_decoded_height) {
    for (unsigned i = 0; i < _batch_size; i++) {
        _output_images[i].channel[0] = static_cast<uint8_t *>(output_buffer[i]);  // For RGB
        // Update the actual decoded width and height based on the crop window and max decode params
        actual_decoded_width[i] = _roi_width[i];
        actual_decoded_height[i] = _roi_height[i];
    }
    CHECK_ROCJPEG(rocJpegDecodeBatched(_rocjpeg_handle, _rocjpeg_streams.data(), _batch_size, _decode_params_batch.data(), _output_images.data()));

    return Status::OK;
}

FusedCropRocJpegDecoder::~FusedCropRocJpegDecoder() {
    CHECK_ROCJPEG(rocJpegDestroy(_rocjpeg_handle));
    for (auto j = 0; j < _batch_size; j++) {
        CHECK_ROCJPEG(rocJpegStreamDestroy(_rocjpeg_streams[j]));
    }
}
#endif
