/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#include "decoders/image/rocjpeg_fused_crop_decoder.h"

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

FusedCropRocJpegDecoder::FusedCropRocJpegDecoder() {}

void FusedCropRocJpegDecoder::initialize(int device_id) {
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
    _device_id = device_id;
}

Decoder::Status FusedCropRocJpegDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *color_comps) {
    RocJpegChromaSubsampling subsampling;
    uint8_t num_components;
    uint32_t widths[4] = {};
    uint32_t heights[4] = {};
    if (rocJpegStreamParse(reinterpret_cast<uint8_t*>(input_buffer), input_size, _rocjpeg_stream) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    if (rocJpegGetImageInfo(_rocjpeg_handle, _rocjpeg_stream, &num_components, &subsampling, widths, heights) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    *width = widths[0];
    *height = heights[0];
    // _rocjpeg_image_buff_size += (((widths[0] + 8) &~ 7) * ((heights[0] + 8) &~ 7));

    if (widths[0] < 64 || heights[0] < 64) {
        return Status::CONTENT_DECODE_FAILED;
    }

    std::string chroma_sub_sampling = "";
    GetChromaSubsamplingStr(subsampling, chroma_sub_sampling);
    if (subsampling == ROCJPEG_CSS_440 || subsampling == ROCJPEG_CSS_411 || subsampling == ROCJPEG_CSS_UNKNOWN) {
        return Status::UNSUPPORTED;
    }
    return Status::OK;
}

Decoder::Status FusedCropRocJpegDecoder::decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                           size_t max_decoded_width, size_t max_decoded_height,
                                           size_t original_image_width, size_t original_image_height,
                                           size_t &actual_decoded_width, size_t &actual_decoded_height,
                                           Decoder::ColorFormat desired_decoded_color_format, DecoderConfig decoder_config, bool keep_original_size) {
    int tjpf = TJPF_RGB;
    int planes = 1;
    switch (desired_decoded_color_format) {
        case Decoder::ColorFormat::GRAY:
            tjpf = TJPF_GRAY;
            planes = 1;
            break;
        case Decoder::ColorFormat::RGB:
            tjpf = TJPF_RGB;
            planes = 3;
            break;
        case Decoder::ColorFormat::BGR:
            tjpf = TJPF_BGR;
            planes = 3;
            break;
    };
    actual_decoded_width = max_decoded_width;
    actual_decoded_height = max_decoded_height;
    // You need get the output of random bbox crop
    // check the vector size for bounding box. If its more than zero go for random bbox crop
    // else go to random crop
    unsigned int x1_diff, crop_width_diff;
    if (_bbox_coord.size() != 0) {
        // Random bbox crop returns normalized crop cordinates
        // hence bringing it back to absolute cordinates
        _crop_window.x = std::lround(_bbox_coord[0] * original_image_width);
        _crop_window.y = std::lround(_bbox_coord[1] * original_image_height);
        _crop_window.W = std::lround((_bbox_coord[2]) * original_image_width);
        _crop_window.H = std::lround((_bbox_coord[3]) * original_image_height);
    }
    _crop_window.W = std::min(_crop_window.W, (unsigned int)max_decoded_width);
    _crop_window.H = std::min(_crop_window.H, (unsigned int)max_decoded_height);
    // TODO : Turbo Jpeg supports multiple color packing and color formats, add more as an option to the API TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_RGBA, TJPF_GRAY, TJPF_CMYK , ...
    if (tjDecompress2_partial(m_jpegDecompressor,
                              input_buffer,
                              input_size,
                              output_buffer,
                              max_decoded_width,
                              max_decoded_width * planes,
                              max_decoded_height,
                              tjpf,
                              TJFLAG_ACCURATEDCT, &x1_diff, &crop_width_diff,
                              _crop_window.x, _crop_window.y, _crop_window.W, _crop_window.H) != 0) {
        WRN("Jpeg image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
        return Status::CONTENT_DECODE_FAILED;
    }

    // x1-diff should be set to x offset in tensor pipeline and removed.
    if (_crop_window.x != x1_diff) {
        unsigned char *src_ptr_temp, *dst_ptr_temp;
        unsigned int elements_in_row = max_decoded_width * planes;
        unsigned int elements_in_crop_row = _crop_window.W * planes;
        unsigned int xoffs = (_crop_window.x - x1_diff) * planes;  // in case _crop_window.x gets adjusted by tjpeg decoder
        src_ptr_temp = output_buffer;
        dst_ptr_temp = output_buffer;
        for (unsigned int i = 0; i < _crop_window.H; i++) {
            memcpy(dst_ptr_temp, src_ptr_temp + xoffs, elements_in_crop_row * sizeof(unsigned char));
            src_ptr_temp += elements_in_row;
            dst_ptr_temp += elements_in_row;
        }
    }
    actual_decoded_width = _crop_window.W;
    actual_decoded_height = _crop_window.H;

    return Status::OK;
}

FusedCropRocJpegDecoder::~FusedCropRocJpegDecoder() {
    tjDestroy(m_jpegDecompressor);
}
#endif
