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

#define CHECK_HIP(call) {                                             \
    hipError_t hip_status = (call);                                   \
    if (hip_status != hipSuccess) {                                   \
        std::cerr << "HIP failure: 'status: " << hipGetErrorName(hip_status) << "' at " << __FILE__ << ":" << __LINE__ << std::endl;\
        return ROCJPEG_STATUS_EXECUTION_FAILED;                       \
    }                                                                 \
}

RocJpegDecoder::RocJpegDecoder() {

};

void initialize(int device_id) {
    int num_devices;
    hipDeviceProp_t hip_dev_prop;
    CHECK_HIP(hipGetDeviceCount(&num_devices));
    if (num_devices < 1) {
        std::cerr << "ERROR: didn't find any GPU!" << std::endl;
        return false;
    }
    if (device_id >= num_devices) {
        std::cerr << "ERROR: the requested device_id is not found!" << std::endl;
        return false;
    }
    // CHECK_HIP(hipSetDevice(device_id));
    CHECK_HIP(hipGetDeviceProperties(&hip_dev_prop, device_id));

    std::cout << "Using GPU device " << device_id << ": " << hip_dev_prop.name << "[" << hip_dev_prop.gcnArchName << "] on PCI bus " <<
    std::setfill('0') << std::setw(2) << std::right << std::hex << hip_dev_prop.pciBusID << ":" << std::setfill('0') << std::setw(2) <<
    std::right << std::hex << hip_dev_prop.pciDomainID << "." << hip_dev_prop.pciDeviceID << std::dec << std::endl;
}

Decoder::Status RocJpegDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *color_comps) {
    // TODO : Use the most recent TurboJpeg API tjDecompressHeader3 which returns the color components
    // if (tjDecompressHeader2(m_jpegDecompressor,
    //                         input_buffer,
    //                         input_size,
    //                         width,
    //                         height,
    //                         color_comps) != 0) {
    //     WRN("Jpeg header decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
    //     return Status::HEADER_DECODE_FAILED;
    // }
    return Status::OK;
}

Decoder::Status RocJpegDecoder::decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                           size_t max_decoded_width, size_t max_decoded_height,
                                           size_t original_image_width, size_t original_image_height,
                                           size_t &actual_decoded_width, size_t &actual_decoded_height,
                                           Decoder::ColorFormat desired_decoded_color_format, DecoderConfig decoder_config, bool keep_original_size) {

    return Status::OK;
}

RocJpegDecoder::~RocJpegDecoder() {
}
#endif
