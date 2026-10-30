/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include <cstddef>
#include <iostream>
#include <vector>
#include "parameters/parameter_factory.h"

namespace rocal {
namespace video {

enum class VideoPixelFormat {
    UNKNOWN = 0,
    RGB24,
    BGR24,
    GRAY8,
    YUV420P,
    NV12
};

}  // namespace video
}  // namespace rocal

class VideoDecoder {
   public:
    enum class Status {
        OK = 0,
        HEADER_DECODE_FAILED,
        CONTENT_DECODE_FAILED,
        UNSUPPORTED,
        FAILED,
        NO_MEMORY
    };
    enum class ColorFormat {
        GRAY = 0,
        RGB,
        BGR
    };
    virtual Status Initialize(const char *src_filename, int device_id = 0) = 0;
    virtual Status Decode(unsigned char *output_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, rocal::video::VideoPixelFormat out_format) = 0;
    virtual void Release() = 0;
    virtual ~VideoDecoder() = default;
};
