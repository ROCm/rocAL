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

#pragma once

#include <cstddef>
#include <iostream>
#include <vector>
#include "parameter_factory.h"
#include "sndfile.h"

class AudioDecoder {
public:
    enum class Status {
        OK = 0,
        HEADER_DECODE_FAILED,
        CONTENT_DECODE_FAILED,
        UNSUPPORTED,
        FAILED,
        NO_MEMORY
    };
    virtual AudioDecoder::Status initialize(const char *src_filename) = 0; // This function is responsible for initializing the audio decoder. It takes the source filename as input and returns the status of the initialization process.
    virtual AudioDecoder::Status decode(float* buffer) = 0; //to pass buffer & number of frames/samples to decode
    virtual AudioDecoder::Status decode_info(int* samples, int* channels, float* sample_rates) = 0; //to decode info about the audio samples
    virtual void release() = 0;
    virtual ~AudioDecoder() = default;
protected:
    const char *_src_filename = NULL;
    SF_INFO _sfinfo;
    SNDFILE* _sf_ptr;
};

