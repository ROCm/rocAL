/*
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

#pragma once

#if ENABLE_ROCJPEG

#include <string>

#include "hip/hip_runtime_api.h"
#include "pipeline/commons.h"
#include "rocjpeg/rocjpeg.h"

// Error checks shared by the rocJPEG decoders (HWRocJpegDecoder and FusedCropRocJpegDecoder).
//
// These replace macros that called exit(1) from inside the library. That made any rocJPEG or HIP
// failure - a decode error, an out-of-memory, an unsupported stream - end the host process with no
// exception the caller could catch, so a Python training script just stopped with status 1. Worse,
// decode_batch() runs inside the loader's `omp parallel for`, so when more than one decoder failed
// the workers called exit() concurrently and the process died with SIGSEGV instead.
//
// Three flavours, chosen by what the calling context can tolerate:
//   *_RETURN - returns a Decoder::Status to the caller. Use in decode_info()/decode_batch(), which
//              run inside the loader's OpenMP region. The loader records the failure and rethrows
//              it from the main thread once the region has ended; throwing out of an OpenMP worker
//              is undefined behaviour, so these must not throw.
//   *_THROW  - throws a RocalException. Only for initialize(), which the ImageReadAndDecode
//              constructor calls on the main thread, and which the base-class Decoder::initialize()
//              already reports failures from with THROW.
//   *_LOG    - logs and carries on. Only for destructors, which must not throw.
#define ROCJPEG_FAILURE_MSG(call_str, error_name)                   \
    (std::string(call_str) + " returned " + (error_name) + " at " + \
     std::string(__FILE__) + ":" + TOSTR(__LINE__))

#define CHECK_HIP_RETURN(call, failure_status)                       \
    {                                                                \
        hipError_t hip_status = (call);                              \
        if (hip_status != hipSuccess) {                              \
            ERR(ROCJPEG_FAILURE_MSG(#call, hipGetErrorName(hip_status))) \
            return failure_status;                                   \
        }                                                            \
    }

#define CHECK_ROCJPEG_RETURN(call, failure_status)                            \
    {                                                                         \
        RocJpegStatus rocjpeg_status = (call);                                \
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {                       \
            ERR(ROCJPEG_FAILURE_MSG(#call, rocJpegGetErrorName(rocjpeg_status))) \
            return failure_status;                                            \
        }                                                                     \
    }

#define CHECK_HIP_THROW(call)                                          \
    {                                                                  \
        hipError_t hip_status = (call);                                \
        if (hip_status != hipSuccess) {                                \
            THROW(ROCJPEG_FAILURE_MSG(#call, hipGetErrorName(hip_status))) \
        }                                                              \
    }

#define CHECK_ROCJPEG_THROW(call)                                               \
    {                                                                           \
        RocJpegStatus rocjpeg_status = (call);                                  \
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {                         \
            THROW(ROCJPEG_FAILURE_MSG(#call, rocJpegGetErrorName(rocjpeg_status))) \
        }                                                                       \
    }

#define CHECK_HIP_LOG(call)                                              \
    {                                                                    \
        hipError_t hip_status = (call);                                  \
        if (hip_status != hipSuccess) {                                  \
            ERR(ROCJPEG_FAILURE_MSG(#call, hipGetErrorName(hip_status))) \
        }                                                                \
    }

#define CHECK_ROCJPEG_LOG(call)                                                  \
    {                                                                            \
        RocJpegStatus rocjpeg_status = (call);                                   \
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {                          \
            ERR(ROCJPEG_FAILURE_MSG(#call, rocJpegGetErrorName(rocjpeg_status))) \
        }                                                                        \
    }

#endif  // ENABLE_ROCJPEG
