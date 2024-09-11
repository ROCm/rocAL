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

#include <half/half.hpp>
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "rocal_hip_kernels.h"

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNHWC_fp32(
    const unsigned char *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    uint4 nchw,
    uint2 out_dims,
    float3 multiplier,
    float3 offset,
    unsigned int reverse_channels) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w;
    const int H = nchw.z;
    const int C = nchw.y;
    const int max_roi_height = out_dims.x;
    const int max_roi_width = out_dims.y;
    const int img_offset = C * W * H;
    const int out_img_offset = C * max_roi_width * max_roi_height;

    if ((x >= max_roi_width) || (y >= max_roi_height))
        return;
    for (unsigned int n = 0; n < nchw.x; n++) {
        unsigned int src_idx = (y * W + x) * C;  // src is RGB
        unsigned int dst_idx = (y * max_roi_width + x) * C;
        // copy float3  pixels to dst
        if (C == 3) {
            float3 dst;
            const uchar *inp_img = &inp_image_u8[n * img_offset];
            float *out_tensor = (float *)((float *)output_tensor + dst_buf_offset + n * out_img_offset);
            if (reverse_channels)
                dst = make_float3((float)inp_img[src_idx + 2], (float)inp_img[src_idx + 1], (float)inp_img[src_idx]) * multiplier + offset;
            else
                dst = make_float3((float)inp_img[src_idx], (float)inp_img[src_idx + 1], (float)inp_img[src_idx + 2]) * multiplier + offset;
            out_tensor[dst_idx] = dst.x;
            out_tensor[dst_idx + 1] = dst.y;
            out_tensor[dst_idx + 2] = dst.z;
        } else {
            const uchar *inp_img = &inp_image_u8[n * img_offset + dst_buf_offset];
            float *out_tensor = (float *)output_tensor + dst_buf_offset + n * out_img_offset;
            out_tensor[dst_idx] = (float)inp_img[src_idx] * multiplier.x + offset.x;
        }
    }
}

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNHWC_fp16(
    const unsigned char *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    uint4 nchw,
    uint2 out_dims,
    float3 multiplier,
    float3 offset,
    const unsigned int reverse_channels) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w;
    const int H = nchw.z;
    const int C = nchw.y;
    const int max_roi_height = out_dims.x;
    const int max_roi_width = out_dims.y;
    const int img_offset = C * W * H;
    const int out_img_offset = C * max_roi_width * max_roi_height;

    if ((x >= max_roi_width) || (y >= max_roi_height))
        return;
    for (unsigned int n = 0; n < nchw.x; n++) {
        __half *out_tensor = (__half *)output_tensor + dst_buf_offset + n * out_img_offset;
        unsigned int src_idx = (y * W + x) * C;
        // copy float3  pixels to dst
        if (C == 3) {
            unsigned int dst_idx = y * max_roi_width + x * 3;
            const uchar *inp_img = &inp_image_u8[n * img_offset];
            float3 dst;
            if (reverse_channels)
                dst = make_float3((float)inp_img[src_idx + 2], (float)inp_img[src_idx + 1], (float)inp_img[src_idx]) * multiplier + offset;
            else
                dst = make_float3((float)inp_img[src_idx], (float)inp_img[src_idx + 1], (float)inp_img[src_idx + 2]) * multiplier + offset;
            out_tensor[dst_idx] = __float2half(dst.x);
            out_tensor[dst_idx + 1] = __float2half(dst.y);
            out_tensor[dst_idx + 2] = __float2half(dst.z);
        } else {
            unsigned int dst_idx = y * max_roi_width + x;
            const uchar *inp_img = &inp_image_u8[n * img_offset];
            float *out_tensor = (float *)output_tensor + n * out_img_offset;
            out_tensor[dst_idx] = __float2half((float)inp_img[src_idx] * multiplier.x + offset.x);
        }
    }
}

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNCHW_fp32(
    const uchar *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    uint4 nchw,
    uint2 out_dims,
    float3 multiplier,
    float3 offset,
    unsigned int reverse_channels) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w;
    const int H = nchw.z;
    const int C = nchw.y;
    const int max_roi_height = out_dims.x;
    const int max_roi_width = out_dims.y;
    const int img_offset = C * W * H;
    const int out_img_offset = C * max_roi_width * max_roi_height;
    unsigned int cstride = max_roi_width * max_roi_height;

    if ((x >= max_roi_width) || (y >= max_roi_height))
        return;
    for (unsigned int n = 0; n < nchw.x; n++) {
        unsigned int src_idx = (y * W + x) * C;
        unsigned int dst_idx = y * max_roi_width + x;
        // copy float3  pixels to dst
        const uchar *inp_img = &inp_image_u8[n * img_offset];
        float *out_tensor = (float *)output_tensor + n * out_img_offset + dst_buf_offset;
        if (C == 3) {
            float3 dst;
            if (reverse_channels)
                dst = make_float3((float)inp_img[src_idx + 2], (float)inp_img[src_idx + 1], (float)inp_img[src_idx]) * multiplier + offset;
            else
                dst = make_float3((float)inp_img[src_idx], (float)inp_img[src_idx + 1], (float)inp_img[src_idx + 2]) * multiplier + offset;
            out_tensor[dst_idx] = dst.x;
            out_tensor[dst_idx + cstride] = dst.y;
            out_tensor[dst_idx + cstride * 2] = dst.z;
        } else {
            out_tensor[dst_idx] = (float)inp_img[src_idx] * multiplier.x + offset.x;
        }
    }
}

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNCHW_fp16(
    const uchar *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    uint4 nchw,
    uint2 out_dims,
    float3 multiplier,
    float3 offset,
    const unsigned int reverse_channels) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w;
    const int H = nchw.z;
    const int C = nchw.y;
    const int max_roi_height = out_dims.x;
    const int max_roi_width = out_dims.y;
    const int img_offset = C * W * H;
    const int out_img_offset = C * max_roi_width * max_roi_height;
    unsigned int cstride = max_roi_width * max_roi_height;

    if ((x >= max_roi_width) || (y >= max_roi_height))
        return;
    for (unsigned int n = 0; n < nchw.x; n++) {
        __half *out_tensor = (__half *)output_tensor + n * out_img_offset + dst_buf_offset;
        const uchar *inp_img = &inp_image_u8[n * img_offset];
        unsigned int src_idx = (y * W + x) * C;
        // copy float3  pixels to dst
        unsigned int dst_idx = y * max_roi_width + x;
        if (C == 3) {
            float3 dst;
            if (reverse_channels)
                dst = make_float3((float)inp_img[src_idx + 2], (float)inp_img[src_idx + 1], (float)inp_img[src_idx]) * multiplier + offset;
            else
                dst = make_float3((float)inp_img[src_idx], (float)inp_img[src_idx + 1], (float)inp_img[src_idx + 2]) * multiplier + offset;
            out_tensor[dst_idx] = __float2half(dst.x);
            out_tensor[dst_idx + cstride] = __float2half(dst.y);
            out_tensor[dst_idx + cstride * 2] = __float2half(dst.z);
        } else {
            out_tensor[dst_idx] = __float2half((float)inp_img[src_idx] * multiplier.x + offset.x);
        }
    }
}

int HipExecCopyInt8ToNHWC(
    hipStream_t stream,
    const void *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    const unsigned int n,
    const unsigned int c,
    const unsigned int h,
    const unsigned int w,
    float multiplier0,
    float multiplier1,
    float multiplier2,
    float offset0,
    float offset1,
    float offset2,
    unsigned int reverse_channels,
    unsigned int fp16,
    const unsigned max_output_height,
    const unsigned max_output_width) {
    int local_threads_x = 16, local_threads_y = 16;
    uint2 out_dims;
    if ((max_output_height == 0) || (max_output_width == 0))
        out_dims = make_uint2(h, w);
    else
        out_dims = make_uint2(max_output_height, max_output_width);
    int global_threads_x = w, global_threads_y = h;
    if (!fp16) {
        hipLaunchKernelGGL(Hip_CopyInt8ToNHWC_fp32,
                           dim3(ceil((float)global_threads_x / local_threads_x), ceil((float)global_threads_y / local_threads_y)),
                           dim3(local_threads_x, local_threads_y),
                           0, stream, (const uchar *)inp_image_u8, output_tensor, dst_buf_offset,
                           make_uint4(n, c, h, w), out_dims,
                           make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                           reverse_channels);
    } else {
        hipLaunchKernelGGL(Hip_CopyInt8ToNHWC_fp16,
                           dim3(ceil((float)global_threads_x / local_threads_x), ceil((float)global_threads_y / local_threads_y)),
                           dim3(local_threads_x, local_threads_y),
                           0, stream, (const uchar *)inp_image_u8, output_tensor, dst_buf_offset,
                           make_uint4(n, c, h, w), out_dims,
                           make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                           reverse_channels);
    }
    return 0;
}

int HipExecCopyInt8ToNCHW(
    hipStream_t stream,
    const void *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    const unsigned int n,
    const unsigned int c,
    const unsigned int h,
    const unsigned int w,
    float multiplier0,
    float multiplier1,
    float multiplier2,
    float offset0,
    float offset1,
    float offset2,
    unsigned int reverse_channels,
    unsigned int fp16,
    const unsigned max_output_height,
    const unsigned max_output_width) {
    int local_threads_x = 16, local_threads_y = 16;
    uint2 out_dims;
    if ((max_output_height == 0) || (max_output_width == 0))
        out_dims = make_uint2(h, w);
    else
        out_dims = make_uint2(max_output_height, max_output_width);
    int global_threads_x = w, global_threads_y = h;
    if (!fp16) {
        hipLaunchKernelGGL(Hip_CopyInt8ToNCHW_fp32,
                           dim3(ceil((float)global_threads_x / local_threads_x), ceil((float)global_threads_y / local_threads_y)),
                           dim3(local_threads_x, local_threads_y),
                           0, stream, (const uchar *)inp_image_u8, output_tensor, dst_buf_offset,
                           make_uint4(n, c, h, w), out_dims,
                           make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                           reverse_channels);
    } else {
        hipLaunchKernelGGL(Hip_CopyInt8ToNCHW_fp16,
                           dim3(ceil((float)global_threads_x / local_threads_x), ceil((float)global_threads_y / local_threads_y)),
                           dim3(local_threads_x, local_threads_y),
                           0, stream, (const uchar *)inp_image_u8, output_tensor, dst_buf_offset,
                           make_uint4(n, c, h, w), out_dims,
                           make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                           reverse_channels);
    }
    return 0;
}
