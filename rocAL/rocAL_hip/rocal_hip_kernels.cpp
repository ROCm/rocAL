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

typedef union { uint ui1[6];    uint2 ui2[3];                                                   }   d_uint6;
typedef union { float f1[8];    float2 f2[4];   float4 f4[2];                                   }   d_float8;
typedef union { float f1[12];   float4 f4[3];                                                   }   d_float12;
typedef union { float f1[16];   float4 f4[4];   d_float8 f8[2];                                 }   d_float16;
typedef union { float f1[24];   float2 f2[12];  float3 f3[8];   float4 f4[6];   d_float8 f8[3]; }   d_float24;
typedef struct { uint   data[ 6]; } d_uint6_s;

#define LOCAL_THREADS_X                 16                  // default rpp hip thread launch config - local threads x = 16
#define LOCAL_THREADS_Y                 16                  // default rpp hip thread launch config - local threads x = 16
#define LOCAL_THREADS_Z                  1                  // default rpp hip thread launch config - local threads x = 16

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

__device__ void resize_roi_and_srclocs_hip_compute(int4 *srcRoiPtr_i4, uint2 *dstDimsWH, int id_x, int id_y, d_float16 *locSrc_f16)
{
    float wRatio = (float)(srcRoiPtr_i4->z - srcRoiPtr_i4->x + 1) / dstDimsWH->x;
    float hRatio = (float)(srcRoiPtr_i4->w - srcRoiPtr_i4->y + 1) / dstDimsWH->y;
    float4 wOffset_f4 = (float4)((wRatio - 1) * 0.5f);
    float4 hOffset_f4 = (float4)((hRatio - 1) * 0.5f);

    d_float8 increment_f8, locDst_f8x, locDst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    locDst_f8x.f4[0] = (float4)id_x + increment_f8.f4[0];
    locDst_f8x.f4[1] = (float4)id_x + increment_f8.f4[1];
    locDst_f8y.f4[0] = (float4)id_y;
    locDst_f8y.f4[1] = (float4)id_y;

    locSrc_f16->f8[0].f4[0] = (locDst_f8x.f4[0] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [0-3]
    locSrc_f16->f8[0].f4[1] = (locDst_f8x.f4[1] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [4-7]
    locSrc_f16->f8[1].f4[0] = (locDst_f8y.f4[0] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [0-3]
    locSrc_f16->f8[1].f4[1] = (locDst_f8y.f4[1] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [4-7]
}

// float bilinear interpolation computation

__device__ __forceinline__ void rpp_hip_interpolate_bilinear(float4 *srcNeighborhood_f4, float2 *weightedWH_f2, float2 *oneMinusWeightedWH_f2, float *dst)
{
    *dst = fmaf(srcNeighborhood_f4->x, oneMinusWeightedWH_f2->y * oneMinusWeightedWH_f2->x,
                fmaf(srcNeighborhood_f4->y, oneMinusWeightedWH_f2->y * weightedWH_f2->x,
                    fmaf(srcNeighborhood_f4->z, weightedWH_f2->y * oneMinusWeightedWH_f2->x,
                        srcNeighborhood_f4->w * weightedWH_f2->y * weightedWH_f2->x)));
}

// ROI range check for source locations calculated

__device__ __forceinline__ void rpp_hip_roi_range_check(float2 *locSrcFloor_f2, int4 *roiPtrSrc_i4, int2 *locSrc_i2)
{
    locSrc_i2->x = (int)fminf(fmaxf(locSrcFloor_f2->x, roiPtrSrc_i4->x), roiPtrSrc_i4->z - 1);
    locSrc_i2->y = (int)fminf(fmaxf(locSrcFloor_f2->y, roiPtrSrc_i4->y), roiPtrSrc_i4->w - 1);
}

__device__ __forceinline__ float rpp_hip_unpack0(uint src)
{
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack1(uint src)
{
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack2(uint src)
{
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_load_pkd3(uchar *srcPtr, uint srcStrideH, float2 *locSrcFloor_f2, int4 *roiPtrSrc_i4, d_float12 *srcNeighborhood_f12)
{
    uint2 src_u2;
    int2 locSrc1_i2, locSrc2_i2;
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc1_i2);
    *locSrcFloor_f2 = *locSrcFloor_f2 + (float2)1.0f;
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc2_i2);
    int2 srcInterRowLoc_i2, srcInterColLoc_i2;
    srcInterRowLoc_i2.x = locSrc1_i2.y * srcStrideH;
    srcInterRowLoc_i2.y = locSrc2_i2.y * srcStrideH;
    srcInterColLoc_i2.x = locSrc1_i2.x * 3;
    srcInterColLoc_i2.y = locSrc2_i2.x * 3;

    int srcIdx1 = srcInterRowLoc_i2.x + srcInterColLoc_i2.x;   // Top Left
    int srcIdx2 = srcInterRowLoc_i2.x + srcInterColLoc_i2.y;   // Top Right
    src_u2.x = *(uint *)&srcPtr[srcIdx1];
    src_u2.y = *(uint *)&srcPtr[srcIdx2];
    srcNeighborhood_f12->f1[0] = rpp_hip_unpack0(src_u2.x);
    srcNeighborhood_f12->f1[1] = rpp_hip_unpack0(src_u2.y);
    srcNeighborhood_f12->f1[4] = rpp_hip_unpack1(src_u2.x);
    srcNeighborhood_f12->f1[5] = rpp_hip_unpack1(src_u2.y);
    srcNeighborhood_f12->f1[8] = rpp_hip_unpack2(src_u2.x);
    srcNeighborhood_f12->f1[9] = rpp_hip_unpack2(src_u2.y);
    srcIdx1 = srcInterRowLoc_i2.y + srcInterColLoc_i2.x;   // Bottom left
    srcIdx2 = srcInterRowLoc_i2.y + srcInterColLoc_i2.y;   // Bottom right
    src_u2.x = *(uint *)&srcPtr[srcIdx1];
    src_u2.y = *(uint *)&srcPtr[srcIdx2];
    srcNeighborhood_f12->f1[ 2] = rpp_hip_unpack0(src_u2.x);
    srcNeighborhood_f12->f1[ 3] = rpp_hip_unpack0(src_u2.y);
    srcNeighborhood_f12->f1[ 6] = rpp_hip_unpack1(src_u2.x);
    srcNeighborhood_f12->f1[ 7] = rpp_hip_unpack1(src_u2.y);
    srcNeighborhood_f12->f1[10] = rpp_hip_unpack2(src_u2.x);
    srcNeighborhood_f12->f1[11] = rpp_hip_unpack2(src_u2.y);
}

// float3 bilinear interpolation pkd3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_pkd3(T *srcPtr, uint srcStrideH, float locSrcX, float locSrcY, int4 *roiPtrSrc_i4, float3 *dst_f3, bool checkRange)
{
    float2 locSrcFloor_f2, weightedWH_f2, oneMinusWeightedWH_f2;
    locSrcFloor_f2.x = floorf(locSrcX);
    locSrcFloor_f2.y = floorf(locSrcY);
    if (checkRange && ((locSrcFloor_f2.x < roiPtrSrc_i4->x) || (locSrcFloor_f2.y < roiPtrSrc_i4->y) || (locSrcFloor_f2.x > roiPtrSrc_i4->z) || (locSrcFloor_f2.y > roiPtrSrc_i4->w)))
    {
        *dst_f3 = (float3) 0.0f;
    }
    else
    {
        weightedWH_f2.x = locSrcX - locSrcFloor_f2.x;
        weightedWH_f2.y = locSrcY - locSrcFloor_f2.y;
        oneMinusWeightedWH_f2.x = 1.0f - weightedWH_f2.x;
        oneMinusWeightedWH_f2.y = 1.0f - weightedWH_f2.y;
        d_float12 srcNeighborhood_f12;
        rpp_hip_interpolate3_bilinear_load_pkd3(srcPtr, srcStrideH, &locSrcFloor_f2, roiPtrSrc_i4, &srcNeighborhood_f12);
        rpp_hip_interpolate_bilinear(&srcNeighborhood_f12.f4[0], &weightedWH_f2, &oneMinusWeightedWH_f2, &(dst_f3->x));
        rpp_hip_interpolate_bilinear(&srcNeighborhood_f12.f4[1], &weightedWH_f2, &oneMinusWeightedWH_f2, &(dst_f3->y));
        rpp_hip_interpolate_bilinear(&srcNeighborhood_f12.f4[2], &weightedWH_f2, &oneMinusWeightedWH_f2, &(dst_f3->z));
    }
}

// d_float24 bilinear interpolation in pkd3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate24_bilinear_pkd3(T *srcPtr, uint srcStrideH, d_float16 *locPtrSrc_f16, int4 *roiPtrSrc_i4, d_float24 *dst_f24, bool checkRange = true)
{
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[0], locPtrSrc_f16->f1[ 8], roiPtrSrc_i4, &(dst_f24->f3[0]), checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[1], locPtrSrc_f16->f1[ 9], roiPtrSrc_i4, &(dst_f24->f3[1]), checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[2], locPtrSrc_f16->f1[10], roiPtrSrc_i4, &(dst_f24->f3[2]), checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[3], locPtrSrc_f16->f1[11], roiPtrSrc_i4, &(dst_f24->f3[3]), checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[4], locPtrSrc_f16->f1[12], roiPtrSrc_i4, &(dst_f24->f3[4]), checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[5], locPtrSrc_f16->f1[13], roiPtrSrc_i4, &(dst_f24->f3[5]), checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[6], locPtrSrc_f16->f1[14], roiPtrSrc_i4, &(dst_f24->f3[6]), checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[7], locPtrSrc_f16->f1[15], roiPtrSrc_i4, &(dst_f24->f3[7]), checkRange);
}

// Packing to U8s
__device__ __forceinline__ uint rpp_hip_pack(float4 src) {
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

// U8 stores without layout toggle PKD3 to PKD3 (24 U8 pixels)
__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(uchar *dstPtr, d_float24 *dstPtr_f24) {
    d_uint6 dst_ui6;

    dst_ui6.ui1[0] = rpp_hip_pack(dstPtr_f24->f4[0]);    // write R00G00B00R01
    dst_ui6.ui1[1] = rpp_hip_pack(dstPtr_f24->f4[1]);    // write G01B01R02G02
    dst_ui6.ui1[2] = rpp_hip_pack(dstPtr_f24->f4[2]);    // write B02R03G03B03
    dst_ui6.ui1[3] = rpp_hip_pack(dstPtr_f24->f4[3]);    // write R04G04B04R05
    dst_ui6.ui1[4] = rpp_hip_pack(dstPtr_f24->f4[4]);    // write G05B05R06G06
    dst_ui6.ui1[5] = rpp_hip_pack(dstPtr_f24->f4[5]);    // write B06R07G07B07

    *(d_uint6_s *)dstPtr = *(d_uint6_s *)&dst_ui6;
}

template <typename T>
__global__ void resize_bilinear_pkd_hip_tensor(T *srcPtr,
                                           uint2 srcStridesNH,
                                           T *dstPtr,
                                           uint2 dstStridesNH,
                                           size_t *srcWidth,
                                           size_t *srcHeight,
                                           size_t *dstWidth,
                                           size_t *dstHeight,
                                           size_t *srcHeightStride,
                                           size_t *srcImgOffset) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstWidth[id_z];
    dstDimsWH.y = dstHeight[id_z];

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = srcImgOffset[id_z];
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    int4 srcRoi_i4;
    srcRoi_i4.x = 0;
    srcRoi_i4.y = 0;
    srcRoi_i4.z = srcWidth[id_z];
    srcRoi_i4.w = srcHeight[id_z];

    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcHeightStride[id_z], &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

void HipExecResizeTensor(
                              hipStream_t stream,
                              void *srcPtr,
                              void *dstPtr,
                              unsigned batchSize,
                              size_t *srcWidth,
                              size_t *srcHeight,
                              size_t *dstWidth,
                              size_t *dstHeight,
                              size_t *srcHeightStride,
                              size_t *srcImgOffset,
                              unsigned channels,
                              const size_t maxSrcWidth,
                              const size_t maxSrcHeight,
                              const size_t maxDstWidth,
                              const size_t maxDstHeight) {
    unsigned globalThreads_x = (maxDstWidth + 7) >> 3;
    unsigned globalThreads_y = maxDstHeight;
    unsigned globalThreads_z = batchSize;
    hipLaunchKernelGGL(resize_bilinear_pkd_hip_tensor,
                        dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                        dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                        0,
                        stream,
                        static_cast<unsigned char *>(srcPtr),
                        make_uint2(maxSrcWidth * maxSrcHeight * channels, maxSrcWidth * channels),
                        static_cast<unsigned char *>(dstPtr),
                        make_uint2(maxDstWidth * maxDstHeight * channels, maxDstWidth * channels),
                        srcWidth,
                        srcHeight,
                        dstWidth,
                        dstHeight,
                        srcHeightStride,
                        srcImgOffset);

}
