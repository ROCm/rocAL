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

// C ABI bridge for executing Python callables from rocAL, callable by external
// consumers (e.g., MIVisionX OpenVX kernels) without exposing any pybind11
// types or requiring Python headers on the caller side.

// This header intentionally depends only on OpenVX public types.

#ifndef ROCAL_PYTHON_BRIDGE_H_
#define ROCAL_PYTHON_BRIDGE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <VX/vx.h>
#include <stddef.h>
#include <stdint.h>

/*
Notes:
- All shapes/strides are expressed per-element counts. The bridge implementation
  will multiply by element size (derived from dtype) where byte strides are needed.
- dtype uses OpenVX vx_enum values (e.g., VX_TYPE_UINT8, VX_TYPE_FLOAT32, ...).
- layout is passed for informational/validation purposes. The bridge does not
  reorder dimensions; the Python callable must honor the provided layout/shape.
*/

#ifndef ROCAL_PY_MAX_TENSOR_DIMS
#define ROCAL_PY_MAX_TENSOR_DIMS 8
#endif

typedef struct RocalPyTensorDesc_ {
    size_t num_dims;                          /* e.g., 4 for [N,H,W,C] */
    size_t shape[ROCAL_PY_MAX_TENSOR_DIMS];   /* lengths per dimension */
    size_t strides[ROCAL_PY_MAX_TENSOR_DIMS]; /* strides in elements */
    vx_enum dtype;                            /* OpenVX scalar type enum */
    int layout;                               /* matches rocAL/vx tensor layout enums */
} RocalPyTensorDesc;

typedef struct RocalPyExecParams_ {
    uint64_t function_id; /* CPython id(function), provided by python front-end */
    RocalPyTensorDesc in_desc;
    RocalPyTensorDesc out_desc;
    int roi_type;         /* reserved for future use; pass-through */
    uint32_t device_type; /* AGO_TARGET_AFFINITY_{CPU,GPU}; currently CPU-only */
} RocalPyExecParams;

/*
Execute the provided Python callable on a batched view of src_ptr described by
params->in_desc. The callable must return a NumPy array matching params->out_desc
(shape, ndim, dtype). The result will be copied into dst_ptr.

Returns:
- VX_SUCCESS on success
- VX_ERROR_INVALID_DIMENSION / VX_ERROR_INVALID_TYPE on validation mismatch
- VX_FAILURE for runtime Python exceptions
- VX_ERROR_NOT_IMPLEMENTED if device_type is GPU or environment cannot execute
*/
vx_status rocal_process_python_function(void* src_ptr, void* dst_ptr, const RocalPyExecParams* params);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCAL_PYTHON_BRIDGE_H_ */
