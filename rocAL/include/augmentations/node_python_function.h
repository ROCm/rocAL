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

#pragma once
#include <stddef.h>
#include <stdint.h>

#include "pipeline/graph.h"
#include "pipeline/node.h"
#include "parameters/parameter_factory.h"
#include "parameters/parameter_vx.h"

#ifdef ROCAL_PYTHON_FUNCTION
#ifndef ROCAL_PY_MAX_TENSOR_DIMS
#define ROCAL_PY_MAX_TENSOR_DIMS 5
#endif

typedef struct {
    size_t num_dims;                          /* e.g., 4 for [N,H,W,C] */
    size_t shape[ROCAL_PY_MAX_TENSOR_DIMS];   /* lengths per dimension */
    size_t strides[ROCAL_PY_MAX_TENSOR_DIMS]; /* strides in elements */
    vx_enum dtype;                            /* OpenVX scalar type enum */
    int layout;                               /* matches rocAL/vx tensor layout enums */
} RocalPyTensorDesc;

typedef struct {
    uint64_t function_id;        /* CPython id(function), provided by python front-end */
    RocalPyTensorDesc in_desc;   /* Input tensor descriptions */
    RocalPyTensorDesc out_desc;  /* Output tensor description */
    uint32_t device_type;        /* AGO_TARGET_AFFINITY_{CPU,GPU}; currently CPU-only */
} RocalPyExecParams;

/*
Execute the provided Python callable on a batched view of src_ptr described by
params->in_desc. The callable must return a NumPy array matching params->out_desc
(shape, ndim, dtype). The result will be copied into dst_ptr.

Parameters:
- src_ptr: Input tensor data pointer
- dst_ptr: Output tensor data pointer
- params: Execution parameters including function ID and tensor descriptions

Returns:
- VX_SUCCESS on success
- VX_ERROR_INVALID_DIMENSION / VX_ERROR_INVALID_TYPE on validation mismatch
- VX_FAILURE for runtime Python exceptions
- VX_ERROR_NOT_IMPLEMENTED if device_type is GPU or environment cannot execute
- VX_ERROR_INVALID_REFERENCE if src_ptr, dst_ptr, or params is null
*/
vx_status rocal_process_python_function(void* src_ptr, void* dst_ptr, const RocalPyExecParams* params);
#endif

class PythonFunctionNode : public Node {
   public:
    PythonFunctionNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    PythonFunctionNode() = delete;

    void init(unsigned long long function_id);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    unsigned long long _function_id = 0;
};
