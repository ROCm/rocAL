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

#include "augmentations/geometry_augmentations/node_remap.h"

#include <vx_ext_rpp.h>
#include <vx_ext_rpp_version.h>
#include <vx_ext_amd.h>
#include "pipeline/exception.h"

RemapNode::RemapNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
    : Node(inputs, outputs) {}

void RemapNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 4)
    // Validate vectors
    if (_row_remap_vec.empty() || _col_remap_vec.empty())
        THROW("Remap node requires non-empty row and col remap vectors")

    // Determine N (sequence-aware) and output H,W
    const auto& in_dims = _inputs[0]->info().dims();
    if (in_dims.empty())
        THROW("Invalid input dims for Remap");
    vx_size N = static_cast<vx_size>(in_dims[0]);
    auto layout = _inputs[0]->info().layout();
    if (layout == RocalTensorlayout::NFCHW || layout == RocalTensorlayout::NFHWC) {
        if (in_dims.size() < 2) THROW("Invalid sequence dims for Remap");
        N = static_cast<vx_size>(in_dims[0] * in_dims[1]);
    }

    const auto& out_max = _outputs[0]->info().max_shape();
    if (out_max.size() < 2)
        THROW("Invalid output tensor shape for Remap");
    // max_shape for images is [W, H]
    const vx_size W = static_cast<vx_size>(out_max[0]);
    const vx_size H = static_cast<vx_size>(out_max[1]);

    const vx_size elems_per_sample = H * W;
    const vx_size total_expected   = N * elems_per_sample;

    // Select memory type and VX mem type
    auto mem_type = _inputs[0]->info().mem_type();
    vx_enum vx_mem = (mem_type == RocalMemType::HIP) ? VX_MEMORY_TYPE_HIP : VX_MEMORY_TYPE_HOST;

    auto make_and_fill_map_handle = [&](const std::vector<float>& vec, void** backing_ptr, vx_tensor* out_t) {
        const bool replicate = (vec.size() == elems_per_sample);
        if (!(replicate || vec.size() == total_expected))
            THROW("Remap table size mismatch. Expected " + TOSTR(total_expected) + " or " + TOSTR(elems_per_sample) + ", got " + TOSTR(vec.size()));

        // Create 4D tensor [N, H, W, 1] FP32 (NHWC with C=1), strides from last to first
        vx_size dims[4]   = {N, H, W, 1};
        vx_size stride[4] = {0, 0, 0, 0};
        stride[0] = sizeof(vx_float32);
        stride[1] = stride[0] * dims[0];
        stride[2] = stride[1] * dims[1];
        stride[3] = stride[2] * dims[2];

        // Allocate buffer and create tensor from handle
        size_t bytes = stride[3];          // total bytes for N images
        allocate_host_or_pinned_mem(backing_ptr, bytes, mem_type);

        vx_tensor t = vxCreateTensorFromHandle(vxGetContext((vx_reference)_graph->get()),
                                               4, dims, VX_TYPE_FLOAT32, 0, stride, *backing_ptr, vx_mem);
        if (!t) THROW("vxCreateTensorFromHandle for remap table failed");
        vx_status s = vxGetStatus((vx_reference)t);
        if (s != VX_SUCCESS)
            THROW("vxCreateTensorFromHandle remap table failed: " + TOSTR(s));

        // Fill backing buffer following declared strides
        float* fptr = static_cast<float*>(*backing_ptr);
        for (vx_size n = 0; n < N; ++n) {
            const float* src = replicate ? vec.data() : (vec.data() + n * elems_per_sample);
            float *dst = fptr + n * elems_per_sample;
            std::memcpy(dst, src, elems_per_sample * sizeof(float));
        }
        *out_t = t;
    };

    // Create handle-backed row/col tables and fill data
    make_and_fill_map_handle(_row_remap_vec, &_row_tbl_ptr, &_row_tbl);
    make_and_fill_map_handle(_col_remap_vec, &_col_tbl_ptr, &_col_tbl);

    // Scalars for interpolation, layout and ROI
    vx_context vx_ctx = vxGetContext((vx_reference)_graph->get());
    vx_scalar interpolation_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &_interpolation_type);

    int input_layout  = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type      = static_cast<int>(_inputs[0]->info().roi_type());

    vx_scalar input_layout_vx  = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx      = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &roi_type);

    // Construct the RPP Remap node
    _node = vxExtRppRemap(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _row_tbl,
                          _col_tbl, interpolation_vx, input_layout_vx, output_layout_vx, roi_type_vx);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS) {
        THROW("Adding the remap (vxExtRppRemap) node failed: " + TOSTR(status));
    }
#else
    THROW("RemapNode: vxExtRppRemap requires amd_rpp version >= 3.1.4");
#endif
}

void RemapNode::update_node() {}

void RemapNode::init(const std::vector<float>& row_remap_vec,
                     const std::vector<float>& col_remap_vec,
                     ResizeInterpolationType interpolation_type) {
    _row_remap_vec = row_remap_vec;
    _col_remap_vec = col_remap_vec;
    _interpolation_type = static_cast<int>(interpolation_type);
}

RemapNode::~RemapNode() {
    if (_inputs.empty() || !_inputs[0]) return;
    auto mem_type = _inputs[0]->info().mem_type();

    if (_row_tbl) vxReleaseTensor(&_row_tbl);
    if (_col_tbl) vxReleaseTensor(&_col_tbl);

    if (mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
        if (_row_tbl_ptr)  {
            hipError_t err = hipHostFree(_row_tbl_ptr);
            if (err != hipSuccess)
                std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
        }
        if (_col_tbl_ptr) {
            hipError_t err = hipHostFree(_col_tbl_ptr);
            if (err != hipSuccess)
                std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
        }
#endif
    } else {
        if (_row_tbl_ptr) free(_row_tbl_ptr);
        if (_col_tbl_ptr) free(_col_tbl_ptr);
    }
    _row_tbl_ptr = _col_tbl_ptr = nullptr;
}
