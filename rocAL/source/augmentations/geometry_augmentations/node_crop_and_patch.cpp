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

#include "augmentations/geometry_augmentations/node_crop_and_patch.h"

#include <vx_ext_rpp.h>
#include <vx_ext_rpp_version.h>
#include <vx_ext_amd.h>
#include "pipeline/exception.h"

CropAndPatchNode::CropAndPatchNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
    : Node(inputs, outputs) {}

void CropAndPatchNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 4)
    // Validate inputs
    if (_crop_roi_vec.empty() || _patch_roi_vec.empty())
        THROW("CropAndPatch node requires non-empty ROI vectors for dst, crop and patch")

    // Determine N (replicate sequence behavior from CropNode)
    const auto& dims_in = _inputs[0]->info().dims();
    if (dims_in.empty())
        THROW("Invalid input dims for CropAndPatch");
    vx_size N = static_cast<vx_size>(dims_in[0]);
    auto layout = _inputs[0]->info().layout();
    if (layout == RocalTensorlayout::NFCHW || layout == RocalTensorlayout::NFHWC) {
        if (dims_in.size() < 2) THROW("Invalid sequence dims for CropAndPatch");
        N = static_cast<vx_size>(dims_in[0] * dims_in[1]);
    }

    const vx_size elems_per_sample = 4;

    // Select memory type and VX mem type
    auto mem_type = _inputs[0]->info().mem_type();
    vx_enum vx_mem = (mem_type == RocalMemType::HIP) ? VX_MEMORY_TYPE_HIP : VX_MEMORY_TYPE_HOST;

    auto make_and_fill_roi_tensor = [&](const std::vector<int>& vec, void** backing_ptr, vx_tensor* out_t) {
        const bool replicate = (vec.size() == elems_per_sample);
        const size_t total_expected = static_cast<size_t>(N) * elems_per_sample;
        if (!(replicate || vec.size() == total_expected))
            THROW("ROI vector size mismatch. Expected " + TOSTR(total_expected) + " or 4, got " + TOSTR(vec.size()));

        // Dims [N,4], strides matching node_crop: stride[0]=elem, stride[1]=elem*N
        vx_size dims[2] = {N, elems_per_sample};
        vx_size stride[2];
        stride[0] = sizeof(vx_int32);
        stride[1] = stride[0] * dims[0];

        // Allocate backing buffer and create tensor from handle
        size_t bytes = stride[1] * dims[1];
        allocate_host_or_pinned_mem(backing_ptr, bytes, mem_type);

        vx_tensor tensor_vx = vxCreateTensorFromHandle(vxGetContext((vx_reference)_graph->get()),
                                               2, dims, VX_TYPE_INT32, 0, stride, *backing_ptr, vx_mem);
        if (!tensor_vx) THROW("vxCreateTensorFromHandle for ROI tensor failed");
        vx_status s = vxGetStatus((vx_reference)tensor_vx);
        if (s != VX_SUCCESS) THROW("vxCreateTensorFromHandle ROI failed: " + TOSTR(s));

        // Fill backing buffer following the declared strides
        int* iptr = static_cast<int*>(*backing_ptr);
        // size_t sC = stride[1] / sizeof(int);   // N
        for (vx_size n = 0; n < N; ++n) {
            const int* src = replicate ? vec.data() : (vec.data() + n * elems_per_sample);
            for (vx_size k = 0; k < elems_per_sample; ++k) {
                iptr[n * elems_per_sample + k] = src[k];
            }
        }
        *out_t = tensor_vx;
    };

    // Create handle-backed ROI tensors and fill data
    make_and_fill_roi_tensor(_crop_roi_vec,  &_crop_roi_ptr,  &_crop_roi_t);
    make_and_fill_roi_tensor(_patch_roi_vec, &_patch_roi_ptr, &_patch_roi_t);

    // Layout and ROI scalars
    vx_context vx_ctx = vxGetContext((vx_reference)_graph->get());

    int input_layout  = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type      = static_cast<int>(_inputs[0]->info().roi_type());

    vx_scalar input_layout_vx  = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx      = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &roi_type);

    // Construct the RPP CropAndPatch node
    _node = vxExtRppCropAndPatch(_graph->get(),
                                 _inputs[0]->handle(),
                                 _inputs[1]->handle(),
                                 _outputs[0]->handle(),
                                 _inputs[1]->get_roi_tensor(),
                                 _crop_roi_t,
                                 _patch_roi_t,
                                 input_layout_vx,
                                 output_layout_vx,
                                 roi_type_vx);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS) {
        THROW("Adding the crop_and_patch (vxExtRppCropAndPatch) node failed: " + TOSTR(status));
    }
#else
    THROW("CropAndPatchNode: vxExtRppCropAndPatch requires amd_rpp version >= 3.1.4");
#endif
}

void CropAndPatchNode::update_node() {}

void CropAndPatchNode::init(const std::vector<int>& crop_roi_vec,
                            const std::vector<int>& patch_roi_vec) {
    _crop_roi_vec  = crop_roi_vec;
    _patch_roi_vec = patch_roi_vec;
}

CropAndPatchNode::~CropAndPatchNode() {
    if (_inputs.empty() || !_inputs[0]) return;
    auto mem_type = _inputs[0]->info().mem_type();
    if (_crop_roi_t) vxReleaseTensor(&_crop_roi_t);
    if (_patch_roi_t) vxReleaseTensor(&_patch_roi_t);

    if (mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
        if (_crop_roi_ptr)  {
            hipError_t err = hipHostFree(_crop_roi_ptr);
            if (err != hipSuccess)
                std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
        }
        if (_patch_roi_ptr)  {
            hipError_t err = hipHostFree(_patch_roi_ptr);
            if (err != hipSuccess)
                std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
        }
#endif
    } else {
        if (_crop_roi_ptr) free(_crop_roi_ptr);
        if (_patch_roi_ptr) free(_patch_roi_ptr);
    }
    _crop_roi_ptr = _patch_roi_ptr = nullptr;
}
