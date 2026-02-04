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

#include "augmentations/geometry_augmentations/node_ricap.h"

#include <vx_ext_rpp.h>
#include <vx_ext_rpp_version.h>
#include <vx_ext_amd.h>
#include "pipeline/exception.h"

RicapNode::RicapNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
    : Node(inputs, outputs) {}

// Unified helper function to create and populate a tensor with optional replication
template<typename T>
vx_tensor RicapNode::create_tensor_with_replication(
    const std::vector<T>& input_vec,
    vx_size N,
    vx_size elems_per_sample,
    void** backing_ptr,
    RocalMemType mem_type,
    vx_enum vx_data_type,
    std::vector<vx_size>&& dims) {
    
    // Calculate total elements needed
    const size_t total_elements = N * elems_per_sample;
    const bool needs_replication = (input_vec.size() == elems_per_sample);
    
    // Validate input size
    if (!(needs_replication || input_vec.size() == total_elements)) {
        THROW("Tensor data size mismatch. Expected " + TOSTR(total_elements) + 
              " or " + TOSTR(elems_per_sample) + ", got " + TOSTR(input_vec.size()));
    }
    
    // Setup dimensions and strides based on number of dimensions
    std::vector<vx_size> strides(dims.size(), 0);
    auto num_dims = dims.size();
    strides[0] = sizeof(T);
    for (int i = 1; i < dims.size(); i++) {
        strides[i] = strides[i - 1] * dims[i - 1];
    }

    // Allocate backing buffer
    size_t bytes = sizeof(T) * total_elements;
    allocate_host_or_pinned_mem(backing_ptr, bytes, mem_type);
    
    // Fill backing buffer with data (with replication if needed)
    T* ptr = static_cast<T*>(*backing_ptr);
    if (needs_replication) {
        // Replicate the data for each sample in the batch
        for (vx_size n = 0; n < N; ++n) {
            std::copy(input_vec.begin(), input_vec.end(), ptr + n * elems_per_sample);
        }
    } else {
        // Direct copy of all data
        std::copy(input_vec.begin(), input_vec.end(), ptr);
    }
    
    // Create tensor from handle
    vx_context ctx = vxGetContext((vx_reference)_graph->get());
    vx_enum vx_mem = (mem_type == RocalMemType::HIP) ? VX_MEMORY_TYPE_HIP : VX_MEMORY_TYPE_HOST;
    vx_tensor tensor = vxCreateTensorFromHandle(ctx, num_dims, dims.data(), vx_data_type, 0, strides.data(), *backing_ptr, vx_mem);
    
    if (!tensor) THROW("vxCreateTensorFromHandle failed");
    
    vx_status status = vxGetStatus((vx_reference)tensor);
    if (status != VX_SUCCESS) THROW("vxCreateTensorFromHandle failed: " + TOSTR(status));
    
    return tensor;
}

void RicapNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 4)
    vx_size N = static_cast<vx_size>(_inputs[0]->info().dims()[0]);
    auto layout = _inputs[0]->info().layout();
    
    // Check for unsupported layouts
    if (layout == RocalTensorlayout::NFCHW || layout == RocalTensorlayout::NFHWC) {
        THROW("NFHWC and NFCHW types are unsupported for Ricap augmentation");
    }

    // Validate inputs
    if (_permutation_vec.empty())
        THROW("Ricap requires non-empty permutation vector of length 4 or N*4");
    if (_crop_rois_vec.empty())
        THROW("Ricap requires non-empty crop_rois vector of length 16 or N*16");

    // Determine memory type
    auto mem_type = _inputs[0]->info().mem_type();
    
    
    // Create permutation tensor (1D tensor with 4 elements per sample)
    _perm_tensor_vx = create_tensor_with_replication<uint32_t>(
        _permutation_vec, N, 4, &_perm_ptr, mem_type, VX_TYPE_UINT32, {N, 4});
    
    // Create crop ROI tensor (2D tensor with 16 elements per sample: 4 ROIs x 4 values)
    _crop_rois_t = create_tensor_with_replication<int>(
        _crop_rois_vec, N, 16, &_crop_rois_ptr, mem_type, VX_TYPE_INT32, {N, 16});
    // Create scalars for layout and ROI type
    vx_context vx_ctx = vxGetContext((vx_reference)_graph->get());
    int input_layout  = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type      = static_cast<int>(_inputs[0]->info().roi_type());
    
    vx_scalar input_layout_vx  = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx      = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &roi_type);

    // Create the Ricap node
    _node = vxExtRppRicap(_graph->get(),
                          _inputs[0]->handle(),
                          _outputs[0]->handle(),
                          _perm_tensor_vx,
                          _crop_rois_t,
                          input_layout_vx,
                          output_layout_vx,
                          roi_type_vx);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS) {
        THROW("Adding the ricap (vxExtRppRicap) node failed: " + TOSTR(status));
    }
#else
    THROW("RicapNode: vxExtRppRicap requires amd_rpp version >= 3.1.4");
#endif
}

void RicapNode::init(const std::vector<unsigned>& permutation,
                     const std::vector<int>& crop_rois) {
    _permutation_vec = permutation;
    _crop_rois_vec = crop_rois;
}

void RicapNode::update_node() {}

RicapNode::~RicapNode() {
    if (_inputs.empty() || !_inputs[0]) return;
    auto mem_type = _inputs[0]->info().mem_type();

    if (_perm_tensor_vx) {
        vxReleaseTensor(&_perm_tensor_vx);
        _perm_tensor_vx = nullptr;
    }
    if (_crop_rois_t) {
        vxReleaseTensor(&_crop_rois_t);
        _crop_rois_t = nullptr;
    }

    if (mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
        if (_perm_ptr) {
            hipError_t err = hipHostFree(_perm_ptr);
            if (err != hipSuccess)
                std::cerr << "\n[ERR] hipHostFree failed for perm_ptr: " << std::to_string(err) << "\n";
        }
        if (_crop_rois_ptr) {
            hipError_t err = hipHostFree(_crop_rois_ptr);
            if (err != hipSuccess)
                std::cerr << "\n[ERR] hipHostFree failed for crop_rois_ptr: " << std::to_string(err) << "\n";
        }
#endif
    } else {
        if (_perm_ptr) free(_perm_ptr);
        if (_crop_rois_ptr) free(_crop_rois_ptr);
    }
    _perm_ptr = nullptr;
    _crop_rois_ptr = nullptr;
}
