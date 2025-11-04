/*
Copyright (c) 2025 Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "augmentations/geometry_augmentations/node_ricap.h"

#include <vx_ext_rpp.h>
#include <vx_ext_amd.h>
#include "pipeline/exception.h"

RicapNode::RicapNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
    : Node(inputs, outputs) {}

void RicapNode::create_node() {
    if (_node)
        return;

    // Determine effective batch (sequence-aware)
    const auto& dims_in = _inputs[0]->info().dims();
    if (dims_in.empty())
        THROW("Invalid input dims for Ricap");
    vx_size N = static_cast<vx_size>(dims_in[0]);
    auto layout = _inputs[0]->info().layout();
    if (layout == RocalTensorlayout::NFCHW || layout == RocalTensorlayout::NFHWC) {
        if (dims_in.size() < 2) THROW("Invalid sequence dims for Ricap");
        N = static_cast<vx_size>(dims_in[0] * dims_in[1]);
    }

    // Validate inputs
    if (_permutation_vec.empty())
        THROW("Ricap requires non-empty permutation vector of length 4 or N*4");
    if (_crop_rois_vec.empty())
        THROW("Ricap requires non-empty crop_rois vector of length 16 or N*16");

    // Determine mem type and VX mem
    auto mem_type = _inputs[0]->info().mem_type();
    vx_enum vx_mem = (mem_type == RocalMemType::HIP) ? VX_MEMORY_TYPE_HIP : VX_MEMORY_TYPE_HOST;

    // 1) Create permutation vx_array (length = N*4, replicate if needed)
    {
        std::vector<vx_uint32> perm(N * 4, 0);
        if (_permutation_vec.size() == 4) {
            for (vx_size n = 0; n < N; ++n) {
                for (int k = 0; k < 4; ++k)
                    perm[n * 4 + k] = static_cast<vx_uint32>(_permutation_vec[k]);
            }
        } else if (_permutation_vec.size() == N * 4) {
            for (vx_size i = 0; i < N * 4; ++i)
                perm[i] = static_cast<vx_uint32>(_permutation_vec[i]);
        } else {
            THROW("Ricap permutation vector size must be 4 or N*4. Got " + TOSTR(_permutation_vec.size()) + ", expected " + TOSTR(N * 4));
        }

        _perm_array_vx = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, perm.size());
        if (!_perm_array_vx) THROW("vxCreateArray for permutation failed");
        vx_status s = vxGetStatus((vx_reference)_perm_array_vx);
        if (s != VX_SUCCESS) THROW("Permutation array creation failed: " + TOSTR(s));
        s = vxAddArrayItems(_perm_array_vx, perm.size(), perm.data(), sizeof(vx_uint32));
        if (s != VX_SUCCESS) THROW("vxAddArrayItems for permutation failed: " + TOSTR(s));
    }

    // 2) Create crop-roi tensor (dims [N, 16], 4 ROIs per sample x 4 ints per ROI), replicate if vector has only 16
    {
        const vx_size elems_per_sample = 16;  // 4 ROIs x 4 ints
        const bool replicate = (_crop_rois_vec.size() == elems_per_sample);
        const size_t total_expected = static_cast<size_t>(N) * elems_per_sample;
        if (!(replicate || _crop_rois_vec.size() == total_expected)) {
            THROW("Ricap crop_rois vector size mismatch. Expected " + TOSTR(total_expected) + " or 16, got " + TOSTR(_crop_rois_vec.size()));
        }

        vx_size dims[2] = {N, elems_per_sample};
        vx_size stride[2];
        stride[0] = sizeof(vx_int32);
        stride[1] = stride[0] * dims[0];

        // Allocate backing buffer and create tensor from handle
        size_t bytes = stride[1] * dims[1];
        allocate_host_or_pinned_mem(&_crop_rois_ptr, bytes, mem_type);

        vx_tensor tensor_vx = vxCreateTensorFromHandle(vxGetContext((vx_reference)_graph->get()),
                                                       2, dims, VX_TYPE_INT32, 0, stride, _crop_rois_ptr, vx_mem);
        if (!tensor_vx) THROW("vxCreateTensorFromHandle for ricap ROI tensor failed");
        vx_status s = vxGetStatus((vx_reference)tensor_vx);
        if (s != VX_SUCCESS) THROW("vxCreateTensorFromHandle ROI failed: " + TOSTR(s));

        // Fill backing buffer following the declared strides
        int* iptr = static_cast<int*>(_crop_rois_ptr);
        for (vx_size n = 0; n < N; ++n) {
            const int* src = replicate ? _crop_rois_vec.data() : (&_crop_rois_vec[n * elems_per_sample]);
            for (vx_size k = 0; k < elems_per_sample; ++k) {
                iptr[n * elems_per_sample + k] = src[k];
            }
        }
        _crop_rois_t = tensor_vx;
    }

    // 3) Prepare scalars
    vx_context vx_ctx = vxGetContext((vx_reference)_graph->get());
    int input_layout  = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type      = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx  = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx      = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &roi_type);

    // 4) Create VX node
    _node = vxExtRppRicap(_graph->get(),
                          _inputs[0]->handle(),
                          _outputs[0]->handle(),
                          _perm_array_vx,
                          _crop_rois_t,
                          input_layout_vx,
                          output_layout_vx,
                          roi_type_vx);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS) {
        THROW("Adding the ricap (vxExtRppRicap) node failed: " + TOSTR(status));
    }
}

void RicapNode::update_node() {
    // No dynamic attributes to update per frame
}

void RicapNode::init(const std::vector<unsigned>& permutation,
                     const std::vector<int>& crop_rois) {
    _permutation_vec = permutation;
    _crop_rois_vec = crop_rois;
}

RicapNode::~RicapNode() {
    if (_inputs.empty() || !_inputs[0]) return;
    auto mem_type = _inputs[0]->info().mem_type();

    if (_perm_array_vx) {
        vxReleaseArray(&_perm_array_vx);
        _perm_array_vx = nullptr;
    }
    if (_crop_rois_t) {
        vxReleaseTensor(&_crop_rois_t);
        _crop_rois_t = nullptr;
    }

    if (mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
        if (_crop_rois_ptr)  {
            hipError_t err = hipHostFree(_crop_rois_ptr);
            if (err != hipSuccess)
                std::cerr << "\n[ERR] hipHostFree failed " << std::to_string(err) << "\n";
        }
#endif
    } else {
        if (_crop_rois_ptr) free(_crop_rois_ptr);
    }
    _crop_rois_ptr = nullptr;
}
