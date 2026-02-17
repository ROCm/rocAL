/*
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "augmentations/geometry_augmentations/node_slice.h"

#include <algorithm>
#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

namespace {
void fill_values_buffer(std::vector<float> &dst, const std::vector<float> &src) {
    if (dst.empty())
        return;

    if (src.empty()) {
        std::fill(dst.begin(), dst.end(), 0.0f);
        return;
    }

    if (src.size() == 1) {
        std::fill(dst.begin(), dst.end(), src[0]);
        return;
    }

    const auto copy_count = std::min(dst.size(), src.size());
    std::copy_n(src.begin(), copy_count, dst.begin());
    if (copy_count < dst.size())
        std::fill(dst.begin() + copy_count, dst.end(), src[copy_count - 1]);
}
}  // namespace

SliceNode::SliceNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void SliceNode::create_node() {
    if (_node)
        return;

    vx_tensor shape_tensor = nullptr;
    if (_use_tensor_shape) {
        if (!_shape_tensor_param)
            THROW("Slice node expects a valid shape tensor when tensor-based API is used");
        shape_tensor = _shape_tensor_param->handle();
    } else {
        create_shape_tensor();
        auto max_shape = _outputs[0]->info().max_shape();
        _slice_roi.resize(_batch_size, std::vector<uint32_t>(max_shape.size()));
        for (uint i = 0; i < _batch_size; i++)
            for (uint j = 0; j < max_shape.size(); j++)
                _slice_roi[i][j] = max_shape[j];
        shape_tensor = _shape_tensor_handle;
    }

    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);
    _fill_values_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size);
    vx_status status = vxAddArrayItems(_fill_values_array, _batch_size, _fill_values_vec.data(), sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the slice (vxExtRppSlice) node: " + TOSTR(status));
    vx_scalar policy = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_policy);
    _node = vxExtRppSlice(_graph->get(),
                          _inputs[0]->handle(),
                          _inputs[0]->get_roi_tensor(),
                          _outputs[0]->handle(),
                          _anchor->handle(),
                          shape_tensor,
                          _fill_values_array,
                          policy,
                          input_layout_vx,
                          roi_type_vx);

    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the slice node (vxRppSlice) failed: " + TOSTR(status))
}

void SliceNode::update_node() {
    if (_use_tensor_shape)
        return;

    fill_values_buffer(_fill_values_vec, _fill_values);
    vx_status status = vxCopyArrayRange((vx_array)_fill_values_array, 0, _batch_size, sizeof(vx_float32), _fill_values_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if (status != 0)
        WRN("ERROR: vxCopyArrayRange failed in the slice node (vxExtRppSlice) node: " + TOSTR(status))
    _outputs[0]->update_tensor_roi(_slice_roi);

    auto *shape_arr = static_cast<int *>(_shape_array);
    for (uint i = 0; i < _batch_size; i++) {
        int sample_idx = i * _shape_vec.size();
        memcpy(&(shape_arr[sample_idx]), _shape_vec.data(), _shape_vec.size() * sizeof(int));
    }
}

void SliceNode::init(Tensor *anchor, std::vector<int> shape, std::vector<float> &fill_values, OutOfBoundsPolicy policy) {
    _policy = static_cast<OutOfBoundsPolicy>(policy);
    _anchor = anchor;
    _shape_vec = std::move(shape);
    _fill_values = fill_values;
    _fill_values_vec.resize(_batch_size);
    fill_values_buffer(_fill_values_vec, _fill_values);
    _use_tensor_shape = false;
}

void SliceNode::init(Tensor *anchor, Tensor *shape, std::vector<float> &fill_values, OutOfBoundsPolicy policy) {
    _policy = policy;
    _anchor = anchor;
    _shape_tensor_param = shape;
    _fill_values = fill_values;
    _fill_values_vec.resize(_batch_size);
    fill_values_buffer(_fill_values_vec, _fill_values);
    _use_tensor_shape = true;
}

// Create vx_tensor for the shape coordinates
void SliceNode::create_shape_tensor() {
    if (_use_tensor_shape)
        return;
    if (_shape_vec.empty())
        THROW("Slice node expects valid shape dimensions when using vector-based API");

    vx_size num_of_dims = 2;
    std::vector<vx_size> stride(num_of_dims);
    std::vector<size_t> _shape_tensor_dims = {_batch_size, _shape_vec.size()};
    stride[0] = sizeof(vx_int32);
    stride[1] = stride[0] * _shape_tensor_dims[0];
    vx_enum mem_type = VX_MEMORY_TYPE_HOST;
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP)
        mem_type = VX_MEMORY_TYPE_HIP;
    allocate_host_or_pinned_mem(&_shape_array, stride[1] * _shape_vec.size(), _inputs[0]->info().mem_type());

    _shape_tensor_handle = vxCreateTensorFromHandle(vxGetContext((vx_reference)_graph->get()), num_of_dims, _shape_tensor_dims.data(), VX_TYPE_INT32, 0,
                                                    stride.data(), reinterpret_cast<void *>(_shape_array), mem_type);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_shape_tensor_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(_shape: failed " + TOSTR(status))
}

SliceNode::~SliceNode() {
    if (_fill_values_array) vxReleaseArray(&_fill_values_array);

    if (_inputs[0]->info().mem_type() == RocalMemType::HIP) {
#if ENABLE_HIP
        if (_shape_array) {
            hipError_t err = hipHostFree(_shape_array);
            if (err != hipSuccess)
                std::cerr << "\n[ERR] hipHostFree failed  " << std::to_string(err) << "\n";
        }
#endif
    } else {
        if (_shape_array) free(_shape_array);
    }
    _shape_array = nullptr;
    if (_shape_tensor_handle) vxReleaseTensor(&_shape_tensor_handle);
}
