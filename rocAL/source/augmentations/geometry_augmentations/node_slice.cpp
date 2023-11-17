/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "node_slice.h"

#include <vx_ext_rpp.h>

#include "exception.h"

SliceNode::SliceNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void SliceNode::create_node() {
    if (_node)
        return;

    create_shape_tensor();
    auto max_shape = _outputs[0]->info().max_shape();
    _slice_roi.resize(_batch_size, std::vector<uint32_t>(max_shape.size()));
    for (uint i = 0; i < _batch_size; i++)
        for (uint j = 0; j < max_shape.size(); j++)
            _slice_roi[i][j] = max_shape[j];
    const int buffer_size = _batch_size;
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);
    _fill_values_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, buffer_size);
    vx_status status = vxAddArrayItems(_fill_values_array, buffer_size, _fill_values_vec.data(), sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the slice (vxExtRppSlice) node: " + TOSTR(status));
    vx_scalar policy = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_policy);
    _node = vxExtRppSlice(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _anchor->handle(),
                          _shape, _fill_values_array, policy, input_layout_vx, roi_type_vx);

    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the slice node (vxRppSlice) failed: " + TOSTR(status))
}

void SliceNode::update_node() {
    // if fill values passed by user less than what is required, replicate the values
    if (_fill_values.size() == 1) {
        std::fill(_fill_values_vec.begin(), _fill_values_vec.end(), _fill_values[0]);
    }
    vx_status status = VX_SUCCESS;
    _outputs[0]->update_tensor_roi(_slice_roi);
    status = vxCopyArrayRange((vx_array)_fill_values_array, 0, _batch_size, sizeof(vx_float32), _fill_values_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if (status != 0)
        WRN("ERROR: vxCopyArrayRange failed in the slice node (vxExtRppSlice) node: " + TOSTR(status))
    int* shape_arr = (int *) _shape_array;
    // replicate shape values for all samples in a batch
    for (uint i = 0; i < _batch_size; i++) {
        int sample_idx = i * _shape_vec.size();
        memcpy(&(shape_arr[sample_idx]), _shape_vec.data(), _shape_vec.size() * sizeof(int));
    }
}

void SliceNode::init(Tensor *anchor, std::vector<int> shape, std::vector<float> &fill_values, RocalOutOfBoundsPolicy policy) {
    _policy = policy;
    _anchor = anchor;
    _shape_vec = shape;
    _fill_values = fill_values;
    _fill_values_vec.resize(_batch_size);
}

// Create vx_tensor for the shape coordinates
void SliceNode::create_shape_tensor() {
    vx_size num_of_dims = 2;
    vx_size stride[num_of_dims];
    std::vector<size_t> _shape_tensor_dims = {_batch_size, _shape_vec.size()};
    stride[0] = sizeof(vx_int32);
    stride[1] = stride[0] * _shape_tensor_dims[0];
    vx_enum mem_type = VX_MEMORY_TYPE_HOST;
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP)
        mem_type = VX_MEMORY_TYPE_HIP;
    allocate_host_or_pinned_mem(&_shape_array, stride[1] * _shape_vec.size(), _inputs[0]->info().mem_type());

    _shape = vxCreateTensorFromHandle(vxGetContext((vx_reference)_graph->get()), num_of_dims, _shape_tensor_dims.data(), VX_TYPE_INT32, 0,
                                      stride, reinterpret_cast<void *>(_shape_array), mem_type);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_shape)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(_shape: failed " + TOSTR(status))
}

SliceNode::~SliceNode() {
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP) {
#if ENABLE_HIP
        hipError_t err = hipHostFree(_shape_array);
        if (err != hipSuccess)
            std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
#endif
    } else {
        if (_shape_array) free(_shape_array);
    }
    if (_shape) vxReleaseTensor(&_shape);
}
