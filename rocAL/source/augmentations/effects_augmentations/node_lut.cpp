/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <vx_ext_rpp.h>
#include <vx_ext_rpp_version.h>
#include "augmentations/effects_augmentations/node_lut.h"
#include "pipeline/exception.h"
#include "pipeline/tensor.h"

LutNode::LutNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void LutNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    create_lut_tensor();  // This allocates and initializes the LUT buffer
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppLut(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(),
                        _lut_tensor, input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the LUT (vxExtRppLut) node failed: " + TOSTR(status))
#else
    THROW("LutNode: vxExtRppLut requires vx_rpp version >= 3.1.6");
#endif
}

void LutNode::update_node() {
    // No parameters to update for LUT
}

void LutNode::create_lut_tensor() {
    const vx_size num_of_dims = 1;
    vx_size stride[num_of_dims];
    std::vector<size_t> lut_tensor_dims = {_lut_size};
    
    // Determine data type based on input tensor data type
    vx_enum data_type = VX_TYPE_UINT8;
    RocalTensorDataType tensor_dtype = _inputs[0]->info().data_type();
    
    if (tensor_dtype == RocalTensorDataType::FP16)
        data_type = VX_TYPE_FLOAT16;
    else if (tensor_dtype == RocalTensorDataType::FP32)
        data_type = VX_TYPE_FLOAT32;
    else if (tensor_dtype == RocalTensorDataType::INT8)
        data_type = VX_TYPE_INT8;
    else
        data_type = VX_TYPE_UINT8;
    
    size_t element_size = (data_type == VX_TYPE_UINT8 || data_type == VX_TYPE_INT8) ? sizeof(uint8_t) : 
                          (data_type == VX_TYPE_FLOAT16) ? sizeof(uint16_t) : sizeof(float);
    
    stride[0] = element_size;
    
    vx_enum mem_type = VX_MEMORY_TYPE_HOST;
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP)
        mem_type = VX_MEMORY_TYPE_HIP;
    
    allocate_host_or_pinned_mem(&_lut_buffer, _lut_size * element_size, _inputs[0]->info().mem_type());

    // Initialize LUT buffer before creating VX tensor
    init_lut_buffer();

    _lut_tensor = vxCreateTensorFromHandle(vxGetContext((vx_reference)_graph->get()), num_of_dims, lut_tensor_dims.data(), data_type, 0,
                                           stride, reinterpret_cast<void *>(_lut_buffer), mem_type);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_lut_tensor)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(_lut_tensor) failed " + TOSTR(status))
}

void LutNode::init_lut_buffer() {
    // Initialize LUT buffer with identity transformation (can be customized by user)
    // For now, creating an inverted LUT similar to RPP test suite
    RocalTensorDataType tensor_dtype = _inputs[0]->info().data_type();
    
    if (tensor_dtype == RocalTensorDataType::UINT8) {
        uint8_t *lut8u = static_cast<uint8_t *>(_lut_buffer);
        for (size_t i = 0; i < _lut_size; i++)
            lut8u[i] = static_cast<uint8_t>(255 - i);  // Inverted LUT
    } else if (tensor_dtype == RocalTensorDataType::FP16) {
        vx_float16 *lut16f = static_cast<vx_float16 *>(_lut_buffer);
        for (size_t i = 0; i < _lut_size; i++) {
            float temp_val = (255.0f - static_cast<float>(i)) / 255.0f;
            lut16f[i] = static_cast<vx_float16>(temp_val);
        }
    } else if (tensor_dtype == RocalTensorDataType::FP32) {
        float *lut32f = static_cast<float *>(_lut_buffer);
        for (size_t i = 0; i < _lut_size; i++)
            lut32f[i] = (255 - i) / 255.0f;
    } else if (tensor_dtype == RocalTensorDataType::INT8) {
        int8_t *lut8s = static_cast<int8_t *>(_lut_buffer);
        for (size_t i = 0; i < _lut_size; i++)
            lut8s[i] = static_cast<int8_t>(127 - (i - 128));  // Inverted LUT for signed values
    }
}

LutNode::~LutNode() {
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP) {
#if ENABLE_HIP
        hipError_t err = hipHostFree(_lut_buffer);
        if (err != hipSuccess)
            std::cerr << "\n[ERR] hipHostFree failed  " << std::to_string(err) << "\n";
#endif
    } else {
        if (_lut_buffer) free(_lut_buffer);
    }
    if (_lut_tensor) vxReleaseTensor(&_lut_tensor);
}
