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

#include <vx_ext_rpp.h>
#include <vx_ext_rpp_version.h>
#include <algorithm>
#include "augmentations/color_augmentations/node_color_cast.h"
#include "pipeline/exception.h"
#include "pipeline/tensor.h"

static void fill_rgb_for_batch(std::vector<float> &rgb_out, unsigned batch_size, const std::vector<float> &rgb_in) {
    rgb_out.resize(batch_size * 3);
    if (rgb_in.size() == 3) {
        // Replicate a single triplet across the batch
        for (unsigned i = 0; i < batch_size; ++i) {
            unsigned base = i * 3;
            rgb_out[base + 0] = rgb_in[0];
            rgb_out[base + 1] = rgb_in[1];
            rgb_out[base + 2] = rgb_in[2];
        }
    } else if (rgb_in.size() == batch_size * 3) {
        // Copy per-sample triplets
        rgb_out = rgb_in;
    } else {
        // Invalid size - fail fast instead of silently defaulting to zeros
        THROW("ColorCast: Invalid RGB array size. Expected 3 (single triplet) or " + 
              std::to_string(batch_size * 3) + " (per-sample triplets), got " + 
              std::to_string(rgb_in.size()));
    }
}

ColorCastNode::ColorCastNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs),
      _alpha(ALPHA_RANGE[0], ALPHA_RANGE[1]) {}

void ColorCastNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 2)
    // Create per-sample arrays
    _alpha.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);

    // Create vx_tensor for the RGB values
    const vx_size num_of_dims = 2;
    vx_size stride[num_of_dims];
    std::vector<size_t> rgb_tensor_dims = {_batch_size, 3};
    
    // Calculate strides for uint8 data type
    stride[0] = sizeof(vx_uint8);
    stride[1] = stride[0] * rgb_tensor_dims[0];
    
    // Determine memory type based on input tensor
    vx_enum mem_type = VX_MEMORY_TYPE_HOST;
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP)
        mem_type = VX_MEMORY_TYPE_HIP;
    
    // Allocate pinned memory for RGB values
    size_t rgb_buffer_size = rgb_tensor_dims[1] * stride[1];
    allocate_host_or_pinned_mem(&_rgb_memory, rgb_buffer_size, _inputs[0]->info().mem_type());
    
    // Convert float RGB values to uint8 and store in the allocated memory
    vx_uint8* rgb_uint8_ptr = static_cast<vx_uint8*>(_rgb_memory);
    for (size_t i = 0; i < _rgb.size(); ++i) {
        rgb_uint8_ptr[i] = static_cast<vx_uint8>(_rgb[i]);
    }

    // Create tensor from the allocated memory
    _rgb_tensor = vxCreateTensorFromHandle(vxGetContext((vx_reference)_graph->get()), num_of_dims, 
                                           rgb_tensor_dims.data(), VX_TYPE_UINT8, 0,
                                           stride, _rgb_memory, mem_type);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_rgb_tensor)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(_rgb_tensor) failed: " + TOSTR(status));

    // Layouts & ROI type
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppColorCast(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(),
                              _rgb_tensor, _alpha.default_array(), input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status nstatus;
    if ((nstatus = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the ColorCast (vxExtRppColorCast) node failed: " + TOSTR(nstatus));
#else
    THROW("ColorCastNode: vxExtRppColorCast requires amd_rpp version >= 3.1.2");
#endif
}

void ColorCastNode::init(FloatParam *alpha_param, std::vector<float> rgb) {
    _alpha.set_param(core(alpha_param));
    fill_rgb_for_batch(_rgb, _batch_size, rgb);
}

void ColorCastNode::init(float alpha, std::vector<float> rgb) {
    _alpha.set_param(alpha);
    fill_rgb_for_batch(_rgb, _batch_size, rgb);
}

void ColorCastNode::update_node() {
    _alpha.update_array();
}

ColorCastNode::~ColorCastNode() {
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP) {
#if ENABLE_HIP
        hipError_t err = hipHostFree(_rgb_memory);
        if (err != hipSuccess)
            std::cerr << "\n[ERR] hipHostFree failed  " << std::to_string(err) << "\n";
#endif
    } else {
        if (_rgb_memory) free(_rgb_memory);
    }
    if (_rgb_tensor) vxReleaseTensor(&_rgb_tensor);
}
