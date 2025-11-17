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
#include "augmentations/color_augmentations/node_color_cast.h"
#include "pipeline/exception.h"

namespace {
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
        // Invalid size, default to zeros
        for (unsigned i = 0; i < batch_size; ++i) {
            unsigned base = i * 3;
            rgb_out[base + 0] = 0.0f;
            rgb_out[base + 1] = 0.0f;
            rgb_out[base + 2] = 0.0f;
        }
    }
}
}  // namespace

ColorCastNode::ColorCastNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs),
      _alpha(ALPHA_RANGE[0], ALPHA_RANGE[1]) {}

void ColorCastNode::create_node() {
    if (_node)
        return;

    // Create per-sample arrays
    _alpha.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);

    // Create and populate the RGB array (flat size = batch_size * 3)
    vx_status status = VX_SUCCESS;
    _rgb_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * 3);
    status |= vxAddArrayItems(_rgb_vx_array, _rgb.size(), _rgb.data(), sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the ColorCast (vxExtRppColorCast) node: " + TOSTR(status) + "  " + TOSTR(status))

    // Layouts & ROI type
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    // Build node
    _node = vxExtRppColorCast(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(),
                              _rgb_vx_array, _alpha.default_array(), input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status nstatus;
    if ((nstatus = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the ColorCast (vxExtRppColorCast) node failed: " + TOSTR(nstatus))
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
    // Update the RGB array content if present
    if (_rgb_vx_array) {
        vx_status status = VX_SUCCESS;
        status = vxCopyArrayRange(_rgb_vx_array, 0, _batch_size * 3, sizeof(vx_float32), _rgb.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != 0)
            THROW(" vxCopyArrayRange failed in update_node (ColorCast): " + TOSTR(status))
    }
}
