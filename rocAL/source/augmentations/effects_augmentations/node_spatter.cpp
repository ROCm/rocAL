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

#include "augmentations/effects_augmentations/node_spatter.h"
#include "pipeline/exception.h"

SpatterNode::SpatterNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs),
      _red_param(COLOR_RANGE[0], COLOR_RANGE[1]),
      _green_param(COLOR_RANGE[0], COLOR_RANGE[1]),
      _blue_param(COLOR_RANGE[0], COLOR_RANGE[1]),
      _color_array(nullptr),
      _color({0, 0, 0}) {}

void SpatterNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    vx_context context = vxGetContext((vx_reference)_graph->get());

    // Create parameter arrays for random value generation
    _red_param.create_array(_graph, VX_TYPE_INT32, _batch_size);
    _green_param.create_array(_graph, VX_TYPE_INT32, _batch_size);
    _blue_param.create_array(_graph, VX_TYPE_INT32, _batch_size);

    // Create a single color array with 3 elements (R, G, B)
    _color_array = vxCreateArray(context, VX_TYPE_UINT8, 3);
    vxAddArrayItems(_color_array, 3, _color.data(), sizeof(vx_uint8));

    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(context, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(context, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(context, VX_TYPE_INT32, &roi_type);

    _node = vxExtRppSpatter(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(),
                            _color_array, input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the Spatter (vxExtRppSpatter) node failed: " + TOSTR(status));
#else
    THROW("SpatterNode: vxExtRppSpatter requires vx_rpp version >= 3.1.5");
#endif
}

void SpatterNode::init(uint8_t red, uint8_t green, uint8_t blue) {
    _red_param.set_param(static_cast<int>(red));
    _green_param.set_param(static_cast<int>(green));
    _blue_param.set_param(static_cast<int>(blue));
}

void SpatterNode::init(IntParam *red, IntParam *green, IntParam *blue) {
    _red_param.set_param(core(red));
    _green_param.set_param(core(green));
    _blue_param.set_param(core(blue));
}

void SpatterNode::update_node() {
    // Update the parameter arrays to generate new random values if using IntParam
    _red_param.update_array();
    _green_param.update_array();
    _blue_param.update_array();

    // Get the generated values from ParameterVX and update the color array
    _color[0] = static_cast<vx_uint8>(_red_param.default_value());
    _color[1] = static_cast<vx_uint8>(_green_param.default_value());
    _color[2] = static_cast<vx_uint8>(_blue_param.default_value());

    vxCopyArrayRange(_color_array, 0, 3, sizeof(vx_uint8), _color.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
}
