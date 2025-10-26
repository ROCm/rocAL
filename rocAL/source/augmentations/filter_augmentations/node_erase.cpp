/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include "augmentations/filter_augmentations/node_erase.h"
#include "pipeline/exception.h"

EraseNode::EraseNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs),
      _num_boxes(NUM_BOXES_RANGE[0], NUM_BOXES_RANGE[1]) {}

void EraseNode::create_node() {
    if (_node)
        return;

    if (!_anchor || !_colors)
        THROW("Erase node requires non-null anchor and colors tensors")

    // Create per-sample array for number of boxes
    _num_boxes.create_array(_graph, VX_TYPE_INT32, _batch_size);

    // Tensor layout and ROI type
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());

    vx_context vx_ctx = vxGetContext((vx_reference)_graph->get());
    vx_scalar input_layout_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &roi_type);

    // Create Erase node via MIVisionX RPP extension
    _node = vxExtRppErase(_graph->get(),
                          _inputs[0]->handle(),
                          _inputs[0]->get_roi_tensor(),
                          _outputs[0]->handle(),
                          _anchor->handle(),
                          _colors->handle(),
                          _num_boxes.default_array(),
                          input_layout_vx,
                          output_layout_vx,
                          roi_type_vx);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the erase (vxExtRppErase) node failed: " + TOSTR(status))
}

void EraseNode::init(Tensor *anchor_box_info, Tensor *colors, int num_boxes_fixed) {
    _anchor = anchor_box_info;
    _colors = colors;
    _num_boxes.set_param(num_boxes_fixed);
}

void EraseNode::init(Tensor *anchor_box_info, Tensor *colors, IntParam *num_boxes_param) {
    _anchor = anchor_box_info;
    _colors = colors;
    _num_boxes.set_param(core(num_boxes_param));
}

void EraseNode::update_node() {
    _num_boxes.update_array();
}
