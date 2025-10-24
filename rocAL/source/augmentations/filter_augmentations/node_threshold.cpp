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
#include "augmentations/filter_augmentations/node_threshold.h"
#include "pipeline/exception.h"

ThresholdNode::ThresholdNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs),
      _min(THRESHOLD_MIN_RANGE[0], THRESHOLD_MIN_RANGE[1]),
      _max(THRESHOLD_MAX_RANGE[0], THRESHOLD_MAX_RANGE[1]) {}

void ThresholdNode::create_node() {
    if (_node)
        return;

    // Create per-sample arrays for min and max threshold values
    _min.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _max.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);

    // Tensor layout and ROI type
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());

    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    // Create Threshold node via MIVisionX RPP extension
    _node = vxExtRppThreshold(_graph->get(),
                              _inputs[0]->handle(),
                              _inputs[0]->get_roi_tensor(),
                              _outputs[0]->handle(),
                              _min.default_array(),
                              _max.default_array(),
                              input_layout_vx,
                              output_layout_vx,
                              roi_type_vx);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the threshold (vxExtRppThreshold) node failed: " + TOSTR(status))
}

void ThresholdNode::init(float min_val, float max_val) {
    _min.set_param(min_val);
    _max.set_param(max_val);
}

void ThresholdNode::init(FloatParam *min_param, FloatParam *max_param) {
    _min.set_param(core(min_param));
    _max.set_param(core(max_param));
}

void ThresholdNode::update_node() {
    _min.update_array();
    _max.update_array();
}
