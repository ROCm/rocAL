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
#include "augmentations/effects_augmentations/node_water.h"
#include "pipeline/exception.h"

WaterNode::WaterNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs),
                                                                                                   _amplitude_x(AMPLITUDE_RANGE[0], AMPLITUDE_RANGE[1]),
                                                                                                   _amplitude_y(AMPLITUDE_RANGE[0], AMPLITUDE_RANGE[1]),
                                                                                                   _frequency_x(FREQUENCY_RANGE[0], FREQUENCY_RANGE[1]),
                                                                                                   _frequency_y(FREQUENCY_RANGE[0], FREQUENCY_RANGE[1]),
                                                                                                   _phase_x(PHASE_RANGE[0], PHASE_RANGE[1]),
                                                                                                   _phase_y(PHASE_RANGE[0], PHASE_RANGE[1]) {}

void WaterNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    _amplitude_x.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _amplitude_y.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _frequency_x.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _frequency_y.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _phase_x.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _phase_y.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);

    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppWater(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(),
                          _amplitude_x.default_array(), _amplitude_y.default_array(),
                          _frequency_x.default_array(), _frequency_y.default_array(),
                          _phase_x.default_array(), _phase_y.default_array(),
                          input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the water (vxExtRppWater) node failed: " + TOSTR(status))
#else
    THROW("WaterNode: vxExtRppWater requires vx_rpp version >= 3.1.5");
#endif
}

void WaterNode::init(float amplitude_x, float amplitude_y, float frequency_x, float frequency_y, float phase_x, float phase_y) {
    _amplitude_x.set_param(amplitude_x);
    _amplitude_y.set_param(amplitude_y);
    _frequency_x.set_param(frequency_x);
    _frequency_y.set_param(frequency_y);
    _phase_x.set_param(phase_x);
    _phase_y.set_param(phase_y);
}

void WaterNode::init(FloatParam *amplitude_x_param, FloatParam *amplitude_y_param,
                     FloatParam *frequency_x_param, FloatParam *frequency_y_param,
                     FloatParam *phase_x_param, FloatParam *phase_y_param) {
    _amplitude_x.set_param(core(amplitude_x_param));
    _amplitude_y.set_param(core(amplitude_y_param));
    _frequency_x.set_param(core(frequency_x_param));
    _frequency_y.set_param(core(frequency_y_param));
    _phase_x.set_param(core(phase_x_param));
    _phase_y.set_param(core(phase_y_param));
}

void WaterNode::update_node() {
    _amplitude_x.update_array();
    _amplitude_y.update_array();
    _frequency_x.update_array();
    _frequency_y.update_array();
    _phase_x.update_array();
    _phase_y.update_array();
}
