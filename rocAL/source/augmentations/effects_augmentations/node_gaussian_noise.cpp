/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#include "node_gaussian_noise.h"
#include "exception.h"

GaussianNoiseNode::GaussianNoiseNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) : Node(inputs, outputs),
                                                                                                      _mean(MEAN_RANGE[0], MEAN_RANGE[1]),
                                                                                                      _stddev(STDDEV_RANGE[0], STDDEV_RANGE[1]) {}

void GaussianNoiseNode::create_node() {
    if (_node)
        return;

    _mean.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _stddev.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    vx_scalar seed = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_seed);
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppGaussianNoise(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _mean.default_array(),
                          _stddev.default_array(), seed, input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the Noise (vxExtRppGaussianNoise) node failed: " + TOSTR(status))
}

void GaussianNoiseNode::init(float mean, float stddev, int seed) {
    _mean.set_param(mean);
    _stddev.set_param(stddev);
    _seed = seed;
}

void GaussianNoiseNode::init(FloatParam* mean_param, FloatParam* stddev_param, int seed) {
    _mean.set_param(core(mean_param));
    _stddev.set_param(core(stddev_param));
    _seed = seed;
}

void GaussianNoiseNode::update_node() {
    _mean.update_array();
    _stddev.update_array();
}
