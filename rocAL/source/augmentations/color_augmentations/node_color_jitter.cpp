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

#include "augmentations/color_augmentations/node_color_jitter.h"
#include "pipeline/exception.h"

ColorJitterNode::ColorJitterNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs),
      _brightness(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1]),
      _contrast(CONTRAST_RANGE[0], CONTRAST_RANGE[1]),
      _hue(HUE_RANGE[0], HUE_RANGE[1]),
      _saturation(SATURATION_RANGE[0], SATURATION_RANGE[1]) {}

void ColorJitterNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    _brightness.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _contrast.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _hue.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);
    _saturation.create_array(_graph, VX_TYPE_FLOAT32, _batch_size);

    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppColorJitter(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(),
                                 _brightness.default_array(), _contrast.default_array(),
                                 _hue.default_array(), _saturation.default_array(),
                                 input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the ColorJitter (vxExtRppColorJitter) node failed: " + TOSTR(status));
#else
    THROW("ColorJitterNode: vxExtRppColorJitter requires vx_rpp version >= 3.1.5");
#endif
}

void ColorJitterNode::init(float brightness, float contrast, float hue, float saturation) {
    _brightness.set_param(brightness);
    _contrast.set_param(contrast);
    _hue.set_param(hue);
    _saturation.set_param(saturation);
}

void ColorJitterNode::init(FloatParam *brightness_param, FloatParam *contrast_param, FloatParam *hue_param, FloatParam *saturation_param) {
    _brightness.set_param(core(brightness_param));
    _contrast.set_param(core(contrast_param));
    _hue.set_param(core(hue_param));
    _saturation.set_param(core(saturation_param));
}

void ColorJitterNode::update_node() {
    _brightness.update_array();
    _contrast.update_array();
    _hue.update_array();
    _saturation.update_array();
}
