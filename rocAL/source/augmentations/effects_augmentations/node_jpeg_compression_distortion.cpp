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
#include "augmentations/effects_augmentations/node_jpeg_compression_distortion.h"
#include "pipeline/exception.h"

JpegCompressionDistortionNode::JpegCompressionDistortionNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs),
                                                                                                                                           _quality(QUALITY_RANGE[0], QUALITY_RANGE[1]) {}

void JpegCompressionDistortionNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    _quality.create_array(_graph, VX_TYPE_INT32, _batch_size);
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppJpegCompressionDistortion(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _quality.default_array(), input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the jpeg compression distortion (vxExtRppJpegCompressionDistortion) node failed: " + TOSTR(status))
#else
    THROW("JpegCompressionDistortionNode: vxExtRppJpegCompressionDistortion requires vx_rpp version >= 3.1.6");
#endif
}

void JpegCompressionDistortionNode::init(int quality) {
    _quality.set_param(quality);
}

void JpegCompressionDistortionNode::init(IntParam *quality_param) {
    _quality.set_param(core(quality_param));
}

void JpegCompressionDistortionNode::update_node() {
    _quality.update_array();
}
