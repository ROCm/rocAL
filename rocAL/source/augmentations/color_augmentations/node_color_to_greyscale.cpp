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

#include "augmentations/color_augmentations/node_color_to_greyscale.h"
#include "pipeline/exception.h"

ColorToGreyscaleNode::ColorToGreyscaleNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs), _layout(SubpixelLayout::RGB) {}

void ColorToGreyscaleNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    vx_context context = vxGetContext((vx_reference)_graph->get());
    int subpixel = static_cast<int>(_layout);
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar subpixel_vx = vxCreateScalar(context, VX_TYPE_INT32, &subpixel);
    vx_scalar input_layout_vx = vxCreateScalar(context, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(context, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(context, VX_TYPE_INT32, &roi_type);

    _node = vxExtRppColorToGreyscale(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(),
                                      subpixel_vx, input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the ColorToGreyscale (vxExtRppColorToGreyscale) node failed: " + TOSTR(status));
#else
    THROW("ColorToGreyscaleNode: vxExtRppColorToGreyscale requires vx_rpp version >= 3.1.6");
#endif
}

void ColorToGreyscaleNode::init(SubpixelLayout layout) {
    _layout = layout;
}
