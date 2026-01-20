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

#include "augmentations/effects_augmentations/node_grid_mask.h"

#include <vx_ext_rpp.h>
#include <vx_ext_rpp_version.h>

#include "pipeline/exception.h"

GridMaskNode::GridMaskNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs) {}

void GridMaskNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 2)
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());

    vx_context ctx = vxGetContext((vx_reference)_graph->get());

    vx_scalar tile_width_vx = vxCreateScalar(ctx, VX_TYPE_UINT32, &_tile_width);
    vx_scalar grid_ratio_vx = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &_grid_ratio);
    vx_scalar grid_angle_vx = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &_grid_angle);
    vx_scalar translate_x_vx = vxCreateScalar(ctx, VX_TYPE_UINT32, &_translate_x);
    vx_scalar translate_y_vx = vxCreateScalar(ctx, VX_TYPE_UINT32, &_translate_y);

    vx_scalar input_layout_vx = vxCreateScalar(ctx, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(ctx, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(ctx, VX_TYPE_INT32, &roi_type);

    _node = vxExtRppGridMask(_graph->get(),
                             _inputs[0]->handle(),
                             _inputs[0]->get_roi_tensor(),
                             _outputs[0]->handle(),
                             tile_width_vx,
                             grid_ratio_vx,
                             grid_angle_vx,
                             translate_x_vx,
                             translate_y_vx,
                             input_layout_vx,
                             output_layout_vx,
                             roi_type_vx);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the grid mask (vxExtRppGridMask) node failed: " + TOSTR(status))
#else
    THROW("GridMaskNode: vxExtRppGridMask requires amd_rpp version >= 3.1.2");
#endif
}

void GridMaskNode::init(unsigned tile_width, float grid_ratio, float grid_angle_radians, unsigned translate_x, unsigned translate_y) {
    _tile_width = tile_width;
    _grid_ratio = grid_ratio;
    _grid_angle = grid_angle_radians;
    _translate_x = translate_x;
    _translate_y = translate_y;
}

void GridMaskNode::update_node() {}
