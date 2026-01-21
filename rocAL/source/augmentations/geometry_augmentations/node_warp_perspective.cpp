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

#include "augmentations/geometry_augmentations/node_warp_perspective.h"

#include <vx_ext_rpp.h>
#include <vx_ext_rpp_version.h>

#include "pipeline/exception.h"

WarpPerspectiveNode::WarpPerspectiveNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
    : Node(inputs, outputs) {
}

void WarpPerspectiveNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 3)
    // Build and allocate perspective array
    const size_t expected_len = static_cast<size_t>(_batch_size) * 9;

    std::vector<float> data;
    data.resize(expected_len);

    if (_perspective.size() == 9) {
        // If a single 9-element matrix is provided, replicate across the batch.
        for (uint i = 0; i < _batch_size; ++i) {
            const size_t base = static_cast<size_t>(i) * 9;
            for (int k = 0; k < 9; ++k) {
                data[base + k] = _perspective[k];
            }
        }
    } else if (_perspective.size() == expected_len) {
        // Already per-sample
        data = _perspective;
    } else {
        THROW("WarpPerspective: perspective matrix length must be 9 or 9 * batch_size, got: " + TOSTR(_perspective.size()));
    }

    // Create vx_array and populate it
    vx_status status;
    vx_context ctx = vxGetContext((vx_reference)_graph->get());
    vx_array perspective_array_vx = vxCreateArray(ctx, VX_TYPE_FLOAT32, expected_len);
    status = vxAddArrayItems(perspective_array_vx, expected_len, data.data(), sizeof(vx_float32));
    if (status != VX_SUCCESS) {
        vxReleaseArray(&perspective_array_vx);
        THROW("WarpPerspective: vxAddArrayItems failed while creating perspective array: " + TOSTR(status));
    }

    // Interpolation scalar
    vx_scalar interpolation_vx = vxCreateScalar(ctx, VX_TYPE_INT32, &_interpolation_type);

    // Layout and ROI type scalars
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());

    vx_scalar input_layout_vx = vxCreateScalar(ctx, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(ctx, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(ctx, VX_TYPE_INT32, &roi_type);

    // Construct the RPP WarpPerspective node
    _node = vxExtRppWarpPerspective(_graph->get(),
                                    _inputs[0]->handle(),
                                    _inputs[0]->get_roi_tensor(),
                                    _outputs[0]->handle(),
                                    perspective_array_vx,
                                    interpolation_vx,
                                    input_layout_vx,
                                    output_layout_vx,
                                    roi_type_vx);

    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS) {
        THROW("Adding the warp perspective (vxExtRppWarpPerspective) node failed: " + TOSTR(status));
    }
#else
    THROW("WarpPerspectiveNode: vxExtRppWarpPerspective requires amd_rpp version >= 3.1.3");
#endif
}

void WarpPerspectiveNode::init(const std::vector<float>& perspective_matrix, ResizeInterpolationType interpolation_type) {
    _perspective = perspective_matrix;
    _interpolation_type = static_cast<int>(interpolation_type);
}

void WarpPerspectiveNode::update_node() {}
