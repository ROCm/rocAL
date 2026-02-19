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
#include "augmentations/color_augmentations/node_channel_permute.h"
#include "pipeline/exception.h"

ChannelPermuteNode::ChannelPermuteNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void ChannelPermuteNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    // Create array for permutation tensor - needs to be batchSize * 3
    _permutation_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size * 3);

    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);

    _node = vxExtRppChannelPermute(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _permutation_array, input_layout_vx, output_layout_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the channel permute (vxExtRppChannelPermute) node failed: " + TOSTR(status))
#else
    THROW("ChannelPermuteNode: vxExtRppChannelPermute requires vx_rpp version >= 3.1.6");
#endif
}

void ChannelPermuteNode::init(const std::vector<unsigned> &permutation_order) {
    _permutation_order = permutation_order;
}

void ChannelPermuteNode::update_node() {
    if (_permutation_order.size() != 3)
        THROW("ChannelPermuteNode: permutation_order must contain 3 elements");
    // Replicate the permutation order for each image in the batch
    std::vector<vx_uint32> perm_tensor(_batch_size * 3);
    for (unsigned i = 0; i < _batch_size; i++) {
        for (unsigned j = 0; j < 3; j++) {
            perm_tensor[i * 3 + j] = _permutation_order[j];
        }
    }
    vx_status status;
    if ((status = vxTruncateArray(_permutation_array, 0)) != VX_SUCCESS)
        THROW("ChannelPermuteNode: vxTruncateArray failed: " + TOSTR(status));
    if ((status = vxAddArrayItems(_permutation_array, _batch_size * 3, perm_tensor.data(), sizeof(vx_uint32))) != VX_SUCCESS)
        THROW("ChannelPermuteNode: vxAddArrayItems failed: " + TOSTR(status));
}
