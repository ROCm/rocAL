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
#include "node_non_silent_region.h"
#include "exception.h"

NonSilentRegionNode::NonSilentRegionNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) :
        Node(inputs, outputs) { }

void NonSilentRegionNode::create_node() {
    if(_node)
        return;

    vx_scalar cutoff_db = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_cutoff_db);
    vx_scalar reference_power = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_reference_power);
    vx_scalar window_length = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_window_length);
    vx_scalar reset_interval = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_reset_interval);
    _node = vxExtRppNonSilentRegion(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _outputs[1]->handle(),
                                    cutoff_db, reference_power, window_length, reset_interval);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the non silent region node (vxRppNonSilentRegion) failed: " + TOSTR(status))
}

void NonSilentRegionNode::update_node() { }

void NonSilentRegionNode::init(float cutoff_db, float reference_power, int window_length, int reset_interval) {
    _cutoff_db = cutoff_db;
    _reference_power = reference_power;
    _window_length = window_length;
    _reset_interval = reset_interval;
}
