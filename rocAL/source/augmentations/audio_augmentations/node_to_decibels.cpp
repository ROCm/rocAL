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
#include "node_to_decibels.h"
#include "exception.h"

ToDeciblesNode::ToDeciblesNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) :
    Node(inputs, outputs) {}

void ToDeciblesNode::create_node() {
    if(_node)
        return;

    vx_status status = VX_SUCCESS;
    vx_scalar cutoff_db = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_cutoff_db);
    vx_scalar multiplier = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_multiplier);
    vx_scalar reference_magnitude = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_reference_magnitude);
    _node = vxExtRppToDecibels(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _dst_tensor_roi, cutoff_db, multiplier, reference_magnitude);

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the to_decibels (vxRppToDecibels) node failed: "+ TOSTR(status))
}

void ToDeciblesNode::update_node() {}

void ToDeciblesNode::init(float cutoff_db, float multiplier, float reference_magnitude) {
    _cutoff_db = cutoff_db;
    _multiplier = multiplier;
    _reference_magnitude = reference_magnitude;
}
