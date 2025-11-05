/*
Copyright (c) 2025
Advanced Micro Devices, Inc. All rights reserved.
*/
#include <vx_ext_rpp.h>
#include "augmentations/filter_augmentations/node_bitwise_ops.h"
#include "pipeline/exception.h"

void BitwiseOpsNode::create_node() {
    if (_node) return;

    if (_inputs.size() < 2)
        THROW("BitwiseOps node needs two input tensors")

    int input_layout  = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type      = static_cast<int>(_inputs[0]->info().roi_type());
    int op            = 0;

    switch (_op) {
        case BitwiseOp::AND: op = 0; break;
        case BitwiseOp::OR:  op = 1; break;
        case BitwiseOp::XOR: op = 2; break;
        default: op = 0; break;
    }

    vx_context ctx = vxGetContext((vx_reference)_graph->get());
    vx_scalar input_layout_vx  = vxCreateScalar(ctx, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(ctx, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx      = vxCreateScalar(ctx, VX_TYPE_INT32, &roi_type);
    vx_scalar op_type_vx       = vxCreateScalar(ctx, VX_TYPE_INT32, &op);

    // pSrcRoi is carried in _inputs[0] ROI tensor
    _node = vxExtRppBitwiseOps(_graph->get(),
                               _inputs[0]->handle(),      // pSrc1
                               _inputs[1]->handle(),      // pSrc2
                               _inputs[0]->get_roi_tensor(), // pSrcRoi (per-sample ROI for inputs)
                               _outputs[0]->handle(),     // pDst
                               input_layout_vx,
                               output_layout_vx,
                               roi_type_vx,
                               op_type_vx);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the bitwise ops (vxExtRppBitwiseOps) node failed: " + TOSTR(status))
}

void BitwiseOpsNode::update_node() {
    // No per-iteration dynamic parameters
}
