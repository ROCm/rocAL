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

#include <functional>

#include "augmentations/arithmetic_augmentations/node_tensor_statistics.h"
#include "pipeline/exception.h"

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 7)
namespace {
inline void tensor_reduction_create(Node *node,
                                    vx_node &vx_node_ref,
                                    vx_tensor input_tensor,
                                    vx_tensor input_roi,
                                    vx_tensor output_tensor,
                                    vx_graph graph,
                                    Tensor *input,
                                    std::function<vx_node(vx_graph, vx_tensor, vx_tensor, vx_tensor, vx_scalar, vx_scalar)> create_fn) {
    if (vx_node_ref)
        return;
    int input_layout = static_cast<int>(input->info().layout());
    int roi_type = static_cast<int>(input->info().roi_type());
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar input_layout_vx = vxCreateScalar(context, VX_TYPE_INT32, &input_layout);
    vx_scalar roi_type_vx = vxCreateScalar(context, VX_TYPE_INT32, &roi_type);
    vx_node_ref = create_fn(graph, input_tensor, input_roi, output_tensor, input_layout_vx, roi_type_vx);
    vx_status status = vxGetStatus((vx_reference)vx_node_ref);
    if (status != VX_SUCCESS)
        THROW("Adding tensor reduction node failed: " + TOSTR(status));
}

inline void tensor_stddev_create(vx_node &vx_node_ref,
                                 vx_tensor input_tensor,
                                 vx_tensor input_roi,
                                 vx_tensor output_tensor,
                                 vx_tensor mean_tensor,
                                 vx_graph graph,
                                 Tensor *input) {
    if (vx_node_ref)
        return;
    int input_layout = static_cast<int>(input->info().layout());
    int roi_type = static_cast<int>(input->info().roi_type());
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar input_layout_vx = vxCreateScalar(context, VX_TYPE_INT32, &input_layout);
    vx_scalar roi_type_vx = vxCreateScalar(context, VX_TYPE_INT32, &roi_type);
    vx_node_ref = vxExtRppTensorStdDev(graph, input_tensor, input_roi, output_tensor, mean_tensor, input_layout_vx, roi_type_vx);
    vx_status status = vxGetStatus((vx_reference)vx_node_ref);
    if (status != VX_SUCCESS)
        THROW("Adding tensor stddev node failed: " + TOSTR(status));
}
}  // namespace

TensorSumNode::TensorSumNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void TensorSumNode::create_node() {
    tensor_reduction_create(this,
                            _node,
                            _inputs[0]->handle(),
                            _inputs[0]->get_roi_tensor(),
                            _outputs[0]->handle(),
                            _graph->get(),
                            _inputs[0],
                            [](vx_graph graph, vx_tensor src, vx_tensor src_roi, vx_tensor dst, vx_scalar input_layout, vx_scalar roi_type) {
                                return vxExtRppTensorSum(graph, src, src_roi, dst, input_layout, roi_type);
                            });
}

TensorMinNode::TensorMinNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void TensorMinNode::create_node() {
    tensor_reduction_create(this,
                            _node,
                            _inputs[0]->handle(),
                            _inputs[0]->get_roi_tensor(),
                            _outputs[0]->handle(),
                            _graph->get(),
                            _inputs[0],
                            [](vx_graph graph, vx_tensor src, vx_tensor src_roi, vx_tensor dst, vx_scalar input_layout, vx_scalar roi_type) {
                                return vxExtRppTensorMin(graph, src, src_roi, dst, input_layout, roi_type);
                            });
}

TensorMaxNode::TensorMaxNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void TensorMaxNode::create_node() {
    tensor_reduction_create(this,
                            _node,
                            _inputs[0]->handle(),
                            _inputs[0]->get_roi_tensor(),
                            _outputs[0]->handle(),
                            _graph->get(),
                            _inputs[0],
                            [](vx_graph graph, vx_tensor src, vx_tensor src_roi, vx_tensor dst, vx_scalar input_layout, vx_scalar roi_type) {
                                return vxExtRppTensorMax(graph, src, src_roi, dst, input_layout, roi_type);
                            });
}

TensorMeanNode::TensorMeanNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void TensorMeanNode::create_node() {
    tensor_reduction_create(this,
                            _node,
                            _inputs[0]->handle(),
                            _inputs[0]->get_roi_tensor(),
                            _outputs[0]->handle(),
                            _graph->get(),
                            _inputs[0],
                            [](vx_graph graph, vx_tensor src, vx_tensor src_roi, vx_tensor dst, vx_scalar input_layout, vx_scalar roi_type) {
                                return vxExtRppTensorMean(graph, src, src_roi, dst, input_layout, roi_type);
                            });
}

TensorStdDevNode::TensorStdDevNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void TensorStdDevNode::create_node() {
    tensor_stddev_create(_node,
                         _inputs[0]->handle(),
                         _inputs[0]->get_roi_tensor(),
                         _outputs[0]->handle(),
                         _inputs[1]->handle(),
                         _graph->get(),
                         _inputs[0]);
}

#endif
