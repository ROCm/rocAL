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
#include <graph.h>
#include "node_crop_resize.h"
#include "exception.h"

CropResizeNode::CropResizeNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : CropNode(inputs, outputs) {
    _crop_param = std::make_shared<RocalRandomCropParam>(_batch_size);
}

void CropResizeNode::create_node() {
    if (_node)
        return;

    _crop_param->create_array(_graph);
    std::vector<uint32_t> dst_roi_width(_batch_size, _outputs[0]->info().max_shape()[0]);
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().max_shape()[1]);
    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    vx_status width_status, height_status;
    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if (width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the crop resize node (vxExtRppResizeCrop)  node: " + TOSTR(width_status) + "  " + TOSTR(height_status))

    create_crop_tensor();
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppResizeCrop(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _crop_tensor, _outputs[0]->handle(),
                               _dst_roi_width, _dst_roi_height, input_layout_vx, output_layout_vx, roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop resize node (vxExtRppResizeCrop) failed: " + TOSTR(status))
}

void CropResizeNode::update_node() {
    _crop_param->set_image_dimensions(_inputs[0]->info().roi().get_2D_roi());
    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
    _outputs[0]->update_tensor_roi(crop_w_dims, crop_h_dims);

    // Obtain the crop coordinates and update the roi
    auto x1 = _crop_param->get_x1_arr_val();
    auto y1 = _crop_param->get_y1_arr_val();
    auto x2 = _crop_param->get_croph_arr_val();
    auto y2 = _crop_param->get_cropw_arr_val();
    Roi2DCords *crop_dims = static_cast<Roi2DCords *>(_crop_coordinates);
    for (unsigned i = 0; i < _batch_size; i++) {
        crop_dims[i].xywh.x = x1[i];
        crop_dims[i].xywh.y = y1[i];
        crop_dims[i].xywh.w = crop_w_dims[i];
        crop_dims[i].xywh.h = crop_h_dims[i];
    }
}

void CropResizeNode::init(float area, float aspect_ratio, float x_center_drift, float y_center_drift) {
    _crop_param->set_area_factor(ParameterFactory::instance()->create_single_value_param(area));
    _crop_param->set_aspect_ratio(ParameterFactory::instance()->create_single_value_param(aspect_ratio));
    _crop_param->set_x_drift_factor(ParameterFactory::instance()->create_single_value_param(x_center_drift));
    _crop_param->set_y_drift_factor(ParameterFactory::instance()->create_single_value_param(y_center_drift));
}

void CropResizeNode::init(FloatParam *area, FloatParam *aspect_ratio, FloatParam *x_center_drift, FloatParam *y_center_drift) {
    _crop_param->set_area_factor(core(area));
    _crop_param->set_aspect_ratio(core(aspect_ratio));
    _crop_param->set_x_drift_factor(core(x_center_drift));
    _crop_param->set_y_drift_factor(core(y_center_drift));
}
