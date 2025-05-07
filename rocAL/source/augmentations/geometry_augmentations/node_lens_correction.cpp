/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include "augmentations/geometry_augmentations/node_lens_correction.h"
#include "pipeline/exception.h"

LensCorrectionNode::LensCorrectionNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void LensCorrectionNode::create_node() {
    if (_node)
        return;

    vx_status status = VX_SUCCESS;
    _camera_matrix_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * 9);
    status |= vxAddArrayItems(_camera_matrix_vx_array, _camera_matrix.size(), _camera_matrix.data(), sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the lens correction (vxExtRppLensCorrection) node: " + TOSTR(status) + "  " + TOSTR(status))
    
    _distortion_coeffs_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * 8);
    status |= vxAddArrayItems(_distortion_coeffs_vx_array, _distortion_coeffs.size(), _distortion_coeffs.data(), sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the lens correction (vxExtRppLensCorrection) node: " + TOSTR(status) + "  " + TOSTR(status))
    
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppLensCorrection(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _camera_matrix_vx_array, _distortion_coeffs_vx_array, input_layout_vx, output_layout_vx,roi_type_vx);
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the lens correction (vxExtRppLensCorrection) node failed: " + TOSTR(status))
}

void LensCorrectionNode::init(std::vector<CameraMatrix> camera_matrix, std::vector<DistortionCoeffs> distortion_coeffs) {
    _camera_matrix.resize(_batch_size);
    _distortion_coeffs.resize(_batch_size);
    for (unsigned i = 0; i < _batch_size; i++) {
        _camera_matrix[i].resize(9);
        _camera_matrix[i][0] = camera_matrix[i].fx;
        _camera_matrix[i][1] = 0;
        _camera_matrix[i][2] = camera_matrix[i].cx;
        _camera_matrix[i][3] = 0;
        _camera_matrix[i][4] = camera_matrix[i].fy;
        _camera_matrix[i][5] = camera_matrix[i].cy;
        _camera_matrix[i][6] = 0;
        _camera_matrix[i][7] = 0;
        _camera_matrix[i][8] = 1;
        _distortion_coeffs[i].resize(8);
        _distortion_coeffs[i][0] = distortion_coeffs[i].k1;
        _distortion_coeffs[i][1] = distortion_coeffs[i].k2;
        _distortion_coeffs[i][2] = distortion_coeffs[i].p1;
        _distortion_coeffs[i][3] = distortion_coeffs[i].p2;
        _distortion_coeffs[i][4] = distortion_coeffs[i].k3;
        _distortion_coeffs[i][5] = 0;
        _distortion_coeffs[i][6] = 0;
        _distortion_coeffs[i][7] = 0;
    }
}
