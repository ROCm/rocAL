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

LensCorrectionNode::LensCorrectionNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {
    _camera_matrix.resize(_batch_size * 9);
    _distortion_coeffs.resize(_batch_size * 8);
}

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

    _node = vxExtRppLensCorrection(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _camera_matrix_vx_array, _distortion_coeffs_vx_array, input_layout_vx, output_layout_vx, roi_type_vx);
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the lens correction (vxExtRppLensCorrection) node failed: " + TOSTR(status))
}

void LensCorrectionNode::init(std::vector<CameraMatrix> camera_matrix, std::vector<DistortionCoeffs> distortion_coeffs) {
    // Handle camera_matrix: if size is 1, replicate for all images in batch
    std::vector<CameraMatrix> camera_matrix_batch;
    if (camera_matrix.size() == 1) {
        // Single camera matrix provided - replicate for all images in batch
        camera_matrix_batch.resize(_batch_size, camera_matrix[0]);
    } else if (camera_matrix.size() == _batch_size) {
        // Camera matrix provided for each image in batch
        camera_matrix_batch = camera_matrix;
    } else {
        THROW("Camera matrix size must be either 1 (to replicate for all images) or equal to batch size (" + TOSTR(_batch_size) + ")")
    }
    
    // Handle distortion_coeffs: if size is 1, replicate for all images in batch
    std::vector<DistortionCoeffs> distortion_coeffs_batch;
    if (distortion_coeffs.size() == 1) {
        // Single distortion coefficients provided - replicate for all images in batch
        distortion_coeffs_batch.resize(_batch_size, distortion_coeffs[0]);
    } else if (distortion_coeffs.size() == _batch_size) {
        // Distortion coefficients provided for each image in batch
        distortion_coeffs_batch = distortion_coeffs;
    } else {
        THROW("Distortion coefficients size must be either 1 (to replicate for all images) or equal to batch size (" + TOSTR(_batch_size) + ")")
    }
    
    // Fill the arrays with the camera matrix and distortion coefficients
    for (unsigned i = 0; i < _batch_size; i++) {
        unsigned camera_matrix_index = i * 9;
        unsigned dist_coeff_index = i * 8;
        _camera_matrix[camera_matrix_index + 0] = camera_matrix_batch[i].fx;
        _camera_matrix[camera_matrix_index + 1] = 0;
        _camera_matrix[camera_matrix_index + 2] = camera_matrix_batch[i].cx;
        _camera_matrix[camera_matrix_index + 3] = 0;
        _camera_matrix[camera_matrix_index + 4] = camera_matrix_batch[i].fy;
        _camera_matrix[camera_matrix_index + 5] = camera_matrix_batch[i].cy;
        _camera_matrix[camera_matrix_index + 6] = 0;
        _camera_matrix[camera_matrix_index + 7] = 0;
        _camera_matrix[camera_matrix_index + 8] = 1;
        _distortion_coeffs[dist_coeff_index + 0] = distortion_coeffs_batch[i].k1;
        _distortion_coeffs[dist_coeff_index + 1] = distortion_coeffs_batch[i].k2;
        _distortion_coeffs[dist_coeff_index + 2] = distortion_coeffs_batch[i].p1;
        _distortion_coeffs[dist_coeff_index + 3] = distortion_coeffs_batch[i].p2;
        _distortion_coeffs[dist_coeff_index + 4] = distortion_coeffs_batch[i].k3;
        _distortion_coeffs[dist_coeff_index + 5] = 0;
        _distortion_coeffs[dist_coeff_index + 6] = 0;
        _distortion_coeffs[dist_coeff_index + 7] = 0;
    }
}
