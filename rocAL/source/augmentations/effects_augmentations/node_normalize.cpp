/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "node_normalize.h"

#include <vx_ext_rpp.h>

#include "exception.h"

NormalizeNode::NormalizeNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void NormalizeNode::create_node() {
    if (_node)
        return;

    _compute_mean = _mean.size() ? 0 : 1;
    _compute_stddev = _std_dev.size() ? 0 : 1;

    int mean_stddev_array_size = 1;
    auto nDim = _inputs[0]->info().num_of_dims() - 1;
    uint axis[nDim];
    for (unsigned i = 0; i < _batch_size; i++) {
        int totalElements = 1;
        unsigned *tensor_shape = _inputs[0]->info().roi()[i].end;
        for (uint j = 0; j < nDim; j++) {
            axis[j] = ((_axis_mask & (int)(pow(2, j))) >= 1) ? 1 : 0;
            totalElements *= !axis[j] ? tensor_shape[j] : 1;
        }
        mean_stddev_array_size = std::max(mean_stddev_array_size, totalElements);
    }
    std::vector<float> mean_vec, stddev_vec;
    mean_vec.resize(_batch_size * mean_stddev_array_size, 0);
    stddev_vec.resize(_batch_size * mean_stddev_array_size, 0);

    if (!_compute_mean && !_compute_stddev) {
        for (uint i = 0; i < _batch_size; i++) {
            for (int j = 0; j < mean_stddev_array_size; j += _mean.size()) {
                for (int k = 0; k < _mean.size(); k++) {
                    mean_vec[i * mean_stddev_array_size + j + k] = _mean[k];
                    stddev_vec[i * mean_stddev_array_size + j + k] = _std_dev[k];
                }
            }
        }
    }
    vx_status status = VX_SUCCESS;
    _mean_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, mean_vec.size());
    status |= vxAddArrayItems(_mean_vx_array, mean_vec.size(), mean_vec.data(), sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the normalize node (vxExtRppNormalize)  node: " + TOSTR(status) + "  " + TOSTR(status))

    _stddev_vx_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, stddev_vec.size());
    status |= vxAddArrayItems(_stddev_vx_array, stddev_vec.size(), stddev_vec.data(), sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the normalize node (vxExtRppNormalize)  node: " + TOSTR(status) + "  " + TOSTR(status))

    vx_scalar axis_mask = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_axis_mask);
    vx_scalar scale = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_scale);
    vx_scalar shift = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_shift);
    vx_scalar compute_mean = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_compute_mean);
    vx_scalar compute_stddev = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_compute_stddev);
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    std::cerr << "Batch size: " << _batch_size << "\n";
    std::cerr << "Axis mask: " << _axis_mask << "\n";
    std::cerr << "Mean std values size: " << mean_stddev_array_size << "\n";
    // for (uint i = 0; i < _batch_size; i++) {
    //     for (int j = 0; j < _mean.size(); j++) {
    //         std::cerr << "Mean: " << mean_vec[i * mean_stddev_array_size + j] << " Std: " << stddev_vec[i * mean_stddev_array_size + j] << "\n";
    //     }
    // }
    std::cerr << "compute_mean: " << _compute_mean << "\n";
    std::cerr << "compute_stddev: " << _compute_stddev << "\n";
    std::cerr << "scale: " << _scale << "\n";
    std::cerr << "shift: " << _shift << "\n";
    std::cerr << "input_layout_vx: " << input_layout << "\n";
    std::cerr << "roi_type_vx: " << roi_type << "\n";

    _node = vxExtRppNormalize(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _outputs[0]->get_roi_tensor(), axis_mask,
                              _mean_vx_array, _stddev_vx_array, compute_mean, compute_stddev, scale, shift, input_layout_vx, roi_type_vx);
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the normalize (vxExtRppNormalize) node failed: " + TOSTR(status))
}

void NormalizeNode::init(std::vector<unsigned> &axes, std::vector<float> &mean, std::vector<float> &std_dev, float scale, float shift) {
    _mean = mean;
    _std_dev = std_dev;
    _scale = scale;
    _shift = shift;
    for (unsigned d = 0; d < axes.size(); d++)
        _axis_mask |= (1 << axes[d]);
}
