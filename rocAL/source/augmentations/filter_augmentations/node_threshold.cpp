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

#include <vx_ext_rpp.h>
#include <vx_ext_rpp_version.h>
#include "augmentations/filter_augmentations/node_threshold.h"
#include "pipeline/exception.h"

ThresholdNode::ThresholdNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs) {}

void fill_vector_with_threshold_values(std::vector<float>& threshold_batch,
                                       std::vector<float>& threshold_values,
                                       size_t no_of_channels) {

    size_t threshold_vec_size = threshold_batch.size();

    if (threshold_values.size() == no_of_channels) {
        for (size_t batch_channel_idx = 0; batch_channel_idx < threshold_vec_size; batch_channel_idx += no_of_channels) {
            for (size_t channel_idx = 0; channel_idx < no_of_channels; channel_idx++) {
                threshold_batch[batch_channel_idx + channel_idx] = threshold_values[channel_idx];
            }
        }
    } else if (threshold_values.size() == threshold_vec_size) {
        threshold_batch = threshold_values;
    } else if (threshold_values.size() > 0) {
        THROW("Threshold: Threshold vector length must be no_of_channels or no_of_channels * batch_size, got: " + TOSTR(threshold_values.size()));
    }
}

void ThresholdNode::create_node() {
    if (_node)
        return;

#if VX_EXT_RPP_CHECK_VERSION(3, 1, 3)
    // Create per-sample arrays for min and max threshold values
    auto no_of_channels = _inputs[0]->info().get_channels();
    auto array_size = _batch_size * no_of_channels;
    std::vector<float> min_array, max_array;
    min_array.resize(array_size, 0.0f);
    max_array.resize(array_size, 0.0f);
    fill_vector_with_threshold_values(min_array, _min, no_of_channels);
    fill_vector_with_threshold_values(max_array, _max, no_of_channels);

    // Create vx_array and populate it
    vx_status status;
    vx_context ctx = vxGetContext((vx_reference)_graph->get());
    vx_array min_array_vx = vxCreateArray(ctx, VX_TYPE_FLOAT32, array_size);
    status = vxAddArrayItems(min_array_vx, array_size, min_array.data(), sizeof(vx_float32));
    if (status != VX_SUCCESS) {
        vxReleaseArray(&min_array_vx);
        THROW("Threshold: vxAddArrayItems failed while creating min array: " + TOSTR(status));
    }

    vx_array max_array_vx = vxCreateArray(ctx, VX_TYPE_FLOAT32, array_size);
    status = vxAddArrayItems(max_array_vx, array_size, max_array.data(), sizeof(vx_float32));
    if (status != VX_SUCCESS) {
        vxReleaseArray(&min_array_vx);
        vxReleaseArray(&max_array_vx);
        THROW("Threshold: vxAddArrayItems failed while creating max array: " + TOSTR(status));
    }

    // Tensor layout and ROI type
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());

    vx_scalar input_layout_vx = vxCreateScalar(ctx, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(ctx, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(ctx, VX_TYPE_INT32, &roi_type);

    // Create Threshold node via MIVisionX RPP extension
    _node = vxExtRppThreshold(_graph->get(),
                              _inputs[0]->handle(),
                              _inputs[0]->get_roi_tensor(),
                              _outputs[0]->handle(),
                              min_array_vx,
                              max_array_vx,
                              input_layout_vx,
                              output_layout_vx,
                              roi_type_vx);

    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the threshold (vxExtRppThreshold) node failed: " + TOSTR(status));
#else
    THROW("ThresholdNode: vxExtRppThreshold requires amd_rpp version >= 3.1.3");
#endif
}

void ThresholdNode::init(std::vector<float>& min_val, std::vector<float>& max_val) {
    _min = min_val;
    _max = max_val;
}

void ThresholdNode::update_node() {}
