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
#include "augmentations/filter_augmentations/node_erase.h"
#include "pipeline/exception.h"

EraseNode::EraseNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs),
      _num_boxes(NUM_BOXES_RANGE[0], NUM_BOXES_RANGE[1]) {}

void EraseNode::create_node() {
    if (_node)
        return;

    // Tensor layout and ROI type
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());

    vx_context vx_ctx = vxGetContext((vx_reference)_graph->get());
    vx_scalar input_layout_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vx_ctx, VX_TYPE_INT32, &roi_type);

    vx_tensor anchor_tensor = nullptr;
    vx_tensor color_tensor  = nullptr;
    vx_array  num_boxes_arr = nullptr;

    if (_use_raw_vectors) {
        // Build per-sample num_boxes array (int32)
        if (_num_boxes_vec.empty()) {
            THROW("Erase raw-vector mode requires non-empty num_boxes vector")
        }
        // Ensure length == batch; replicate if single value
        if (_num_boxes_vec.size() != _batch_size) {
            if (_num_boxes_vec.size() == 1) {
                _num_boxes_vec.resize(_batch_size, _num_boxes_vec[0]);
            } else {
                THROW("num_boxes vector length must be 1 or equal to batch size")
            }
        }
        int max_boxes = 0;
        for (int nb : _num_boxes_vec) max_boxes = std::max(max_boxes, nb);
        if (max_boxes <= 0) max_boxes = 1; // avoid zero-sized tensors

        // Create vx_array for num_boxes
        num_boxes_arr = vxCreateArray(vx_ctx, VX_TYPE_INT32, _batch_size);
        vx_status st = vxAddArrayItems(num_boxes_arr, _batch_size, _num_boxes_vec.data(), sizeof(int32_t));
        if (st != VX_SUCCESS) THROW("vxAddArrayItems failed while creating num_boxes array: " + TOSTR(st))

        // Prepare anchor tensor [N, max_boxes, 4] FP32
        vx_size anchor_dims[3] = { (vx_size)_batch_size, (vx_size)max_boxes, (vx_size)4 };
        anchor_tensor = vxCreateTensor(vx_ctx, 3, anchor_dims, VX_TYPE_FLOAT32, 0);
        if (vxGetStatus((vx_reference)anchor_tensor) != VX_SUCCESS)
            THROW("vxCreateTensor failed for anchor tensor")

        // Prepare color tensor [N, max_boxes, 3] FP32
        vx_size color_dims[3] = { (vx_size)_batch_size, (vx_size)max_boxes, (vx_size)3 };
        color_tensor = vxCreateTensor(vx_ctx, 3, color_dims, VX_TYPE_FLOAT32, 0);
        if (vxGetStatus((vx_reference)color_tensor) != VX_SUCCESS)
            THROW("vxCreateTensor failed for color tensor")

        // Compute strides (in bytes)
        vx_size anchor_strides[3] = {
            (vx_size)(max_boxes * 4 * sizeof(float)),   // stride for N
            (vx_size)(4 * sizeof(float)),               // stride for boxes
            (vx_size)(sizeof(float))                    // stride for channels (4)
        };
        vx_size color_strides[3] = {
            (vx_size)(max_boxes * 3 * sizeof(float)),   // stride for N
            (vx_size)(3 * sizeof(float)),               // stride for boxes
            (vx_size)(sizeof(float))                    // stride for channels (3)
        };

        // Fill host buffers for anchors/colors
        std::vector<float> anchor_buf(_batch_size * max_boxes * 4, 0.0f);
        std::vector<float> color_buf (_batch_size * max_boxes * 3, 0.0f);

        // Determine source interpretation: per-batch concatenation or single-sample replicated
        auto total_anchor_needed = 0ul;
        for (int nb : _num_boxes_vec) total_anchor_needed += (unsigned long)(nb * 4);
        auto total_color_needed = 0ul;
        for (int nb : _num_boxes_vec) total_color_needed += (unsigned long)(nb * 3);

        bool anchor_is_batch_concat = (_anchor_vec.size() == total_anchor_needed);
        bool color_is_batch_concat  = (_colors_vec.size() == total_color_needed);

        // If single-sample provided, assume first sample data replicated/truncated per nb[i]
        bool anchor_is_single = (_anchor_vec.size() == (size_t)(4 * std::max(1, max_boxes)));
        bool color_is_single  = (_colors_vec.size() == (size_t)(3 * std::max(1, max_boxes)));

        // Fill per sample
        size_t a_src_off = 0, c_src_off = 0;
        for (size_t i = 0; i < _batch_size; ++i) {
            int nb = _num_boxes_vec[i];
            // Anchors
            for (int b = 0; b < nb; ++b) {
                const float* src_a = nullptr;
                if (anchor_is_batch_concat) {
                    src_a = &_anchor_vec[a_src_off + b * 4];
                } else {
                    // replicate from the first nb anchors of single-sample vector
                    if ((size_t)((b + 1) * 4) > _anchor_vec.size())
                        THROW("anchor_box_info vector smaller than required");
                    src_a = &_anchor_vec[b * 4];
                }
                size_t dst_idx = (i * max_boxes + b) * 4;
                anchor_buf[dst_idx + 0] = src_a[0];
                anchor_buf[dst_idx + 1] = src_a[1];
                anchor_buf[dst_idx + 2] = src_a[2];
                anchor_buf[dst_idx + 3] = src_a[3];
            }
            // Colors
            for (int b = 0; b < nb; ++b) {
                const float* src_c = nullptr;
                if (color_is_batch_concat) {
                    src_c = &_colors_vec[c_src_off + b * 3];
                } else {
                    if ((size_t)((b + 1) * 3) > _colors_vec.size())
                        THROW("colors vector smaller than required");
                    src_c = &_colors_vec[b * 3];
                }
                size_t dst_idx = (i * max_boxes + b) * 3;
                color_buf[dst_idx + 0] = src_c[0];
                color_buf[dst_idx + 1] = src_c[1];
                color_buf[dst_idx + 2] = src_c[2];
            }
            if (anchor_is_batch_concat) a_src_off += (size_t)(nb * 4);
            if (color_is_batch_concat)  c_src_off += (size_t)(nb * 3);
        }

        // Copy buffers into vx_tensors
        vx_size start[3] = {0, 0, 0};
        vx_size end_a[3] = { (vx_size)_batch_size, (vx_size)max_boxes, (vx_size)4 };
        vx_status st_a = vxCopyTensorPatch(anchor_tensor, 3, start, end_a, anchor_strides,
                                           anchor_buf.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        if (st_a != VX_SUCCESS) THROW("vxCopyTensorPatch failed for anchor tensor: " + TOSTR(st_a))

        vx_size end_c[3] = { (vx_size)_batch_size, (vx_size)max_boxes, (vx_size)3 };
        vx_status st_c = vxCopyTensorPatch(color_tensor, 3, start, end_c, color_strides,
                                           color_buf.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        if (st_c != VX_SUCCESS) THROW("vxCopyTensorPatch failed for color tensor: " + TOSTR(st_c))
    } else {
        if (!_anchor || !_colors)
            THROW("Erase node requires non-null anchor and colors tensors")
        // ParameterVX path for num_boxes
        _num_boxes.create_array(_graph, VX_TYPE_INT32, _batch_size);
        anchor_tensor = _anchor->handle();
        color_tensor  = _colors->handle();
        num_boxes_arr = _num_boxes.default_array();
    }

    // Create Erase node via MIVisionX RPP extension
    _node = vxExtRppErase(_graph->get(),
                          _inputs[0]->handle(),
                          _inputs[0]->get_roi_tensor(),
                          _outputs[0]->handle(),
                          anchor_tensor,
                          color_tensor,
                          num_boxes_arr,
                          input_layout_vx,
                          output_layout_vx,
                          roi_type_vx);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the erase (vxExtRppErase) node failed: " + TOSTR(status))
}

void EraseNode::init(Tensor *anchor_box_info, Tensor *colors, int num_boxes_fixed) {
    _anchor = anchor_box_info;
    _colors = colors;
    _num_boxes.set_param(num_boxes_fixed);
    _use_raw_vectors = false;
}

void EraseNode::init(Tensor *anchor_box_info, Tensor *colors, IntParam *num_boxes_param) {
    _anchor = anchor_box_info;
    _colors = colors;
    _num_boxes.set_param(core(num_boxes_param));
    _use_raw_vectors = false;
}

// New: raw vector-based init (replicates across batch if needed)
void EraseNode::init(std::vector<float> anchor,
                     std::vector<float> shape,
                     std::vector<unsigned> num_boxes
                     std::vector<float> fill_value) {

    // Validate anchor and shape should be same size
    // _anchor_vec = std::move(anchor);
    // _shape_vec = std::move(shape);
    
    _num_boxes_vec.resize(_batch_size);
    if (num_boxes.size() == 1) {
        std::fill(_num_boxes_vec.begin(), _num_boxes_vec.end(), num_boxes[0]);
    } else if (num_boxes.size() == _batch_size) {
        _num_boxes_vec = num_boxes;
    } else {
        THROW("Invalid number of elements passed for num of boxes")
    }
    _total_boxes = std::accumulate(_num_boxes_vec.begin(), _num_boxes_vec.end(), 0);

    _fill_values_vec.resize(_total_boxes * _batch_size * _inputs[0]->info().get_channels());
    if (fill_value.size() == 1) {
        std::fill(_fill_values_vec.begin(), _fill_values_vec.end(), fill_value[0]);
    } else if (fill_value.size() == _inputs[0]->info().get_channels()) {
        for (int i = 0; i < _batch_size; ++i)
            std::copy(_fill_values_vec.begin(), _fill_values_vec.end(), fill_value.begin() + i * fill_value.size());
    } else if (fill_value.size() == _batch_size) {
        const int channels = _inputs[0]->info().get_channels();
        for (int i = 0; i < _batch_size; ++i) {
            float* dst = _fill_values_vec.data() + static_cast<size_t>(i) * channels;
            std::fill(dst, dst + channels, fill_value[i]);
        }
    } else if (fill_value.size() == (_total_boxes * _batch_size * _inputs[0]->info().get_channels())) {
        _fill_values_vec = std::move(fill_value);
    } else {
        THROW("Invalid number of values passed for fill value")
    }
    
    _anchor_box_vec.resize(_total_boxes * 4);
    if (num_boxes.size() == 1 && anchor.size() == num_boxes[0] * 2) {
        for (int i = 0; i < _batch_size; i++) {
            for (int n = 0; n < _num_boxes_vec[i]; n++) {
                _anchor_box_vec[(i * _num_boxes_vec[i] + n) * 4 + 0] = anchor[n * 2];
                _anchor_box_vec[(i * _num_boxes_vec[i] + n) * 4 + 1] = anchor[n * 2 + 1];
            }
        }

    } else if (anchor.size() == (_total_boxes * 2)) {
        for (int i = 0; i < _batch_size; i++) {
            for (int n = 0; n < _num_boxes_vec[i]; n++) {
                _anchor_box_vec[(i * _num_boxes_vec[i] + n) * 4 + 0] = anchor[(i * _num_boxes_vec[i] + n) * 2];
                _anchor_box_vec[(i * _num_boxes_vec[i] + n) * 4 + 1] = anchor[(i * _num_boxes_vec[i] + n) * 2 + 1];
            }
        }
    }

    if (num_boxes.size() == 1 && shape.size() == num_boxes[0] * 2) {
        for (int i = 0; i < _batch_size; i++) {
            for (int n = 0; n < _num_boxes_vec[i]; n++) {
                _anchor_box_vec[(i * _num_boxes_vec[i] + n) * 4 + 2] = shape[n * 2];
                _anchor_box_vec[(i * _num_boxes_vec[i] + n) * 4 + 3] = shape[n * 2 + 1];
            }
        }

    } else if (shape.size() == (_total_boxes * 2)) {
        for (int i = 0; i < _batch_size; i++) {
            for (int n = 0; n < _num_boxes_vec[i]; n++) {
                _anchor_box_vec[(i * _num_boxes_vec[i] + n) * 4 + 2] = shape[(i * _num_boxes_vec[i] + n) * 2];
                _anchor_box_vec[(i * _num_boxes_vec[i] + n) * 4 + 3] = shape[(i * _num_boxes_vec[i] + n) * 2 + 1];
            }
        }
    }
}

void EraseNode::update_node() {
    if (_use_raw_vectors) {
        // Raw vectors are static; nothing to update per run
        return;
    }
    _num_boxes.update_array();
}
