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
#include <vx_ext_amd.h>
#include "augmentations/filter_augmentations/node_erase.h"
#include "pipeline/exception.h"
#include "pipeline/tensor.h"
#include <cstring>

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
        // Fill up anchor and shape

        // Create vx_array for num_boxes
        num_boxes_arr = vxCreateArray(vx_ctx, VX_TYPE_INT32, _batch_size);
        vx_status st = vxAddArrayItems(num_boxes_arr, _batch_size, _num_boxes_vec.data(), sizeof(int32_t));
        if (st != VX_SUCCESS) THROW("vxAddArrayItems failed while creating num_boxes array: " + TOSTR(st))

        // Select memory type and VX mem type for handle-backed tensors
        auto mem_type = _inputs[0]->info().mem_type();
        vx_enum vx_mem = (mem_type == RocalMemType::HIP) ? VX_MEMORY_TYPE_HIP : VX_MEMORY_TYPE_HOST;

        // Anchor tensor handle
        vx_size anchor_dims[2]   = { (vx_size)_total_boxes, (vx_size)4 };
        vx_size anchor_stride[2] = { 0, 0 };
        anchor_stride[0] = sizeof(vx_int32);
        anchor_stride[1] = anchor_stride[0] * anchor_dims[0];

        size_t bytes_a = anchor_stride[1] * anchor_dims[1];
        allocate_host_or_pinned_mem(&_anchor_ptr, bytes_a, mem_type);
        std::memcpy(_anchor_ptr, _anchor_vec.data(), bytes_a);

        _vx_anchor = vxCreateTensorFromHandle(vxGetContext((vx_reference)_graph->get()),
                                              2, anchor_dims, VX_TYPE_INT32, 0, anchor_stride, _anchor_ptr, vx_mem);
        if (!_vx_anchor) THROW("vxCreateTensorFromHandle for anchor tensor failed");
        {
            vx_status s = vxGetStatus((vx_reference)_vx_anchor);
            if (s != VX_SUCCESS) THROW("vxCreateTensorFromHandle anchor failed: " + TOSTR(s));
        }

        // Color tensor handle
        vx_size color_dims[2]   = { (vx_size)_total_boxes, (vx_size)_inputs[0]->info().get_channels() };
        vx_size color_stride[2] = { 0, 0 };
        color_stride[0] = sizeof(vx_float32);
        color_stride[1] = color_stride[0] * color_dims[0];

        size_t bytes_c = color_stride[1] * color_dims[1];
        allocate_host_or_pinned_mem(&_color_ptr, bytes_c, mem_type);
        std::memcpy(_color_ptr, _fill_values_vec.data(), bytes_c);

        _vx_colors = vxCreateTensorFromHandle(vxGetContext((vx_reference)_graph->get()),
                                              2, color_dims, VX_TYPE_FLOAT32, 0, color_stride, _color_ptr, vx_mem);
        if (!_vx_colors) THROW("vxCreateTensorFromHandle for color tensor failed");
        {
            vx_status s = vxGetStatus((vx_reference)_vx_colors);
            if (s != VX_SUCCESS) THROW("vxCreateTensorFromHandle color failed: " + TOSTR(s));
        }

        // Output tensors for node creation
        anchor_tensor = _vx_anchor;
        color_tensor  = _vx_colors;
        _vx_num_boxes = num_boxes_arr;
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
                     std::vector<unsigned> num_boxes,
                     std::vector<float> fill_value) {
    // Use raw-vector path
    _use_raw_vectors = true;

    // Keep fill pattern for colors; create_node expands it to per-box/channel
    _fill_values = std::move(fill_value);

    // Normalize num_boxes to batch size
    _num_boxes_vec.clear();
    if (num_boxes.size() == 1) {
        _num_boxes_vec.assign(static_cast<size_t>(_batch_size), static_cast<int>(num_boxes[0]));
    } else if (num_boxes.size() == static_cast<size_t>(_batch_size)) {
        _num_boxes_vec.assign(num_boxes.begin(), num_boxes.end());
    } else {
        THROW("num_boxes vector length must be 1 or equal to batch size");
    }

    // Compute prefix offsets and total boxes
    std::vector<int> prefix(static_cast<size_t>(_batch_size) + 1, 0);
    for (int i = 0; i < _batch_size; ++i) prefix[i + 1] = prefix[i] + _num_boxes_vec[i];
    _total_boxes = static_cast<size_t>(prefix[_batch_size]);

    auto channels = _inputs[0]->info().get_channels();
    _fill_values_vec.resize(static_cast<size_t>(_total_boxes) * channels);

    const auto fill_sz = _fill_values.size();

    if (fill_sz == 1) {
        // One scalar for everything
        std::fill(_fill_values_vec.begin(), _fill_values_vec.end(), _fill_values[0]);
    }
    else if (fill_sz == static_cast<size_t>(channels)) {
        // Per-channel pattern replicated to each box
        float* dst = _fill_values_vec.data();
        for (int b = 0; b < _total_boxes; ++b) {
            std::copy_n(_fill_values.data(), channels, dst);
            dst += channels;
        }
    }
    else if (fill_sz == static_cast<size_t>(_batch_size)) {
        // Per-sample scalar replicated across its boxes (and channels)
        float* dst = _fill_values_vec.data();
        for (int i = 0; i < _batch_size; ++i) {
            const int nb = _num_boxes_vec[i];
            const float v = _fill_values[i];
            const size_t count = static_cast<size_t>(nb) * channels;
            std::fill_n(dst, count, v);
            dst += count;
        }
    }
    else if (fill_sz == static_cast<size_t>(_batch_size) * channels) {
        // Per-sample per-channel pattern replicated across that sample’s boxes
        float* dst = _fill_values_vec.data();
        const float* src = _fill_values.data();
        for (int i = 0; i < _batch_size; ++i) {
            for (int b = 0; b < _num_boxes_vec[i]; ++b) {
                std::copy_n(src + i * channels, channels, dst);
                dst += channels;
            }
        }
    }
    else if (num_boxes.size() == 1 &&
            fill_sz == static_cast<size_t>(num_boxes[0]) * channels) {
        // Single-sample per-box per-channel replicated across batch.
        // All samples must have the same nb equal to num_boxes[0].
        const int nb_single = num_boxes[0];
        float* dst = _fill_values_vec.data();
        for (int i = 0; i < _batch_size; ++i) {
            if (_num_boxes_vec[i] != nb_single)
                THROW("num_boxes mismatch across samples for single-sample fill pattern");
            const float* src = _fill_values.data();
            std::copy_n(src, static_cast<size_t>(nb_single) * channels, dst);
            dst += static_cast<size_t>(nb_single) * channels;
        }
    }
    else if (fill_sz == static_cast<size_t>(_total_boxes) * channels) {
        // Fully specified flattened values
        std::copy(_fill_values.begin(), _fill_values.end(), _fill_values_vec.begin());
    }
    else {
        THROW("Invalid number of values passed for fill value");
    }
    
    // Build contiguous [x1, y1, w, h] for all boxes
    _anchor_vec.assign(_total_boxes * 4, 0.0f);

    // Case 1: single-sample vectors replicated across batch
    if (num_boxes.size() == 1 &&
        anchor.size() == static_cast<size_t>(num_boxes[0]) * 2 &&
        shape.size()  == static_cast<size_t>(num_boxes[0]) * 2) {
        const int nb_single = static_cast<int>(num_boxes[0]);
        for (int i = 0; i < _batch_size; ++i) {
            if (_num_boxes_vec[i] != nb_single)
                THROW("num_boxes mismatch across samples for single-sample anchor/shape pattern");
            for (int n = 0; n < nb_single; ++n) {
                const size_t dst = (static_cast<size_t>(prefix[i] + n) * 4);
                _anchor_vec[dst + 0] = static_cast<int>(anchor[static_cast<size_t>(n) * 2 + 0]); // x1
                _anchor_vec[dst + 1] = static_cast<int>(anchor[static_cast<size_t>(n) * 2 + 1]); // y1
                _anchor_vec[dst + 2] = static_cast<int>(shape [static_cast<size_t>(n) * 2 + 0]); // w
                _anchor_vec[dst + 3] = static_cast<int>(shape [static_cast<size_t>(n) * 2 + 1]); // h
            }
        }
    }
    // Case 2: fully specified per-sample concatenated anchor/shape
    else if (anchor.size() == _total_boxes * 2 && shape.size() == _total_boxes * 2) {
        for (int i = 0; i < _batch_size; ++i) {
            const int nb = _num_boxes_vec[i];
            const size_t base = static_cast<size_t>(prefix[i]);
            for (int n = 0; n < nb; ++n) {
                const size_t dst = (base + static_cast<size_t>(n)) * 4;
                const size_t src = (base + static_cast<size_t>(n)) * 2;
                _anchor_vec[dst + 0] = static_cast<int>(anchor[src + 0]); // x1
                _anchor_vec[dst + 1] = static_cast<int>(anchor[src + 1]); // y1
                _anchor_vec[dst + 2] = static_cast<int>(shape [src + 0]); // w
                _anchor_vec[dst + 3] = static_cast<int>(shape [src + 1]); // h
            }
        }
    }
    else {
        THROW("Invalid anchor/shape vector sizes");
    }
}

void EraseNode::update_node() {
    if (_use_raw_vectors) {
        // Raw vectors are static; nothing to update per run
        return;
    }
    _num_boxes.update_array();
}

EraseNode::~EraseNode() {
    if (_inputs.empty() || !_inputs[0]) return;
    auto mem_type = _inputs[0]->info().mem_type();

    if (_vx_anchor) vxReleaseTensor(&_vx_anchor);
    if (_vx_colors) vxReleaseTensor(&_vx_colors);
    if (_vx_num_boxes) vxReleaseArray(&_vx_num_boxes);

    if (mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
        if (_anchor_ptr)  {
            hipError_t err = hipHostFree(_anchor_ptr);
            if (err != hipSuccess)
                std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
        }
        if (_color_ptr)  {
            hipError_t err = hipHostFree(_color_ptr);
            if (err != hipSuccess)
                std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
        }
#endif
    } else {
        if (_anchor_ptr) free(_anchor_ptr);
        if (_color_ptr) free(_color_ptr);
    }
    _anchor_ptr = _color_ptr = nullptr;
}
