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

#pragma once
#include <list>
#include <VX/vx.h>

#include "pipeline/graph.h"
#include "pipeline/node.h"
#include "parameters/parameter_vx.h"

// Erase augmentation node: wraps MIVisionX vxExtRppErase
// Inputs:
//   - _inputs[0]: source tensor
//   - anchor_box_info: aux tensor holding per-sample per-box LTRB anchors
//   - colors: aux tensor holding per-sample per-box RGB colors
//   - _num_boxes: per-sample number of boxes (vx_array)
// Output:
//   - _outputs[0]: destination tensor
class EraseNode : public Node {
   public:
    EraseNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    EraseNode() = delete;
    ~EraseNode();

    // Overloads for dynamic vs fixed parameters
    void init(Tensor *anchor_box_info, Tensor *colors, int num_boxes_fixed);
    void init(Tensor *anchor_box_info, Tensor *colors, IntParam *num_boxes_param);
    // New: raw vector-based init (replicates across batch as needed)
    void init(std::vector<float> anchor,
              std::vector<float> shape,
              std::vector<unsigned> num_boxes,
              std::vector<float> fill_value);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    ParameterVX<int> _num_boxes;
    Tensor *_anchor = nullptr;
    Tensor *_colors = nullptr;

    // Raw-vector mode
    bool _use_raw_vectors = false;
    std::vector<int> _anchor_vec;
    std::vector<float> _colors_vec, _fill_values_vec, _fill_values;
    std::vector<unsigned> _num_boxes_vec;
    vx_tensor _vx_anchor = nullptr;
    vx_tensor _vx_colors = nullptr;
    vx_tensor  _vx_num_boxes = nullptr;
    void* _anchor_ptr = nullptr;
    void* _color_ptr = nullptr;
    void* _num_box_ptr = nullptr;
    unsigned _total_boxes = 0;

    // Conservative default range for number of boxes per sample
    constexpr static int NUM_BOXES_RANGE[2] = {0, 1024};
};
