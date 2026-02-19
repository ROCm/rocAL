/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include "pipeline/graph.h"
#include "pipeline/node.h"
#include "parameters/parameter_factory.h"
#include "parameters/parameter_vx.h"

class SliceNode : public Node {
   public:
    SliceNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    SliceNode() = delete;
    ~SliceNode();
    void init(Tensor *anchor_param, const std::vector<int> &shape_param, std::vector<float> &fill_values_param, OutOfBoundsPolicy policy);
    void init(Tensor *anchor_param, Tensor *shape_param, std::vector<float> &fill_values_param, OutOfBoundsPolicy policy);

   protected:
    void create_node() override;
    void update_node() override;
    /// Allocate and create a vx_tensor from _shape_vec for the fixed-shape path.
    void create_shape_tensor();

   private:
    void *_shape_array = nullptr;              ///< Raw host/pinned buffer holding the per-sample shape values (fixed-shape mode)
    Tensor *_anchor = nullptr;                 ///< Tensor providing per-sample anchor (starting) coordinates
    Tensor *_shape_tensor = nullptr;           ///< External Tensor providing per-sample shape (tensor-shape mode)
    vx_tensor _shape_tensor_handle = nullptr;  ///< OpenVX tensor handle wrapping _shape_array (fixed-shape mode)
    std::vector<float> _fill_values, _fill_values_vec;  ///< Fill values for out-of-bounds padding, expanded to batch size
    std::vector<int> _shape_vec;               ///< Fixed shape dimensions (excluding batch), used when _use_shape_tensor is false
    bool _use_shape_tensor = false;            ///< Selects between fixed-shape (false) and tensor-shape (true) modes
    OutOfBoundsPolicy _policy = OutOfBoundsPolicy::ERROR;  ///< Policy for handling out-of-bounds regions
};
