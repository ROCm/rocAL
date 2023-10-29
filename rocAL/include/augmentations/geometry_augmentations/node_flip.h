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

#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"

class FlipNode : public Node {
   public:
    FlipNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    FlipNode() = delete;
    void init(int h_flag, int v_flag, int d_flag);
    void init(IntParam *h_flag_param, IntParam *v_flag_param, IntParam *d_flag_param);
    vx_array get_horizontal_flip() { return _horizontal.default_array(); }
    vx_array get_vertical_flip() { return _vertical.default_array(); }
    vx_array get_depth_flip() { return _depth.default_array(); }

   protected:
    void create_node() override;
    void update_node() override;

   private:
    ParameterVX<int> _horizontal, _vertical, _depth;
    constexpr static int HORIZONTAL_RANGE[2] = {0, 1};
    constexpr static int VERTICAL_RANGE[2] = {0, 1};
    constexpr static int DEPTH_RANGE[2] = {0, 1};
};
