/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <array>

#include "pipeline/graph.h"
#include "pipeline/node.h"
#include "parameters/parameter_factory.h"
#include "parameters/parameter_vx.h"

class SpatterNode : public Node {
   public:
    SpatterNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    SpatterNode() = delete;
    void init(uint8_t red, uint8_t green, uint8_t blue);
    void init(IntParam *red, IntParam *green, IntParam *blue);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    ParameterVX<int> _red_param;
    ParameterVX<int> _green_param;
    ParameterVX<int> _blue_param;
    constexpr static int COLOR_RANGE[2] = {0, 255};
    vx_array _color_array;
    std::array<vx_uint8, 3> _color;
};
