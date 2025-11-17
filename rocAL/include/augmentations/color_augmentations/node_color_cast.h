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

#pragma once
#include "pipeline/node.h"
#include "parameters/parameter_factory.h"
#include "parameters/parameter_vx.h"

class ColorCastNode : public Node {
   public:
    ColorCastNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ColorCastNode() = delete;

    // Per-sample alpha as random parameter, and RGB triplet(s)
    void init(FloatParam *alpha_param, std::vector<float> rgb);
    // Fixed alpha and RGB triplet(s)
    void init(float alpha, std::vector<float> rgb);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    // Per-sample alpha (size batch_size)
    ParameterVX<float> _alpha;
    // Flat RGB triplets (size batch_size * 3)
    std::vector<float> _rgb;
    vx_array _rgb_vx_array = nullptr;

    // Reasonable range for alpha [0, 1]
    constexpr static float ALPHA_RANGE[2] = {0.0f, 1.0f};
};
