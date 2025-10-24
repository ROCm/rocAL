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

#include "pipeline/graph.h"
#include "pipeline/node.h"
#include "parameters/parameter_vx.h"

class ThresholdNode : public Node {
   public:
    ThresholdNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ThresholdNode() = delete;

    // Overloads for dynamic vs fixed parameters
    void init(float min_val, float max_val);
    void init(FloatParam *min_param, FloatParam *max_param);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    ParameterVX<float> _min, _max;
    // Default ranges for thresholds on U8 inputs; for floating types values are passed as-is
    constexpr static float THRESHOLD_MIN_RANGE[2] = {0.0f, 255.0f};
    constexpr static float THRESHOLD_MAX_RANGE[2] = {0.0f, 255.0f};
};
