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

#include "graph.h"
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"

class GaussianNoiseNode : public Node {
   public:
    GaussianNoiseNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    GaussianNoiseNode() = delete;
    void init(float mean, float std_dev, int seed, int conditional_execution);
    void init(FloatParam *mean_param, FloatParam *stddev_param, int seed, IntParam *condition_execution_param);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    ParameterVX<float> _mean, _stddev;
    ParameterVX<int> _conditional_execution;
    constexpr static float MEAN_RANGE[2] = {0, 5};
    constexpr static float STDDEV_RANGE[2] = {1, 5};
    constexpr static int CONDITIONAL_EXECUTION_RANGE[2] = {0, 1};
    int _seed;
};
