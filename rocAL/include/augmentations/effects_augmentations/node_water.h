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

#include "pipeline/node.h"
#include "parameters/parameter_factory.h"
#include "parameters/parameter_vx.h"

class WaterNode : public Node {
   public:
    WaterNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    WaterNode() = delete;
    void init(float amplitude_x, float amplitude_y, float frequency_x, float frequency_y, float phase_x, float phase_y);
    void init(FloatParam *amplitude_x_param, FloatParam *amplitude_y_param, 
              FloatParam *frequency_x_param, FloatParam *frequency_y_param,
              FloatParam *phase_x_param, FloatParam *phase_y_param);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    ParameterVX<float> _amplitude_x;
    ParameterVX<float> _amplitude_y;
    ParameterVX<float> _frequency_x;
    ParameterVX<float> _frequency_y;
    ParameterVX<float> _phase_x;
    ParameterVX<float> _phase_y;
    constexpr static float AMPLITUDE_RANGE[2] = {0.0, 10.0};
    constexpr static float FREQUENCY_RANGE[2] = {0.0, 1.0};
    constexpr static float PHASE_RANGE[2] = {0.0, 6.28318};  // 0 to 2*PI
};
