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

#include "pipeline/node.h"
#include "parameters/parameter_factory.h"
#include "parameters/parameter_vx.h"

class RainNode : public Node {
   public:
    RainNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    RainNode() = delete;
    void init(float rain_percentage, int rain_width, int rain_height, float rain_slant_angle, float rain_transparency);
    void init(float rain_percentage, int rain_width, int rain_height, float rain_slant_angle, FloatParam *rain_transparency);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    float _rain_percentage;
    int _rain_width;
    int _rain_height;
    float _rain_slant_angle;
    ParameterVX<float> _rain_transparency;
    constexpr static float RAIN_TRANSPARENCY_RANGE[2] = {0.2, 0.3};
};