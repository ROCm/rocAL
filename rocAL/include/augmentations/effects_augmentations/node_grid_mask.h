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

class GridMaskNode : public Node {
   public:
    GridMaskNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    GridMaskNode() = delete;

    // Scalar-only parameters (uniform across the batch)
    void init(unsigned tile_width, float grid_ratio, float grid_angle_radians, unsigned translate_x, unsigned translate_y);

   protected:
    void create_node() override;
    void update_node() override;

   private:
    unsigned _tile_width = 0;
    float _grid_ratio = 0.0f;
    float _grid_angle = 0.0f; // radians
    unsigned _translate_x = 0;
    unsigned _translate_y = 0;
};
