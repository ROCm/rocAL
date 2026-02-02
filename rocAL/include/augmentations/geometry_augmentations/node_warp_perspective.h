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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
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

// WarpPerspective tensor node: applies 3x3 perspective transform per-sample.
// The perspective matrix is provided as 9 floats either per-batch (size 9) replicated,
// or per-sample (size 9 * batch_size).
class WarpPerspectiveNode : public Node {
public:
    WarpPerspectiveNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    WarpPerspectiveNode() = delete;

    // Initialize with a perspective matrix (size == 9 or size == 9 * batch_size)
    void init(const std::vector<float>& perspective_matrix, ResizeInterpolationType interpolation_type);

protected:
    void create_node() override;
    void update_node() override;

private:
    std::vector<float> _perspective; // length 9 or 9 * batch_size
    int _interpolation_type = 0;
};
