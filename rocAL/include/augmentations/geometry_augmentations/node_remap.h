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
#include "pipeline/graph.h"
#include "pipeline/node.h"
#include "parameters/parameter_factory.h"
#include "parameters/parameter_vx.h"

class RemapNode : public Node {
public:
    RemapNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    RemapNode() = delete;
    ~RemapNode();

    // Initialize with row/col remap vectors and interpolation policy.
    // Vectors can be size H*W (replicated for all N) or N*H*W (per-sample).
    void init(const std::vector<float>& row_remap_vec,
              const std::vector<float>& col_remap_vec,
              ResizeInterpolationType interpolation_type);

protected:
    void create_node() override;
    void update_node() override;

private:
    // Data provided by API
    std::vector<float> _row_remap_vec;
    std::vector<float> _col_remap_vec;

    // Internal vx_tensors created from vectors (backed by external handles)
    vx_tensor _row_tbl = nullptr;
    vx_tensor _col_tbl = nullptr;

    // Backing buffers (host or pinned) for remap tables
    void* _row_tbl_ptr = nullptr;
    void* _col_tbl_ptr = nullptr;

    int _interpolation_type = 0;
};
