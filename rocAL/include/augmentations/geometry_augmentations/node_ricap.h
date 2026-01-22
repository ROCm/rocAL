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
#include "parameters/parameter_vx.h"

// Ricap tensor node: Random Image Cropping And Patching.
// Inputs:
//  - Single input tensor
//  - Outputs a tensor with same dims/layout (unless user changes layout via output tensor info)
// Parameters provided via init:
//  - permutation vector: length N*4 (4 indices per output sample selecting source samples per quadrant)
//  - crop_rois vector: per-sample 4 ROIs (XYWH/LTRB as per roiType) flattened; size either 16 or N*16 ints
//    Order per sample: roi0[4], roi1[4], roi2[4], roi3[4]
class RicapNode : public Node {
public:
    RicapNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    RicapNode() = delete;
    ~RicapNode();

    // permutation: length 4 or N*4 (replicated if 4)
    // crop_rois: length 16 or N*16 (replicated if 16) with per-sample 4x [x,y,w,h] or [l,t,r,b]
    void init(const std::vector<unsigned>& permutation,
              const std::vector<int>& crop_rois);

protected:
    void create_node() override;
    void update_node() override;

private:
    // Unified helper method for tensor creation with optional replication
    template<typename T>
    vx_tensor create_tensor_with_replication(
        const std::vector<T>& input_vec,
        vx_size N,
        vx_size elems_per_sample,
        void** backing_ptr,
        RocalMemType mem_type,
        vx_enum vx_data_type,
        std::vector<vx_size>&& num_dims);

    // Parameter storage from API
    std::vector<unsigned> _permutation_vec;
    std::vector<int> _crop_rois_vec;

    // Internal VX objects created in create_node()
    vx_tensor _perm_tensor_vx = nullptr;
    vx_tensor _crop_rois_t = nullptr;

    // Backing buffers for tensors (host or pinned)
    void* _perm_ptr = nullptr;
    void* _crop_rois_ptr = nullptr;
};
