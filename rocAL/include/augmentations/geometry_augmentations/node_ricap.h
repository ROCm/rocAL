/*
Copyright (c) 2025 Advanced Micro Devices, Inc.
All rights reserved.
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
    // Parameter storage from API
    std::vector<unsigned> _permutation_vec;
    std::vector<int> _crop_rois_vec;

    // Internal VX objects created in create_node()
    vx_array _perm_array_vx = nullptr;
    vx_tensor _crop_rois_t = nullptr;

    // Backing buffer for ROI tensor (host or pinned)
    void* _crop_rois_ptr = nullptr;
};
