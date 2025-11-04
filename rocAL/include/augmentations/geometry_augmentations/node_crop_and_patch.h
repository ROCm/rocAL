/*
Copyright (c) 2025 Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once
#include "pipeline/graph.h"
#include "pipeline/node.h"
#include "parameters/parameter_factory.h"
#include "parameters/parameter_vx.h"

// CropAndPatch tensor node: crops a region from input1 and patches into input2 (or vice versa)
// based on the per-sample ROI tensors provided for destination, crop, and patch regions.
class CropAndPatchNode : public Node {
public:
    CropAndPatchNode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    CropAndPatchNode() = delete;
    ~CropAndPatchNode();

    // Initialize with ROI vectors (either size 4 or N*4)
    // Each vector supports per-sample [x, y, w, h] or [l, t, r, b] based on roi type.
    void init(const std::vector<int>& crop_roi_vec,
              const std::vector<int>& patch_roi_vec);

protected:
    void create_node() override;
    void update_node() override;

private:
    // Store ROI specs provided by API; tensors will be created internally in create_node()
    std::vector<int> _crop_roi_vec;
    std::vector<int> _patch_roi_vec;

    // Internal OpenVX tensors created from ROI vectors (backed by external handles)
    vx_tensor _crop_roi_t  = nullptr;
    vx_tensor _patch_roi_t = nullptr;

    // Backing buffers (host or pinned) for ROI tensors
    void* _crop_roi_ptr  = nullptr;
    void* _patch_roi_ptr = nullptr;
};
