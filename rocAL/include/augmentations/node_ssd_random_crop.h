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
#include "augmentations/geometry_augmentations/node_crop.h"
#include "parameters/parameter_crop_factory.h"
#include "parameters/parameter_factory.h"
#include "pipeline/seed_rng.h"

class SSDRandomCropNode : public CropNode {
   public:
    SSDRandomCropNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    SSDRandomCropNode() = delete;
    void init(FloatParam *crop_area_factor, FloatParam *crop_aspect_ratio, FloatParam *x_drift, FloatParam *y_drift, int num_of_attempts);
    unsigned int get_dst_width() { return _outputs[0]->info().max_shape()[0]; }
    unsigned int get_dst_height() { return _outputs[0]->info().max_shape()[1]; }
    std::shared_ptr<RocalRandomCropParam> get_crop_param() { return _crop_param; }
    float get_threshold() { return _threshold; }
    std::vector<std::pair<float, float>> get_iou_range() { return _iou_range; }
    bool is_entire_iou() { return _entire_iou; }
    void set_meta_data_batch() {}

   protected:
    void create_node() override;
    void update_node() override;

   private:
    std::shared_ptr<RocalRandomCropParam> _meta_crop_param;
    std::vector<uint> _x1_val, _y1_val, _crop_width_val, _crop_height_val;
    size_t _dest_width;
    size_t _dest_height;
    float _threshold = 0.05;
    std::vector<std::pair<float, float>> _iou_range;
    int _num_of_attempts = 20;
    bool _entire_iou = false;
    std::shared_ptr<RocalRandomCropParam> _crop_param;
    SeededRNG<std::mt19937, 4> _rngs;  // setting the state_size to 4 for 4 random parameters.
};
