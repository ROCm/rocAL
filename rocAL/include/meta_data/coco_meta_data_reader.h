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
#include <cstdint>
#include <map>
#include <unordered_map>

#include "pipeline/commons.h"
#include "meta_data/meta_data.h"
#include "meta_data/meta_data_reader.h"
#include "pipeline/timing_debug.h"
#include "maskApi.h"

class COCOMetaDataReader : public MetaDataReader {
   public:
    void init(const MetaDataConfig& cfg, pMetaDataBatch meta_data_batch) override;
    void lookup(const std::vector<std::string>& image_names) override;
    ImgSize lookup_image_size(const std::string& image_name) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    bool set_timestamp_mode() override { return false; }
    std::pair<uint32_t, uint32_t> get_max_size() override { return std::make_pair(_max_height, _max_width); }
    const std::map<std::string, std::shared_ptr<MetaData>>& get_map_content() override { return _map_content; }
    void set_aspect_ratio_grouping(bool aspect_ratio_grouping) override { _aspect_ratio_grouping = aspect_ratio_grouping; }
    bool get_aspect_ratio_grouping() const override { return _aspect_ratio_grouping; }
    COCOMetaDataReader();

   private:
    struct RLEMaskInfo {
        int mask_idx = -1;
        int h = -1;
        int w = -1;
        std::string counts_str;
        std::vector<uint32_t> counts;
    };

    pMetaDataBatch _output;
    std::string _path;
    bool _avoid_class_remapping;
    void add(std::string image_name, BoundingBoxCords bbox, Labels labels, ImgSize image_size, int image_id = 0);
    void add(std::string image_name, BoundingBoxCords bbox, Labels labels, ImgSize image_size, MaskCords mask_cords, std::vector<int> polygon_count, std::vector<std::vector<int>> vertices_count, int image_id = 0);  // To add Mask coordinates to Metadata struct
    bool exists(const std::string& image_name) override;
    void generate_pixelwise_mask(const std::string& filename, const std::vector<RLEMaskInfo>* rle_masks);
    std::map<std::string, std::shared_ptr<MetaData>> _map_content;
    std::map<std::string, std::shared_ptr<MetaData>>::iterator _itr;
    std::map<std::string, ImgSize> _map_img_sizes;
    std::map<int, std::string> _map_image_names_to_id;  // Maps image names to their image IDs
    std::unordered_map<std::string, std::vector<RLEMaskInfo>> _rle_masks_by_image;
    std::map<std::string, ImgSize>::iterator itr;
    std::map<int, int> _label_info = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8},  \
    {9, 9}, {10, 10}, {11, 11}, {13, 12}, {14, 13}, {15, 14}, {16, 15}, {17, 16},  \
    {18, 17}, {19, 18}, {20, 19}, {21, 20}, {22, 21}, {23, 22}, {24, 23}, {25, 24}, {27, 25}, {28, 26},  \
    {31, 27}, {32, 28}, {33, 29}, {34, 30}, {35, 31}, {36, 32}, {37, 33}, {38, 34}, {39, 35}, {40, 36},  \
    {41, 37}, {42, 38}, {43, 39}, {44, 40}, {46, 41}, {47, 42}, {48, 43}, {49, 44}, {50, 45}, {51, 46},  \
    {52, 47}, {53, 48}, {54, 49}, {55, 50}, {56, 51}, {57, 52}, {58, 53}, {59, 54}, {60, 55}, {61, 56},  \
    {62, 57}, {63, 58}, {64, 59}, {65, 60}, {67, 61}, {70, 62}, {72, 63}, {73, 64}, {74, 65}, {75, 66},  \
    {76, 67}, {77, 68}, {78, 69}, {79, 70}, {80, 71}, {81, 72}, {82, 73}, {84, 74}, {85, 75}, {86, 76},  \
    {87, 77}, {88, 78}, {89, 79}, {90, 80}};
    uint32_t _max_width = 0;
    uint32_t _max_height = 0;
    std::map<int, int>::iterator _it_label;
    TimingDbg _coco_metadata_read_time;
};
