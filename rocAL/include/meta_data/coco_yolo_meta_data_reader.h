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
#include <map>
#include <set>
#include <string>
#include <vector>

#include "pipeline/commons.h"
#include "meta_data/meta_data.h"
#include "meta_data/meta_data_reader.h"
#include "pipeline/timing_debug.h"
#include "pipeline/filesystem.h"

/**
 * @brief COCO YOLO Metadata Reader
 *
 * Reads annotation data from YOLO-style .txt label files (Ultralytics format).
 * Each image has a corresponding .txt file with the same basename containing:
 * - Detection rows: class x_center y_center width height (5 values)
 * - Segmentation rows: class x1 y1 x2 y2 ... xn yn (polygon vertices)
 *
 * Coordinates in YOLO files are normalized [0,1]. This reader converts them
 * to pixel coordinates for compatibility with rocAL's metadata pipeline.
 */
class COCOYoloMetaDataReader : public MetaDataReader {
   public:
    void init(const MetaDataConfig& cfg, pMetaDataBatch meta_data_batch) override;
    void lookup(const std::vector<std::string>& image_names) override;
    ImgSize lookup_image_size(const std::string& image_name) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    bool set_timestamp_mode() override { return false; }
    const std::map<std::string, std::shared_ptr<MetaData>>& get_map_content() override { return _map_content; }
    void set_aspect_ratio_grouping(bool aspect_ratio_grouping) override { _aspect_ratio_grouping = aspect_ratio_grouping; }
    bool get_aspect_ratio_grouping() const override { return _aspect_ratio_grouping; }
    std::vector<std::string> get_relative_file_path() override { return _relative_file_paths; }
    bool exists(const std::string& image_name) override;
    COCOYoloMetaDataReader();

   private:
    pMetaDataBatch _output;
    std::string _labels_path;   // Path to labels directory
    std::string _images_path;   // Path to images directory
    bool _avoid_class_remapping;

    // Add bounding box metadata with PIXEL coords to _map_content
    void add(std::string image_name, BoundingBoxCords bbox, Labels labels, ImgSize image_size, int image_id = 0);
    // Add polygon mask metadata with PIXEL coords to _map_content
    void add(std::string image_name, BoundingBoxCords bbox, Labels labels, ImgSize image_size,
             MaskCords mask_cords, std::vector<int> polygon_count,
             std::vector<std::vector<int>> vertices_count, int image_id = 0);

    // Parse a single YOLO label file and add PIXEL coords to _map_content
    void parse_label_file(const filesys::path& label_path, const std::string& image_key, ImgSize image_size);

    // Convert normalized YOLO (x_center, y_center, w, h) to pixel LTRB
    BoundingBoxCord convert_yolo_to_ltrb(float x_center, float y_center, float width, float height,
                                          int img_width, int img_height);
    // Compute pixel LTRB from normalized polygon vertices
    BoundingBoxCord compute_bbox_from_polygon(const std::vector<float>& polygon_coords,
                                               int img_width, int img_height);
    // Convert normalized polygon coords to pixel coords
    std::vector<float> convert_polygon_to_pixel(const std::vector<float>& norm_coords, int img_width, int img_height);

    // Probe image dimensions from JPEG header
    ImgSize probe_image_size(const filesys::path& image_path);

    // Parse JPEG header to get dimensions
    ImgSize parse_jpeg_header(const std::string& file_path);

    // Normalize lookup key (strip directory and extension) - USED BY exists(), lookup(), lookup_image_size()
    std::string normalize_key(const std::string& image_name);

    std::map<std::string, std::shared_ptr<MetaData>> _map_content;  // ALWAYS contains PIXEL coords
    std::map<std::string, std::shared_ptr<MetaData>>::iterator _itr;
    std::map<std::string, ImgSize> _map_img_sizes;  // basename → {width, height} from image headers
    std::map<int, int> _label_info;                  // For class remapping
    std::set<int> _observed_class_ids;
    std::vector<std::string> _relative_file_paths;  // Full filenames WITH extensions (e.g., "image.jpg")
    TimingDbg _coco_yolo_metadata_read_time;
};
