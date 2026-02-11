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

#include "meta_data/coco_yolo_meta_data_reader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>

using namespace std;

namespace {
constexpr size_t kYoloBboxCoordsCount = 4;
constexpr size_t kMinPolygonCoordsCount = 6;
constexpr int kRectVertices = 4;
constexpr int kCoordsPerVertex = 2;
constexpr int kRectPolygonCoordsCount = kRectVertices * kCoordsPerVertex;
}  // namespace

COCOYoloMetaDataReader::COCOYoloMetaDataReader() : _coco_yolo_metadata_read_time("coco yolo meta read time", DBG_TIMING) {
}

void COCOYoloMetaDataReader::init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch) {
    _labels_path = cfg.path();
    _images_path = cfg.images_path();
    _avoid_class_remapping = cfg.class_remapping();
    this->set_aspect_ratio_grouping(cfg.get_aspect_ratio_grouping());
    _output = meta_data_batch;
    _output->set_metadata_type(cfg.type());
}

std::string COCOYoloMetaDataReader::normalize_key(const std::string &image_name) {
    // Strip directory path if present
    std::string result = image_name;
    auto last_slash = result.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        result = result.substr(last_slash + 1);
    }
    // Strip extension if present
    auto dot_pos = result.find_last_of('.');
    if (dot_pos != std::string::npos) {
        result = result.substr(0, dot_pos);
    }
    return result;
}

bool COCOYoloMetaDataReader::exists(const std::string &image_name) {
    std::string key = normalize_key(image_name);
    return _map_content.find(key) != _map_content.end();
}

ImgSize COCOYoloMetaDataReader::lookup_image_size(const std::string &image_name) {
    std::string key = normalize_key(image_name);
    auto it = _map_img_sizes.find(key);
    if (_map_img_sizes.end() == it)
        THROW("ERROR: Given name not present in the image size map: " + image_name)
    return it->second;
}

void COCOYoloMetaDataReader::lookup(const std::vector<std::string> &image_names) {
    if (image_names.empty()) {
        ERR("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    for (unsigned i = 0; i < image_names.size(); i++) {
        std::string key = normalize_key(image_names[i]);
        auto it = _map_content.find(key);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map: " + image_names[i] + " (key: " + key + ")")
        _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
        _output->get_labels_batch()[i] = it->second->get_labels();
        _output->get_img_sizes_batch()[i] = it->second->get_img_size();
        _output->get_image_id_batch()[i] = it->second->get_image_id();
        if (_output->get_metadata_type() == MetaDataType::PolygonMask) {
            auto mask_cords = it->second->get_mask_cords();
            _output->get_mask_cords_batch()[i] = mask_cords;
            _output->get_mask_polygons_count_batch()[i] = it->second->get_polygon_count();
            _output->get_mask_vertices_count_batch()[i] = it->second->get_vertices_count();
        }
    }
}

void COCOYoloMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, Labels bb_labels, ImgSize image_size, MaskCords mask_cords, std::vector<int> polygon_count, std::vector<std::vector<int>> vertices_count, int image_id) {
    if (exists(image_name)) {
        std::string key = normalize_key(image_name);
        auto it = _map_content.find(key);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_labels().push_back(bb_labels[0]);
        it->second->get_mask_cords().insert(it->second->get_mask_cords().end(), mask_cords.begin(), mask_cords.end());
        it->second->get_polygon_count().push_back(polygon_count[0]);
        it->second->get_vertices_count().push_back(vertices_count[0]);
        return;
    }
    std::string key = normalize_key(image_name);
    pMetaDataPolygonMask info = std::make_shared<PolygonMask>(bb_coords, bb_labels, image_size, mask_cords, polygon_count, vertices_count, image_id);
    _map_content.insert(pair<std::string, std::shared_ptr<PolygonMask>>(key, info));
}

void COCOYoloMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, Labels bb_labels, ImgSize image_size, int image_id) {
    std::string key = normalize_key(image_name);
    if (_map_content.find(key) != _map_content.end()) {
        auto it = _map_content.find(key);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels, image_size, image_id);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(key, info));
}

BoundingBoxCord COCOYoloMetaDataReader::convert_yolo_to_ltrb(float x_center, float y_center, float width, float height, int img_width, int img_height) {
    // Clamp normalized values to [0, 1]
    x_center = std::max(0.0f, std::min(1.0f, x_center));
    y_center = std::max(0.0f, std::min(1.0f, y_center));
    width = std::max(0.0f, std::min(1.0f, width));
    height = std::max(0.0f, std::min(1.0f, height));

    // Convert normalized center/size to pixel LTRB
    float half_w = width / 2.0f;
    float half_h = height / 2.0f;

    float l = (x_center - half_w) * img_width;
    float t = (y_center - half_h) * img_height;
    float r = (x_center + half_w) * img_width;
    float b = (y_center + half_h) * img_height;

    // Clamp to image bounds
    l = std::max(0.0f, std::min(static_cast<float>(img_width), l));
    t = std::max(0.0f, std::min(static_cast<float>(img_height), t));
    r = std::max(0.0f, std::min(static_cast<float>(img_width), r));
    b = std::max(0.0f, std::min(static_cast<float>(img_height), b));

    return BoundingBoxCord(l, t, r, b);
}

BoundingBoxCord COCOYoloMetaDataReader::compute_bbox_from_polygon(const std::vector<float>& polygon_coords, int img_width, int img_height) {
    if (polygon_coords.size() < 4) {
        return BoundingBoxCord(0, 0, 0, 0);
    }

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    // polygon_coords contains already converted pixel coordinates
    for (size_t i = 0; i < polygon_coords.size(); i += 2) {
        float x = polygon_coords[i];
        float y = polygon_coords[i + 1];
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
    }

    // Clamp to image bounds
    min_x = std::max(0.0f, std::min(static_cast<float>(img_width), min_x));
    min_y = std::max(0.0f, std::min(static_cast<float>(img_height), min_y));
    max_x = std::max(0.0f, std::min(static_cast<float>(img_width), max_x));
    max_y = std::max(0.0f, std::min(static_cast<float>(img_height), max_y));

    return BoundingBoxCord(min_x, min_y, max_x, max_y);
}

std::vector<float> COCOYoloMetaDataReader::convert_polygon_to_pixel(const std::vector<float>& norm_coords, int img_width, int img_height) {
    std::vector<float> pixel_coords;
    pixel_coords.reserve(norm_coords.size());
    for (size_t i = 0; i < norm_coords.size(); i += 2) {
        float x = std::max(0.0f, std::min(1.0f, norm_coords[i])) * img_width;
        float y = std::max(0.0f, std::min(1.0f, norm_coords[i + 1])) * img_height;
        pixel_coords.push_back(x);
        pixel_coords.push_back(y);
    }
    return pixel_coords;
}

ImgSize COCOYoloMetaDataReader::parse_jpeg_header(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        THROW("Failed to open JPEG file for header parsing: " + file_path);
    }

    unsigned char buf[2];
    file.read(reinterpret_cast<char*>(buf), 2);
    if (buf[0] != 0xFF || buf[1] != 0xD8) {
        THROW("Invalid JPEG file (missing SOI marker): " + file_path);
    }

    while (file.good()) {
        // Read marker
        file.read(reinterpret_cast<char*>(buf), 2);
        if (buf[0] != 0xFF) {
            continue;  // Skip non-marker bytes
        }

        unsigned char marker = buf[1];

        // Skip padding bytes (0xFF)
        while (marker == 0xFF && file.good()) {
            file.read(reinterpret_cast<char*>(&marker), 1);
        }

        // SOF0, SOF1, SOF2 markers contain image dimensions
        if (marker >= 0xC0 && marker <= 0xC2) {
            unsigned char header[7];
            file.read(reinterpret_cast<char*>(header), 7);
            // header[0-1]: segment length
            // header[2]: precision
            // header[3-4]: height (big-endian)
            // header[5-6]: width (big-endian)
            int height = (header[3] << 8) | header[4];
            int width = (header[5] << 8) | header[6];
            return ImgSize{width, height};
        }

        // Read segment length and skip
        unsigned char len_buf[2];
        file.read(reinterpret_cast<char*>(len_buf), 2);
        int segment_length = (len_buf[0] << 8) | len_buf[1];
        if (segment_length > 2) {
            file.seekg(segment_length - 2, std::ios::cur);
        }
    }

    THROW("Failed to find SOF marker in JPEG file: " + file_path);
}

ImgSize COCOYoloMetaDataReader::probe_image_size(const filesys::path& image_path) {
    std::string path_str = image_path.string();
    std::string ext = image_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    ImgSize size;
    if (ext != ".jpg" && ext != ".jpeg") {
        THROW("Unsupported image format for size probing (JPEG only): " + ext);
    }
    size = parse_jpeg_header(path_str);

    return size;
}

void COCOYoloMetaDataReader::parse_label_file(const filesys::path& label_path, const std::string& image_key, ImgSize image_size) {
    std::ifstream file(label_path.string());
    if (!file.is_open()) {
        ERR("Failed to open label file: " + label_path.string());
        bool is_polygon_mode = (_output->get_metadata_type() == MetaDataType::PolygonMask);
        BoundingBoxCords empty_bbox;
        Labels empty_labels;
        if (is_polygon_mode) {
            MaskCords empty_mask;
            std::vector<int> empty_polygon_count;
            std::vector<std::vector<int>> empty_vertices_count;
            add(image_key, empty_bbox, empty_labels, image_size, empty_mask, empty_polygon_count, empty_vertices_count);
        } else {
            add(image_key, empty_bbox, empty_labels, image_size);
        }
        return;
    }

    bool is_polygon_mode = (_output->get_metadata_type() == MetaDataType::PolygonMask);
    int img_width = image_size.w;
    int img_height = image_size.h;

    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;

        // Skip empty or whitespace-only lines
        if (line.find_first_not_of(" \t\n\r") == std::string::npos) {
            continue;
        }

        std::istringstream iss(line);
        std::vector<float> tokens;
        int class_id;

        // Read class ID first
        if (!(iss >> class_id)) {
            ERR("Invalid line format (missing class ID) in " + label_path.string() + " line " + std::to_string(line_num));
            continue;
        }
        if (class_id < 0) {
            ERR("Invalid class ID (negative) in " + label_path.string() + " line " + std::to_string(line_num));
            continue;
        }

        // Read remaining floats
        float val;
        while (iss >> val) {
            tokens.push_back(val);
        }

        if (tokens.size() == kYoloBboxCoordsCount) {
            // Detection format: x_center y_center width height
            if (tokens[2] <= 0.0f || tokens[3] <= 0.0f) {
                ERR("Invalid bbox width/height in " + label_path.string() + " line " + std::to_string(line_num));
                continue;
            }
            BoundingBoxCord box = convert_yolo_to_ltrb(tokens[0], tokens[1], tokens[2], tokens[3], img_width, img_height);
            BoundingBoxCords bbox;
            bbox.push_back(box);
            Labels labels;
            labels.push_back(class_id);
            _observed_class_ids.insert(class_id);

            if (is_polygon_mode) {
                // For polygon mode, create a rectangular polygon from the bbox
                MaskCords mask_cords;
                // Rectangle vertices: top-left, top-right, bottom-right, bottom-left
                mask_cords.push_back(box.l);
                mask_cords.push_back(box.t);
                mask_cords.push_back(box.r);
                mask_cords.push_back(box.t);
                mask_cords.push_back(box.r);
                mask_cords.push_back(box.b);
                mask_cords.push_back(box.l);
                mask_cords.push_back(box.b);

                std::vector<int> polygon_count = {1};  // One polygon per object
                std::vector<std::vector<int>> vertices_count = {{kRectPolygonCoordsCount}};  // 4 vertices x 2 coords each

                add(image_key, bbox, labels, image_size, mask_cords, polygon_count, vertices_count);
            } else {
                add(image_key, bbox, labels, image_size);
            }
        } else if (tokens.size() >= kMinPolygonCoordsCount && tokens.size() % 2 == 0) {
            // Segmentation format: x1 y1 x2 y2 ... xn yn (polygon vertices)
            std::vector<float> pixel_coords = convert_polygon_to_pixel(tokens, img_width, img_height);
            BoundingBoxCord box = compute_bbox_from_polygon(pixel_coords, img_width, img_height);

            BoundingBoxCords bbox;
            bbox.push_back(box);
            Labels labels;
            labels.push_back(class_id);
            _observed_class_ids.insert(class_id);

            if (is_polygon_mode) {
                MaskCords mask_cords(pixel_coords.begin(), pixel_coords.end());
                std::vector<int> polygon_count = {1};
                std::vector<std::vector<int>> vertices_count = {{static_cast<int>(pixel_coords.size())}};

                add(image_key, bbox, labels, image_size, mask_cords, polygon_count, vertices_count);
            } else {
                add(image_key, bbox, labels, image_size);
            }
        } else {
            ERR("Invalid annotation format in " + label_path.string() + " line " + std::to_string(line_num) +
                " (expected " + std::to_string(kYoloBboxCoordsCount) + " values for bbox or >= " + std::to_string(kMinPolygonCoordsCount) +
                " even values for polygon, got " + std::to_string(tokens.size()) + ")");
        }
    }

    // If file was empty or had no valid annotations, create an empty entry (background image)
    std::string key = normalize_key(image_key);
    if (_map_content.find(key) == _map_content.end()) {
        BoundingBoxCords empty_bbox;
        Labels empty_labels;
        if (is_polygon_mode) {
            MaskCords empty_mask;
            std::vector<int> empty_polygon_count;
            std::vector<std::vector<int>> empty_vertices_count;
            add(image_key, empty_bbox, empty_labels, image_size, empty_mask, empty_polygon_count, empty_vertices_count);
        } else {
            add(image_key, empty_bbox, empty_labels, image_size);
        }
    }

}

void COCOYoloMetaDataReader::read_all(const std::string &path) {
    _coco_yolo_metadata_read_time.start();

    if (!filesys::exists(path) || !filesys::is_directory(path)) {
        THROW("Labels directory does not exist or is not a directory: " + path);
    }

    if (!filesys::exists(_images_path) || !filesys::is_directory(_images_path)) {
        THROW("Images directory does not exist or is not a directory: " + _images_path);
    }

    // Index label files by stem (basename) so we can iterate images and attach labels if present.
    std::unordered_map<std::string, filesys::path> label_by_stem;
    for (const auto& entry : filesys::directory_iterator(path)) {
        if (!entry.is_regular_file())
            continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext != ".txt")
            continue;
        std::string stem = entry.path().stem().string();
        auto [it, inserted] = label_by_stem.emplace(stem, entry.path());
        if (!inserted) {
            ERR("Duplicate label file stem detected, using first occurrence. Ignoring: " + entry.path().string());
        }
    }

    int files_processed = 0;
    int files_skipped = 0;

    // Iterate images to ensure every decoded image has a metadata entry (empty if missing a label file).
    for (const auto& img_entry : filesys::directory_iterator(_images_path)) {
        if (!img_entry.is_regular_file())
            continue;

        auto image_path = img_entry.path();
        std::string ext = image_path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext != ".jpg" && ext != ".jpeg")
            continue;

        std::string basename = image_path.stem().string();

        ImgSize img_size;
        try {
            img_size = probe_image_size(image_path);
        } catch (const std::exception& e) {
            ERR("Failed to probe image size for " + image_path.string() + ": " + e.what());
            files_skipped++;
            continue;
        }
        if (img_size.w <= 0 || img_size.h <= 0) {
            ERR("Invalid probed image size for " + image_path.string() + " (w=" + std::to_string(img_size.w) + ", h=" + std::to_string(img_size.h) + ")");
            files_skipped++;
            continue;
        }

        _map_img_sizes[basename] = img_size;
        _relative_file_paths.push_back(image_path.filename().string());

        auto label_it = label_by_stem.find(basename);
        if (label_it != label_by_stem.end()) {
            parse_label_file(label_it->second, basename, img_size);
            files_processed++;
        } else {
            // No label file: add empty entry so lookup() won't throw during decode.
            bool is_polygon_mode = (_output->get_metadata_type() == MetaDataType::PolygonMask);
            BoundingBoxCords empty_bbox;
            Labels empty_labels;
            if (is_polygon_mode) {
                MaskCords empty_mask;
                std::vector<int> empty_polygon_count;
                std::vector<std::vector<int>> empty_vertices_count;
                add(basename, empty_bbox, empty_labels, img_size, empty_mask, empty_polygon_count, empty_vertices_count);
            } else {
                add(basename, empty_bbox, empty_labels, img_size);
            }
        }
    }

    // Warn about label files with no matching image.
    for (const auto& kv : label_by_stem) {
        if (_map_img_sizes.find(kv.first) == _map_img_sizes.end()) {
            ERR("No matching image found for label file: " + kv.second.string());
        }
    }

    // Apply class remapping if needed
    if (!_avoid_class_remapping && !_observed_class_ids.empty()) {
        // Build sorted list of class IDs
        std::vector<int> sorted_ids(_observed_class_ids.begin(), _observed_class_ids.end());
        std::sort(sorted_ids.begin(), sorted_ids.end());

        // Create mapping from original ID to a dense, continuous index range.
        // We intentionally use 1-based indices here to match the COCO reader and other
        // parts of the pipeline that assume class IDs start at 1.
        //
        // If the source annotations are 0-based (i.e., they contain class_id 0), then 0
        // will be treated like any other valid class and remapped to 1, 1 -> 2, etc.
        for (size_t i = 0; i < sorted_ids.size(); i++) {
            _label_info[sorted_ids[i]] = static_cast<int>(i + 1);
        }

        // Apply remapping to all stored metadata
        for (auto& elem : _map_content) {
            Labels& labels = elem.second->get_labels();
            Labels remapped_labels;
            for (int label : labels) {
                auto it = _label_info.find(label);
                if (it != _label_info.end()) {
                    remapped_labels.push_back(it->second);
                } else {
                    remapped_labels.push_back(label);
                }
            }
            elem.second->set_labels(remapped_labels);
        }
    }

    (void)files_processed;
    (void)files_skipped;
    _coco_yolo_metadata_read_time.end();
    LOG("COCOYoloMetaDataReader: Processed " + std::to_string(files_processed) + " label files, skipped " +
        std::to_string(files_skipped) + " files");
}

void COCOYoloMetaDataReader::print_map_contents() {
    BoundingBoxCords bb_coords;
    Labels bb_labels;
    ImgSize img_size;
    MaskCords mask_cords;
    std::vector<int> polygon_size;
    std::vector<std::vector<int>> vertices_count;

    std::cout << "\nBBox Annotations List (YOLO): \n";
    for (auto &elem : _map_content) {
        std::cout << "\nName:\t " << elem.first;
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_labels();
        img_size = elem.second->get_img_size();
        std::cout << " <wxh, num of bboxes>: " << img_size.w << " X " << img_size.h << " , " << bb_coords.size() << std::endl;
        for (unsigned int i = 0; i < bb_coords.size(); i++) {
            std::cout << " l: " << bb_coords[i].l << " t: " << bb_coords[i].t << " r: " << bb_coords[i].r << " b: " << bb_coords[i].b << " Label Id: " << bb_labels[i] << std::endl;
        }
        if (_output->get_metadata_type() == MetaDataType::PolygonMask) {
            int count = 0;
            mask_cords = elem.second->get_mask_cords();
            polygon_size = elem.second->get_polygon_count();
            vertices_count = elem.second->get_vertices_count();
            std::cout << "\nNumber of objects: " << bb_coords.size() << std::endl;
            for (unsigned int i = 0; i < bb_coords.size(); i++) {
                std::cout << "\nNumber of polygons for object[" << i << "]: " << polygon_size[i];
                for (int j = 0; j < polygon_size[i]; j++) {
                    std::cout << "\nPolygon size: " << vertices_count[i][j] << " Elements: ";
                    for (int k = 0; k < vertices_count[i][j]; k++, count++)
                        std::cout << "\t " << mask_cords[count];
                }
            }
        }
    }
}

void COCOYoloMetaDataReader::release(std::string image_name) {
    std::string key = normalize_key(image_name);
    if (_map_content.find(key) == _map_content.end()) {
        ERR("Given name not present in the map: " + image_name);
        return;
    }
    _map_content.erase(key);
}

void COCOYoloMetaDataReader::release() {
    _map_content.clear();
    _map_img_sizes.clear();
    _observed_class_ids.clear();
    _label_info.clear();
    _relative_file_paths.clear();
}
