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

#include "meta_data/coco_meta_data_reader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <utility>

#include "meta_data/lookahead_parser.h"
#include "maskApi.h"

using namespace std;

void COCOMetaDataReader::init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch) {
    _path = cfg.path();
    _avoid_class_remapping = cfg.class_remapping();
    this->set_aspect_ratio_grouping(cfg.get_aspect_ratio_grouping());
    _output = meta_data_batch;
    _output->set_metadata_type(cfg.type());
    _max_width = 0;
    _max_height = 0;
    _rle_masks_by_image.clear();
}

bool COCOMetaDataReader::exists(const std::string &image_name) {
    return _map_content.find(image_name) != _map_content.end();
}

ImgSize COCOMetaDataReader::lookup_image_size(const std::string &image_name) {
    auto it = _map_content.find(image_name);
    if (_map_content.end() == it)
        THROW("ERROR: Given name not present in the map " + image_name)
    return it->second->get_img_size();
}

void COCOMetaDataReader::lookup(const std::vector<std::string> &image_names) {
    if (image_names.empty()) {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    for (unsigned i = 0; i < image_names.size(); i++) {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name)
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
        if (_output->get_metadata_type() == MetaDataType::PixelwiseMask)
            _output->get_pixelwise_labels_batch()[i] = it->second->get_pixelwise_label();
    }
}

void COCOMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, Labels bb_labels, ImgSize image_size, MaskCords mask_cords, std::vector<int> polygon_count, std::vector<std::vector<int>> vertices_count, int image_id) {
    if (exists(image_name)) {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_labels().push_back(bb_labels[0]);
        it->second->get_mask_cords().insert(it->second->get_mask_cords().end(), mask_cords.begin(), mask_cords.end());
        it->second->get_polygon_count().push_back(polygon_count[0]);
        it->second->get_vertices_count().push_back(vertices_count[0]);
        return;
    }
    if (_output->get_metadata_type() == MetaDataType::PolygonMask) {
        pMetaDataPolygonMask info = std::make_shared<PolygonMask>(bb_coords, bb_labels, image_size, mask_cords, polygon_count, vertices_count, image_id);
        _map_content.insert(pair<std::string, std::shared_ptr<PolygonMask>>(image_name, info));
    } else if (_output->get_metadata_type() == MetaDataType::PixelwiseMask) {
        pMetaDataPixelwiseMask info = std::make_shared<PixelwiseMask>(bb_coords, bb_labels, image_size, mask_cords, polygon_count, vertices_count, image_id);
        _map_content.insert(pair<std::string, std::shared_ptr<PixelwiseMask>>(image_name, info));
    }
}

void COCOMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, Labels bb_labels, ImgSize image_size, int image_id) {
    if (exists(image_name)) {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels, image_size, image_id);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void COCOMetaDataReader::print_map_contents() {
    BoundingBoxCords bb_coords;
    Labels bb_labels;
    ImgSize img_size;
    MaskCords mask_cords;
    std::vector<int> polygon_size;
    std::vector<std::vector<int>> vertices_count;

    std::cout << "\nBBox Annotations List: \n";
    for (auto &elem : _map_content) {
        std::cout << "\nName :\t " << elem.first;
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_labels();
        img_size = elem.second->get_img_size();
        std::cout << "<wxh, num of bboxes>: " << img_size.w << " X " << img_size.h << " , " << bb_coords.size() << std::endl;
        for (unsigned int i = 0; i < bb_coords.size(); i++) {
            std::cout << " l : " << bb_coords[i].l << " t: :" << bb_coords[i].t << " r : " << bb_coords[i].r << " b: :" << bb_coords[i].b << "Label Id : " << bb_labels[i] << std::endl;
        }
        if (_output->get_metadata_type() == MetaDataType::PolygonMask) {
            int count = 0;
            mask_cords = elem.second->get_mask_cords();
            polygon_size = elem.second->get_polygon_count();
            vertices_count = elem.second->get_vertices_count();
            std::cout << "\nNumber of objects : " << bb_coords.size() << std::endl;
            for (unsigned int i = 0; i < bb_coords.size(); i++) {
                std::cout << "\nNumber of polygons for object[ << " << i << "]:" << polygon_size[i];
                for (int j = 0; j < polygon_size[i]; j++) {
                    std::cout << "\nPolygon size :" << vertices_count[i][j] << "Elements::";
                    for (int k = 0; k < vertices_count[i][j]; k++, count++)
                        std::cout << "\t " << mask_cords[count];
                }
            }
        }
    }
}

void COCOMetaDataReader::generate_pixelwise_mask(const std::string &filename, const std::vector<RLEMaskInfo> *rle_masks) {
    std::map<int, std::vector<RLE>> label_rles;
    auto it = _map_content.find(filename);
    if (it == _map_content.end()) {
        return;
    }
    auto &bb_labels = it->second->get_labels();
    ImgSize img_size = it->second->get_img_size();
    MaskCords mask_cords = it->second->get_mask_cords();
    std::vector<int> polygon_size = it->second->get_polygon_count();
    std::vector<std::vector<int>> vertices_count = it->second->get_vertices_count();
    auto &pixelwise_labels = it->second->get_pixelwise_label();

    int h = img_size.h;
    int w = img_size.w;
    pixelwise_labels.assign(h * w, 0);

    if (bb_labels.empty()) {
        return;
    }

    // Generate RLEs from all polygons in image.
    int count = 0;
    for (unsigned int i = 0; i < polygon_size.size(); i++) {
        for (int j = 0; j < polygon_size[i]; j++) {
            std::vector<double> in;
            for (int k = 0; k < vertices_count[i][j]; k++, count++) {
                in.push_back(mask_cords[count]);
            }
            auto label = bb_labels[i];
            RLE M;
            rleInit(&M, 0, 0, 0, 0);
            rleFrPoly(&M, in.data(), in.size() / 2, h, w);
            label_rles[label].push_back(M);
        }
    }

    // Add the run-length encoded masks (if any), mapped from mask_idx -> label.
    if (rle_masks) {
        for (const auto &mask : *rle_masks) {
            if (mask.mask_idx < 0 || static_cast<size_t>(mask.mask_idx) >= bb_labels.size()) {
                continue;
            }
            int label = bb_labels[mask.mask_idx];
            RLE M;
            rleInit(&M, 0, 0, 0, 0);
            const int mask_h = (mask.h > 0) ? mask.h : h;
            const int mask_w = (mask.w > 0) ? mask.w : w;
            if (mask_h != h || mask_w != w) {
                std::cerr << "WARNING: RLE mask size mismatch for " << filename << " (mask "
                          << mask_w << "x" << mask_h << " vs image " << w << "x" << h << ")\n";
                continue;
            }
            if (!mask.counts_str.empty()) {
                std::string s = mask.counts_str;
                rleFrString(&M, s.data(), mask_h, mask_w);
            } else if (!mask.counts.empty()) {
                rleInit(&M, mask_h, mask_w, mask.counts.size(), const_cast<uint *>(mask.counts.data()));
            } else {
                continue;
            }
            label_rles[label].push_back(M);
        }
    }

    std::set<int> labels(bb_labels.data(), bb_labels.data() + bb_labels.size());
    if (labels.empty()) {
        for (auto &rles : label_rles)
            for (auto &rle : rles.second)
                rleFree(&rle);
        return;
    }

    RLE *r_out;
    rlesInit(&r_out, *labels.rbegin() + 1);

    for (const auto &rles : label_rles) {
        if (!rles.second.empty())
            rleMerge(rles.second.data(), &r_out[rles.first], rles.second.size(), 0);
    }

    // Find the first non-empty label; an image can contain labels with no segmentation data.
    auto base_label = labels.begin();
    while (base_label != labels.end() && r_out[*base_label].cnts == nullptr) {
        ++base_label;
    }
    if (base_label == labels.end()) {
        rlesFree(&r_out, *labels.rbegin() + 1);
        for (auto &rles : label_rles)
            for (auto &rle : rles.second)
                rleFree(&rle);
        return;
    }

    struct Encoding {
        uint m;
        std::unique_ptr<uint[]> cnts;
        std::unique_ptr<int[]> vals;
    };
    Encoding A;
    A.cnts = std::make_unique<uint[]>(h * w + 1);  // upper-bound
    A.vals = std::make_unique<int[]>(h * w + 1);

    // First copy the content of the first label to the output.
    bool v = false;
    A.m = r_out[*base_label].m;
    for (siz a = 0; a < r_out[*base_label].m; a++) {
        A.cnts[a] = r_out[*base_label].cnts[a];
        A.vals[a] = v ? *base_label : 0;
        v = !v;
    }

    // Then merge the other labels.
    std::unique_ptr<uint[]> cnts = std::make_unique<uint[]>(h * w + 1);
    std::unique_ptr<int[]> vals = std::make_unique<int[]>(h * w + 1);
    for (auto label = std::next(base_label); label != labels.end(); label++) {
        RLE B = r_out[*label];
        if (B.cnts == nullptr)
            continue;

        uint cnt_a = A.cnts[0];
        uint cnt_b = B.cnts[0];
        int next_val_a = A.vals[0];
        int val_a = next_val_a;
        int val_b = *label;
        bool next_vb = false;
        bool vb = next_vb;
        uint nb_seq_a, nb_seq_b;
        nb_seq_a = nb_seq_b = 1;
        int m = 0;

        int cnt_tot = 1;  // check if we advanced at all
        while (cnt_tot > 0) {
            uint c = std::min(cnt_a, cnt_b);
            cnt_tot = 0;
            // advance A
            cnt_a -= c;
            if (!cnt_a && nb_seq_a < A.m) {
                cnt_a = A.cnts[nb_seq_a];  // next sequence for A
                next_val_a = A.vals[nb_seq_a];
                nb_seq_a++;
            }
            cnt_tot += cnt_a;
            // advance B
            cnt_b -= c;
            if (!cnt_b && nb_seq_b < B.m) {
                cnt_b = B.cnts[nb_seq_b++];  // next sequence for B
                next_vb = !next_vb;
            }
            cnt_tot += cnt_b;

            if (val_a && vb) {
                vals[m] = (!cnt_a) ? val_a : val_b;
            } else if (val_a) {
                vals[m] = val_a;
            } else if (vb) {
                vals[m] = val_b;
            } else {
                vals[m] = 0;
            }
            cnts[m] = c;
            m++;

            // since we switched sequence for A or B, apply the new value from now on
            val_a = next_val_a;
            vb = next_vb;

            if (cnt_a == 0) break;
        }
        // copy back the buffers to the destination encoding
        A.m = m;
        for (int i = 0; i < m; i++) A.cnts[i] = cnts[i];
        for (int i = 0; i < m; i++) A.vals[i] = vals[i];
    }

    // Decode final pixelwise masks encoded via RLE and polygons.
    int x = 0, y = 0;
    for (uint i = 0; i < A.m; i++) {
        for (uint j = 0; j < A.cnts[i]; j++) {
            pixelwise_labels[x + y * w] = A.vals[i];
            if (++y >= h) {
                y = 0;
                x++;
            }
        }
    }

    // Destroy RLEs.
    rlesFree(&r_out, *labels.rbegin() + 1);
    for (auto &rles : label_rles)
        for (auto &rle : rles.second)
            rleFree(&rle);
}

void COCOMetaDataReader::read_all(const std::string &path) {
    _coco_metadata_read_time.start();  // Debug timing
    uint32_t max_width = 0, max_height = 0;
    std::ifstream f;
    f.open(path, std::ifstream::in | std::ios::binary);
    if (f.fail()) THROW("ERROR: Given annotations file not present " + path);
    f.ignore(std::numeric_limits<std::streamsize>::max());
    auto file_size = f.gcount();
    f.clear();             //  Since ignore will have set eof.
    if (file_size == 0) {  // If file is empty return
        f.close();
        THROW("ERROR: Given annotations file not valid " + path);
    }
    std::unique_ptr<char, std::function<void(char *)>> buff(
        new char[file_size + 1],
        [](char *data) { delete[] data; });
    f.seekg(0, std::ios::beg);
    buff.get()[file_size] = '\0';
    f.read(buff.get(), file_size);
    f.close();

    LookaheadParser parser(buff.get());

    BoundingBoxCords bb_coords;
    Labels bb_labels;
    ImgSizes img_sizes;
    std::vector<int> polygon_count;
    std::vector<std::vector<int>> vertices_count;

    BoundingBoxCord box;
    ImgSize img_size;
    RAPIDJSON_ASSERT(parser.PeekType() == kObjectType);
    parser.EnterObject();
    while (const char *key = parser.NextObjectKey()) {
        if (0 == std::strcmp(key, "images")) {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            while (parser.NextArrayValue()) {
                int image_id = -1;
                string image_name;
                if (parser.PeekType() != kObjectType) {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey()) {
                    if (0 == std::strcmp(internal_key, "width")) {
                        img_size.w = parser.GetInt();
                        max_width = std::max((uint32_t)img_size.w, max_width);
                    } else if (0 == std::strcmp(internal_key, "height")) {
                        img_size.h = parser.GetInt();
                        max_height = std::max((uint32_t)img_size.h, max_height);
                    } else if (0 == std::strcmp(internal_key, "file_name")) {
                        image_name = parser.GetString();
                    } else if (0 == std::strcmp(internal_key, "id")) {
                        image_id = parser.GetInt();
                    } else {
                        parser.SkipValue();
                    }
                }
                _map_image_names_to_id.insert(pair<int, std::string>(image_id, image_name));
                _map_img_sizes.insert(pair<std::string, ImgSize>(image_name, img_size));
                img_size = {};
            }
        } else if (0 == std::strcmp(key, "categories")) {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            int id = 1;
            std::vector<int> category_ids;
            while (parser.NextArrayValue()) {
                if (parser.PeekType() != kObjectType) {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey()) {
                    if (0 == std::strcmp(internal_key, "id")) {
                        id = parser.GetInt();
                    } else {
                        parser.SkipValue();
                    }
                }
                category_ids.push_back(id);
            }
            std::sort(category_ids.begin(), category_ids.end());
            category_ids.erase(std::unique(category_ids.begin(), category_ids.end()), category_ids.end());
            _label_info.clear();
            int continuous_idx = 1;
            for (int cat_id : category_ids) {
                _label_info[cat_id] = continuous_idx++;
            }
        } else if (0 == std::strcmp(key, "annotations")) {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            while (parser.NextArrayValue()) {
                int id = 1, label = 0, iscrowd = 0;
                std::array<double, 4> bbox;
                std::vector<float> mask;
                std::vector<int> vertices_array;
                int polygon_size = 0;
	                bool has_rle = false;
	                bool rle_valid = true;
	                RLEMaskInfo rle_info;
                if (parser.PeekType() != kObjectType) {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey()) {
                    if (0 == std::strcmp(internal_key, "image_id")) {
                        id = parser.GetInt();
                    } else if (0 == std::strcmp(internal_key, "category_id")) {
                        label = parser.GetInt();
                    } else if (0 == std::strcmp(internal_key, "iscrowd")) {
                        iscrowd = parser.GetInt();
                    } else if (0 == std::strcmp(internal_key, "bbox")) {
                        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                        parser.EnterArray();
                        int i = 0;
                        while (parser.NextArrayValue()) {
                            bbox[i] = parser.GetDouble();
                            ++i;
                        }
                    } else if ((_output->get_metadata_type() == MetaDataType::PolygonMask || _output->get_metadata_type() == MetaDataType::PixelwiseMask) && 0 == std::strcmp(internal_key, "segmentation")) {
                        if (parser.PeekType() == kObjectType && _output->get_metadata_type() == MetaDataType::PixelwiseMask) {
                            parser.EnterObject();
                            int h = -1, w = -1;
                            while (const char *another_key = parser.NextObjectKey()) {
                                if (0 == std::strcmp(another_key, "size")) {
                                    RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                                    parser.EnterArray();
                                    parser.NextArrayValue();
                                    h = parser.GetInt();
                                    parser.NextArrayValue();
                                    w = parser.GetInt();
                                    parser.NextArrayValue();
	                                } else if (0 == std::strcmp(another_key, "counts")) {
	                                    if (parser.PeekType() == kStringType) {
	                                        rle_info.counts_str = parser.GetString();
	                                    } else if (parser.PeekType() == kArrayType) {
	                                        parser.EnterArray();
	                                        while (parser.NextArrayValue()) {
	                                            int v = parser.GetInt();
	                                            if (v < 0) {
	                                                rle_valid = false;
	                                                continue;
	                                            }
	                                            rle_info.counts.push_back(static_cast<uint32_t>(v));
	                                        }
	                                    } else {
	                                        parser.SkipValue();
	                                    }
                                } else {
                                    parser.SkipValue();
                                }
	                            }
	                            rle_info.h = h;
	                            rle_info.w = w;
	                            has_rle = rle_valid && (!rle_info.counts_str.empty() || !rle_info.counts.empty());
	                        } else {
                            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                            parser.EnterArray();
                            while (parser.NextArrayValue()) {
                                polygon_size += 1;
                                int vertex_count = 0;
                                parser.EnterArray();
                                while (parser.NextArrayValue()) {
                                    mask.push_back(parser.GetDouble());
                                    vertex_count += 1;
                                }
                                vertices_array.push_back(vertex_count);
                            }
                        }
                    } else {
                        parser.SkipValue();
                    }
                }

                auto itr = _map_image_names_to_id.find(id);
                auto it = _map_img_sizes.find(itr->second);
                ImgSize image_size = it->second;  // Convert to "ltrb" format
                const bool is_polygon = (_output->get_metadata_type() == MetaDataType::PolygonMask);
                const bool is_pixelwise = (_output->get_metadata_type() == MetaDataType::PixelwiseMask);
                // Polygon masks are represented as polygons in COCO when iscrowd == 0.
                // Pixelwise masks can be generated from polygons (iscrowd == 0) and/or RLE (iscrowd == 1).
                if ((is_polygon && iscrowd == 0) || is_pixelwise) {
                    int mask_idx = 0;
                    if (exists(itr->second)) {
                        mask_idx = _map_content[itr->second]->get_labels().size();
                    }
                    box.l = bbox[0];
                    box.t = bbox[1];
                    box.r = (bbox[0] + bbox[2]);
                    box.b = (bbox[1] + bbox[3]);
                    bb_coords.push_back(box);
                    bb_labels.push_back(label);
                    polygon_count.push_back(polygon_size);
                    vertices_count.push_back(vertices_array);
                    add(itr->second, bb_coords, bb_labels, image_size, mask, polygon_count, vertices_count, id);
                    if (has_rle && is_pixelwise) {
                        rle_info.mask_idx = mask_idx;
                        if (rle_info.h <= 0) rle_info.h = image_size.h;
                        if (rle_info.w <= 0) rle_info.w = image_size.w;
	                        if (rle_info.h != image_size.h || rle_info.w != image_size.w) {
	                            std::cerr << "WARNING: RLE mask size mismatch for " << itr->second << " (mask "
	                                      << rle_info.w << "x" << rle_info.h << " vs image "
	                                      << image_size.w << "x" << image_size.h << ")\n";
	                        } else if (!rle_info.counts.empty()) {
	                            int64_t total = 0;
	                            for (uint32_t c : rle_info.counts) {
	                                total += static_cast<int64_t>(c);
	                            }
	                            int64_t expected = static_cast<int64_t>(rle_info.h) * static_cast<int64_t>(rle_info.w);
	                            if (expected <= 0 || total != expected) {
	                                std::cerr << "WARNING: Invalid RLE counts for " << itr->second
	                                          << " (sum=" << total << " expected=" << expected << ")\n";
	                            } else {
	                                _rle_masks_by_image[itr->second].push_back(std::move(rle_info));
	                            }
	                            has_rle = false;  // handled
	                        }
	                        if (has_rle && rle_info.h == image_size.h && rle_info.w == image_size.w) {
	                            _rle_masks_by_image[itr->second].push_back(std::move(rle_info));
	                        }
	                    }
                    mask.clear();
                    polygon_size = 0;
                    polygon_count.clear();
                    vertices_count.clear();
                    vertices_array.clear();
                    bb_coords.clear();
                    bb_labels.clear();
                } else if (!(_output->get_metadata_type() == MetaDataType::PolygonMask || _output->get_metadata_type() == MetaDataType::PixelwiseMask)) {
                    box.l = bbox[0];
                    box.t = bbox[1];
                    box.r = (bbox[0] + bbox[2]);
                    box.b = (bbox[1] + bbox[3]);
                    bb_coords.push_back(box);
                    bb_labels.push_back(label);
                    add(itr->second, bb_coords, bb_labels, image_size, id);
                    bb_coords.clear();
                    bb_labels.clear();
                }
                image_size = {};
            }
        } else {
            parser.SkipValue();
        }
    }
    for (auto &elem : _map_content) {
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_labels();
        Labels continuous_label_id;
        for (unsigned int i = 0; i < bb_coords.size(); i++) {
            if (_avoid_class_remapping) {
                continuous_label_id.push_back(bb_labels[i]);
                continue;
            }
            auto _it_label = _label_info.find(bb_labels[i]);
            if (_it_label == _label_info.end()) {
                continuous_label_id.push_back(bb_labels[i]);
                continue;
            }
            continuous_label_id.push_back(_it_label->second);
        }
        elem.second->set_labels(continuous_label_id);
    }
    if (_output->get_metadata_type() == MetaDataType::PixelwiseMask) {
        for (auto &elem : _map_content) {
            const std::vector<RLEMaskInfo> *rle_masks = nullptr;
            auto rle_it = _rle_masks_by_image.find(elem.first);
            if (rle_it != _rle_masks_by_image.end())
                rle_masks = &rle_it->second;
            generate_pixelwise_mask(elem.first, rle_masks);
        }
    }
    _max_width = max_width;
    _max_height = max_height;
    _coco_metadata_read_time.end();  // Debug timing
    // print_map_contents();
    //  std::cout << "coco read time in sec: " << _coco_metadata_read_time.get_timing() / 1000 << std::endl;
}

void COCOMetaDataReader::release(std::string image_name) {
    if (!exists(image_name)) {
        WRN("ERROR: Given name not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
    _rle_masks_by_image.erase(image_name);
}

void COCOMetaDataReader::release() {
    _map_content.clear();
    _map_img_sizes.clear();
    _rle_masks_by_image.clear();
}

COCOMetaDataReader::COCOMetaDataReader() : _coco_metadata_read_time("coco meta read time", DBG_TIMING) {
}
