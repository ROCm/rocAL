/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "pipeline/select_mask_polygon.h"

#include <map>
#include <set>

SelectMaskPolygon::SelectMaskPolygon(size_t user_batch_size) : _user_batch_size(user_batch_size) {}

// Filter per-sample polygon masks to only include objects whose index appears in mask_ids.
// For each selected object, copies its polygon vertex coordinates into the output buffer
// and records the per-polygon vertex counts and mask IDs. When reindex_mask is true,
// the output mask IDs are remapped to sequential indices (0, 1, 2, ...) based on their
// position in the mask_ids list; otherwise the original object indices are preserved.
TensorList *SelectMaskPolygon::run(rocalTensorList *mask_data,
                                  const std::vector<std::vector<int>> &polygon_counts,
                                  const std::vector<std::vector<std::vector<int>>> &vertices_counts,
                                  const std::vector<int> &mask_ids,
                                  std::vector<std::vector<int>> &sel_vertices_counts,
                                  std::vector<std::vector<int>> &sel_mask_ids,
                                  bool reindex_mask,
                                  TensorList &out_list) {
    std::set<int> unique_ids(mask_ids.begin(), mask_ids.end());
    if (unique_ids.size() != mask_ids.size())
        THROW("mask_ids should not contain duplicates");

    _output.clear();
    _output.resize(_user_batch_size);
    sel_vertices_counts.resize(_user_batch_size);
    sel_mask_ids.resize(_user_batch_size);

    std::map<int, int> mask_id_to_idx;
    for (size_t idx = 0; idx < mask_ids.size(); idx++) {
        mask_id_to_idx[mask_ids[idx]] = static_cast<int>(idx);
    }

    for (unsigned i = 0; i < _user_batch_size; i++) {
        float *mask_buffer = static_cast<float *>(mask_data->at(i)->buffer());
        auto objects = polygon_counts[i].size();
        for (auto mask_id : mask_ids) {
            if (mask_id < 0 || static_cast<size_t>(mask_id) >= objects)
                THROW("Requested mask id " + std::to_string(mask_id) + " is not present in the sample");
        }

        size_t buffer_offset = 0;
        for (unsigned obj_idx = 0; obj_idx < objects; obj_idx++) {
            bool select_object = unique_ids.find(static_cast<int>(obj_idx)) != unique_ids.end();
            for (unsigned poly_idx = 0; poly_idx < static_cast<unsigned>(polygon_counts[i][obj_idx]); poly_idx++) {
                auto vertex_count = vertices_counts[i][obj_idx][poly_idx];
                if (select_object) {
                    for (int v = 0; v < vertex_count; v++) {
                        _output[i].push_back(mask_buffer[buffer_offset + v]);
                    }
                    sel_vertices_counts[i].push_back(vertex_count);
                    if (reindex_mask)
                        sel_mask_ids[i].push_back(mask_id_to_idx[static_cast<int>(obj_idx)]);
                    else
                        sel_mask_ids[i].push_back(static_cast<int>(obj_idx));
                }
                buffer_offset += static_cast<size_t>(vertex_count);
            }
        }
    }

    for (unsigned i = 0; i < _user_batch_size; i++) {
        float *select_mask_buffers = _output[i].data();
        out_list[i]->set_dims({_output[i].size(), 1});
        out_list[i]->set_mem_handle(static_cast<void *>(select_mask_buffers));
    }
    return &out_list;
}
