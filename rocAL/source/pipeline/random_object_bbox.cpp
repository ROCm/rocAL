/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "pipeline/random_object_bbox.h"

#include <algorithm>
#include <numeric>
#include <omp.h>

#include "parameters/parameter_factory.h"
#include "pipeline/log.h"

RandomObjectBbox::RandomObjectBbox(vx_context context, size_t user_batch_size, size_t cpu_num_threads)
    : _context(context), _user_batch_size(user_batch_size), _cpu_num_threads(cpu_num_threads) {}

RandomObjectBbox::~RandomObjectBbox() {
    _tensor_list.release();
    if (_box1_buf != nullptr) {
        free(_box1_buf);
        _box1_buf = nullptr;
    }
    if (_box2_buf != nullptr) {
        free(_box2_buf);
        _box2_buf = nullptr;
    }
}

// Allocate output tensors based on the chosen format and store configuration.
// For "anchor_shape" and "start_end", two tensors are created (one for each component).
// For "box", a single tensor with double the spatial dims holds concatenated start+end.
TensorList *RandomObjectBbox::init(Tensor *input, std::string output_format, int k_largest, float foreground_prob, bool cache_objects) {
    if (output_format != "start_end" && output_format != "anchor_shape" && output_format != "box")
        THROW("RandomObjectBbox: invalid output_format '" + output_format + "'. Must be one of: 'anchor_shape', 'start_end', 'box'")
    _label_tensor = input;
    _k_largest = k_largest;
    _foreground_prob = foreground_prob;
    _cache_boxes = cache_objects;
    auto output_dims = _label_tensor->num_of_dims() - 1;
    _output_format = output_format;
    if (output_format == "start_end" || output_format == "anchor_shape") {
        // create new instance of tensor class
        std::vector<size_t> box1_dims = {_user_batch_size, output_dims};
        auto box1_info = TensorInfo(std::move(box1_dims), RocalMemType::HOST, RocalTensorDataType::INT32);
        _box1_tensor = new Tensor(box1_info);

        // allocate memory for the raw buffer pointer in tensor object
        allocate_host_or_pinned_mem(&_box1_buf, _user_batch_size * output_dims * sizeof(int), RocalMemType::HOST);
        _box1_tensor->create_from_ptr(_context, _box1_buf);

        // create new instance of tensor class
        std::vector<size_t> box2_dims = {_user_batch_size, output_dims};
        auto box2_info = TensorInfo(std::move(box2_dims), RocalMemType::HOST, RocalTensorDataType::INT32);
        _box2_tensor = new Tensor(box2_info);

        // allocate memory for the raw buffer pointer in tensor object
        allocate_host_or_pinned_mem(&_box2_buf, _user_batch_size * output_dims * sizeof(int), RocalMemType::HOST);
        _box2_tensor->create_from_ptr(_context, _box2_buf);
        _tensor_list.push_back(_box1_tensor);
        _tensor_list.push_back(_box2_tensor);
    } else if (output_format == "box") {
        // create new instance of tensor class
        std::vector<size_t> box1_dims = {_user_batch_size, output_dims * 2};
        auto box1_info = TensorInfo(std::move(box1_dims), RocalMemType::HOST, RocalTensorDataType::INT32);
        _box1_tensor = new Tensor(box1_info);

        // allocate memory for the raw buffer pointer in tensor object
        allocate_host_or_pinned_mem(&_box1_buf, _user_batch_size * output_dims * 2 * sizeof(int), RocalMemType::HOST);
        _box1_tensor->create_from_ptr(_context, _box1_buf);
        _tensor_list.push_back(_box1_tensor);
    }
    return &_tensor_list;
}

// Called once per pipeline iteration to recompute bounding boxes for the current batch.
// For each sample:
//   1. Decide foreground vs background based on _foreground_prob.
//   2. Run connected-component labeling (labelMergeFunc) to find objects in the segmentation mask.
//   3. Compute axis-aligned bounding boxes around each connected component.
//   4. Randomly pick one box (optionally restricted to k-largest by volume).
//   5. Write the selected box coordinates into the output buffers in the requested format.
// When caching is disabled, samples are processed in parallel via OpenMP.
void RandomObjectBbox::update() {
    uint8_t *input = static_cast<uint8_t *>(_label_tensor->buffer());
    auto roi_dims = reinterpret_cast<int *>(_label_tensor->info().roi().get_ptr());
    std::vector<size_t> max_size = _label_tensor->info().max_shape();
    auto single_image_size = _label_tensor->data_size() / _user_batch_size;
    auto input_dims = _label_tensor->num_of_dims() - 1;
    int64_t seed = ParameterFactory::instance()->get_seed_from_seedsequence();
    BatchRNG _rng = {seed, static_cast<int>(_user_batch_size)};
    std::uniform_real_distribution<float> foreground(0.0f, 1.0f);
    int *box1_buf = static_cast<int *>(_box1_buf);
    int *box2_buf = (_box2_buf != nullptr) ? static_cast<int *>(_box2_buf) : nullptr;

    auto process_sample = [&](uint i) {
        auto sample_idx = i * input_dims;
        int *input_shape = &roi_dims[sample_idx * 2 + input_dims];
        std::vector<int> roi_size;
        for (uint j = 0; j < input_dims; j++) {
            roi_size.push_back(input_shape[j]);
        }
        std::vector<int> output_compact;
        auto label = input + i * single_image_size;
        int total_box = 0;
        bool fg = foreground(_rng[i]) < _foreground_prob;
        CacheEntry *cache_entry = nullptr;
        content_hash_t hash = {};
        if (_cache_boxes) {
            content_hash(hash, label, single_image_size * sizeof(uint8_t));
            // Bound the number of cached entries to avoid unbounded memory growth.
            static const std::size_t max_cache_entries = 1024;
            if (_boxes_cache.size() > max_cache_entries) {
                _boxes_cache.clear();
            }
            cache_entry = &_boxes_cache[hash];
        }
        int selected_label = -1;
        std::vector<std::vector<std::vector<unsigned>>> boxes;  // total - lo,hi - 4D
        // Validate roi_size dimensionality before running 4D-specific processing
        if (roi_size.size() < 4) {
            THROW("RandomObjectBbox: Expected roi_size with at least 4 dimensions, got " + TOSTR(roi_size.size()))
        } else if (fg) {
            total_box = labelMergeFunc(label, selected_label, roi_size, max_size, output_compact, _rng[i], cache_entry);
        }
        if (total_box) {
            if (!cache_entry || !cache_entry->Get(boxes, selected_label)) {
            std::vector<std::pair<unsigned, unsigned>> ranges;      // totalbox - lo,hi
            std::vector<unsigned> hits;
            boxes.resize(total_box);
            ranges.resize(total_box);
            hits.resize((total_box / 32 + !!(total_box % 32)));
            auto out_row = output_compact.data();
            for (int d1 = 0; d1 < roi_size[0]; d1++) {
                for (int d2 = 0; d2 < roi_size[1]; d2++) {
                    for (int d3 = 0; d3 < roi_size[2]; d3++) {
                        std::vector<int> origin{d1, d2, d3, 0};
                        get_label_boundingboxes(boxes, ranges, hits, out_row, origin, roi_size[3]);
                        out_row += roi_size[3];
                    }
                }
                }
                if (cache_entry)
                    cache_entry->Put(selected_label, boxes);
            }
            int chosen_box_idx = pick_box(boxes, _rng[i], _k_largest);
            if (chosen_box_idx == -1) { ERR("No ROI regions found in input. Setting input shape as ROI region"); }
            if (_output_format == "box") {
                for (uint j = 0; j < input_dims; j++) {
                    if (chosen_box_idx >= 0) {
                        box1_buf[sample_idx + j] = boxes[chosen_box_idx][0][j];
                        box1_buf[sample_idx + j + input_dims] = boxes[chosen_box_idx][1][j];
                    }
                    else {
                        box1_buf[sample_idx + j] = 0;
                        box1_buf[sample_idx + j + input_dims] = input_shape[j];
                    }
                }
            } else if (_output_format == "anchor_shape") {
                for (uint j = 0; j < input_dims; j++) {
                    if (chosen_box_idx >= 0) {
                        box1_buf[sample_idx + j] = boxes[chosen_box_idx][0][j];
                        box2_buf[sample_idx + j] = boxes[chosen_box_idx][1][j] - boxes[chosen_box_idx][0][j];
                    }
                    else {
                        box1_buf[sample_idx + j] = 0;
                        box2_buf[sample_idx + j] = input_shape[j];
                    }
                }
            } else if (_output_format == "start_end") {
                for (uint j = 0; j < input_dims; j++) {
                    if (chosen_box_idx >= 0) {
                        box1_buf[sample_idx + j] = boxes[chosen_box_idx][0][j];
                        box2_buf[sample_idx + j] = boxes[chosen_box_idx][1][j];
                    }
                    else {
                        box1_buf[sample_idx + j] = 0;
                        box2_buf[sample_idx + j] = input_shape[j];
                    }
                }
            }
        } else {
            if (_output_format == "box") {
                for (uint j = 0; j < input_dims; j++) {
                    box1_buf[sample_idx + j] = 0;
                    box1_buf[sample_idx + j + input_dims] = input_shape[j];
                }
            } else {
                for (uint j = 0; j < input_dims; j++) {
                    box1_buf[sample_idx + j] = 0;
                    box2_buf[sample_idx + j] = input_shape[j];
                }
            }
        }
    };

    if (_cache_boxes) {
        for (uint i = 0; i < _user_batch_size; i++) {
            process_sample(i);
        }
    } else {
        auto num_threads = _cpu_num_threads;
#pragma omp parallel for num_threads(num_threads)
        for (uint i = 0; i < _user_batch_size; i++) {
            process_sample(i);
        }
    }
}

// Select a random bounding box index from the list of boxes.
// When k_largest > 0, sort boxes by volume (descending), then uniformly sample among the top-k.
// Otherwise, uniformly sample from all available boxes.
// Returns -1 when no boxes are available.
int RandomObjectBbox::pick_box(const std::vector<std::vector<std::vector<unsigned>>> &boxes, std::mt19937 &rng, int k_largest) {
    int n = boxes.size();
    if (n <= 0)
        return -1;
    if (k_largest > 0 && k_largest < n) {
        std::vector<std::pair<int64_t, int>> vol_idx;
        vol_idx.resize(n);
        for (int i = 0; i < n; i++) {
            std::vector<unsigned> crop_region;
            std::transform(boxes[i][1].begin(),boxes[i][1].end(), boxes[i][0].begin(),
               std::back_inserter(crop_region),
               [](const auto& hi, const auto& lo)
               {
                   return hi - lo;
               });
            int64_t volume_val = 1;
            for (auto val : crop_region) {
                volume_val *= val;
            }
            vol_idx[i] = {-volume_val, i};
        }
        std::sort(vol_idx.begin(), vol_idx.end());
        std::uniform_int_distribution<int> dist(0, std::min(n, k_largest) - 1);
        return vol_idx[dist(rng)].second;
    } else {
        std::uniform_int_distribution<int> dist(0, n - 1);
        return dist(rng);
    }
}

// Scan the input label tensor within the ROI and collect all unique label values.
// Uses stride-based indexing to handle the gap between ROI dimensions and the
// underlying max-allocated tensor dimensions.  Skips runs of identical values
// for efficiency.
void RandomObjectBbox::findLabels(const uint8_t *input, std::set<int> &labels, std::vector<int> roi_size, std::vector<size_t> max_size) {
    if (!roi_size.size() || !max_size.size())
        return;
    int prev = input[0];
    labels.insert(prev);
    int num_dims = roi_size.size();
    std::vector<unsigned> strides(num_dims + 1);
    strides[num_dims] = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
        strides[i] = strides[i + 1] * max_size[i];
    }
    auto index = 0;
    for (int c = 0; c < roi_size[0]; c++) {
        int outerDim1 = index;
        for (int d = 0; d < roi_size[1]; d++) {
            int outerDim2 = outerDim1;
            for (int h = 0; h < roi_size[2]; h++) {
                int outerDim3 = outerDim2;
                for (int w = 0; w < roi_size[3]; w++) {
                    auto value = input[outerDim3++];
                    if (value == prev)
                        continue;  // skip runs of equal labels
                    labels.insert(value);
                    prev = value;
                }
                outerDim2 += strides[3];
            }
            outerDim1 += strides[2];
        }
        index += strides[1];
    }
}

// Produce a binary mask from the input tensor: output[i] = 1 where input[i] == label, 0 otherwise.
// Uses stride-based indexing identical to findLabels to respect the gap between ROI and max dims.
void RandomObjectBbox::filterByLabel(const uint8_t *input, std::vector<int> &output, std::vector<int> roi_size, std::vector<size_t> max_size, int label) {
    int num_dims = roi_size.size();
    std::vector<unsigned> strides(num_dims + 1);
    strides[num_dims] = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
        strides[i] = strides[i + 1] * max_size[i];
    }
    int index = 0;
    int out_index = 0;
    for (int c = 0; c < roi_size[0]; c++) {
        int outerDim1 = index;
        for (int d = 0; d < roi_size[1]; d++) {
            int outerDim2 = outerDim1;
            for (int h = 0; h < roi_size[2]; h++) {
                int outerDim3 = outerDim2;
                for (int w = 0; w < roi_size[3]; w++) {
                    output[out_index++] = input[outerDim3++] == label;
                }
                outerDim2 += strides[3];
            }
            outerDim1 += strides[2];
        }
        index += strides[1];
    }
}

// Assign connected-component labels to a single row using run-length encoding.
// Each contiguous run of foreground pixels (in_row[i] != 0) receives a unique label
// derived from its position relative to label_base.  Background pixels get the
// sentinel value -1.
void RandomObjectBbox::labelRow(const int *label_base, const int *in_row, int *out_row, unsigned length) {
    int curr_label = -1;
    int bg_label = -1;
    int prev = 0;
    for (unsigned i = 0; i < length; i++) {
        if (in_row[i] != prev) {
            if (in_row[i] != 0) {
                curr_label = out_row + i - label_base;
            } else {
                curr_label = bg_label;
            }
        }
        out_row[i] = curr_label;
        prev = in_row[i];
    }
}

// Assign element x to a new group (new_id) and return its previous group.
// Used by the union-find algorithm during label merging.
int RandomObjectBbox::disjointSetGroup(int &x, int new_id) {
    int old = x;
    x = new_id;
    return old;
}

// Union-Find: find the root representative of element x with path compression.
// After finding the root, all intermediate elements on the path are updated to
// point directly to the root, speeding up subsequent lookups.
int RandomObjectBbox::disjointFind(int *items, int x) {
    int x0 = x;

    // find the label
    for (;;) {
        int g = disjointGetGroup(items[x]);
        if (g == x)
            break;
        x = g;
    }

    int r = x;

    // assign all intermediate labels to save time in subsequent calls
    x = x0;
    while (x != disjointGetGroup(items[x])) {
        x0 = disjointSetGroup(items[x], r);
        x = x0;
    }

    return r;
}

// Union-Find: merge the sets containing x and y.
// The smaller root becomes the child of the larger to maintain a balanced structure.
// Returns the new root of the merged set.
int RandomObjectBbox::disjointMerge(int *items, int x, int y) {
    y = disjointFind(items, y);
    x = disjointFind(items, x);
    if (x < y) {
        disjointSetGroup(items[y], x);
        return x;
    } else if (y < x) {
        disjointSetGroup(items[x], y);
        return y;
    } else {
        // already merged
        return x;
    }
}

// Merge connected-component labels between two adjacent rows (or slices).
// For each position where both rows have the same filtered label value, unify
// their component labels using the disjoint-set structure.  This propagates
// connectivity across the height/depth dimensions.
void RandomObjectBbox::mergeRow(int *label_base, const int *in1, const int *in2, int *out1, int *out2, unsigned n) {
    int bg_label = -1;
    int prev1 = bg_label;
    int prev2 = bg_label;
    for (unsigned i = 0, in_offset = 0, out_offset = 0; i < n; i++, in_offset += 1, out_offset += 1) {
        int &o1 = out1[out_offset];
        int &o2 = out2[out_offset];
        if (o1 != prev1 || o2 != prev2) {
            if (o1 != bg_label) {
                if (in1[in_offset] == in2[in_offset]) {
                    disjointMerge(label_base, o1, o2);
                }
            }
            prev1 = o1;
            prev2 = o2;
        }
    }
}

// Core connected-component labeling pipeline for a single sample.
//
// Steps:
//   1. Find all unique foreground labels in the input (or use cached labels).
//   2. Randomly select one label class.
//   3. Check the cache for a pre-computed result; if found, return early.
//   4. Filter the input to a binary mask for the selected label.
//   5. Label each row of the innermost dimension (width) independently using run-length encoding.
//   6. Merge labels between adjacent rows within each 2D slice (height direction).
//   7. Merge labels across slices (depth direction) using a hierarchical merge strategy.
//   8. Flatten the disjoint-set forest via path compression and remap labels to
//      sequential IDs (0, 1, 2, ...).
//
// Returns the total number of distinct connected components found.
int RandomObjectBbox::labelMergeFunc(const uint8_t *input, int &selected_label, std::vector<int> &size, std::vector<size_t> &max_size, std::vector<int> &output_compact, std::mt19937 &rng, CacheEntry *cache_entry) {
    int64_t total_buf_size = 1;
    for (auto val : size)
        total_buf_size *= val;
    std::vector<int> output_filtered;
    output_filtered.resize(total_buf_size);
    output_compact.resize(total_buf_size);
    std::fill(output_filtered.begin(), output_filtered.end(), 0);
    std::fill(output_compact.begin(), output_compact.end(), -1);

    if (selected_label == -1) {
        const std::set<int> *labels_ptr = nullptr;
        std::set<int> labels_found;

        if (cache_entry && !cache_entry->labels.empty()) {
            labels_ptr = &cache_entry->labels;
        } else {
            findLabels(input, labels_found, size, max_size);
            labels_found.erase(0);  // remove background label
            if (labels_found.empty())
                return 0;  // all labels belong to background
            if (cache_entry && cache_entry->labels.empty())
                cache_entry->labels = labels_found;
            labels_ptr = &labels_found;
        }

        if (labels_ptr->size() == 1) {
            selected_label = *labels_ptr->begin();
        } else {
            std::uniform_int_distribution<size_t> label_dist(0, labels_ptr->size() - 1);
            selected_label = *std::next(labels_ptr->begin(), label_dist(rng));
        }
    }
    if (cache_entry) {
        auto it = cache_entry->total_boxes.find(selected_label);
        if (it != cache_entry->total_boxes.end()) {
            int cached_total = it->second;
            if (cached_total == 0 || cache_entry->class_boxes.count(selected_label)) {
                return cached_total;
            }
        }
    }
    filterByLabel(input, output_filtered, size, max_size, selected_label);
    for (int i = 0; i < size[0]; i++) {
        for (int j = 0; j < size[1]; j++) {
            for (int k = 0; k < size[2]; k++) {
                labelRow(output_compact.data(),
                         output_filtered.data() + (i * size[1] * size[2] * size[3]) + (j * (size[2] * size[3])) + (k * size[3]),
                         output_compact.data() + (i * size[1] * size[2] * size[3]) + (j * (size[2] * size[3])) + (k * size[3]),
                         size[3]);
                if (k > 0) {
                    mergeRow(output_compact.data(),
                             output_filtered.data() + (i * size[1] * size[2] * size[3]) + (j * (size[2] * size[3])) + (k - 1) * size[3],
                             output_filtered.data() + (i * size[1] * size[2] * size[3]) + (j * (size[2] * size[3])) + (k)*size[3],
                             output_compact.data() + (i * size[1] * size[2] * size[3]) + (j * (size[2] * size[3])) + ((k - 1) * size[3]),
                             output_compact.data() + (i * size[1] * size[2] * size[3]) + (j * (size[2] * size[3])) + (k * size[3]),
                             size[3]);
                }
            }
        }
    }
    for (int k = 0; k < size[0]; k++) {
        for (int stride = 1; stride <= size[1]; stride *= 2) {
            for (int i = stride; i < size[1]; i += 2 * stride) {
                auto out_slice = output_compact.data() + (i * size[2] * size[3]);
                auto in_slice = output_filtered.data() + (i * size[2] * size[3]);
                auto prev_out = output_compact.data() + ((i - 1) * size[2] * size[3]);
                auto prev_in = output_filtered.data() + ((i - 1) * size[2] * size[3]);
                mergeRow(output_compact.data(),
                         prev_in, in_slice, prev_out, out_slice, size[2] * size[3]);
            }
        }
    }
    constexpr int kBackground = -1;
    std::set<int> roots;
    int prev = kBackground;
    int remapped = kBackground;
    for (int64_t i = 0; i < total_buf_size; i++) {
        int curr = output_compact[i];
        if (curr == kBackground) {
            prev = kBackground;
            continue;
        }
        if (curr != prev) {
            prev = curr;
            remapped = disjointFind(output_compact.data(), static_cast<int>(i));
            roots.insert(remapped);
        } else {
            output_compact[i] = remapped;
        }
    }

    std::map<int, int> label_map;
    int counter = 0;
    for (int root : roots) {
        label_map[root] = counter++;
    }

    prev = kBackground;
    remapped = kBackground;
    for (int64_t i = 0; i < total_buf_size; i++) {
        int curr = output_compact[i];
        if (curr == kBackground) {
            prev = kBackground;
            continue;
        }
        if (curr != prev) {
            prev = curr;
            remapped = label_map[curr];
        }
        output_compact[i] = remapped;
    }

    if (cache_entry)
        cache_entry->total_boxes[selected_label] = counter;
    return counter;
}

// Test-and-set a bit in the hit bitmap.  Returns true if the bit was already set
// (i.e., the label was already encountered in this row), false if this is the
// first occurrence.  Used to track which connected-component labels appear in
// a row so their coordinate ranges can be initialized or extended.
bool RandomObjectBbox::hit(std::vector<unsigned> &hits, unsigned idx) {
    unsigned flag = (1u << (idx & 31));
    unsigned &h = hits[idx >> 5];
    bool ret = h & flag;
    h |= flag;
    return ret;
}

// Process one row (innermost dimension) of the compact label tensor to update
// axis-aligned bounding boxes.
//
// For each label found in the row:
//   - Record the min/max column index in `ranges` (used for the width dimension).
//   - If this is the first time the label's box is seen, create it from `origin`.
//   - Otherwise, expand the existing box to encompass the new coordinates.
//
// The `hits` bitmap tracks which labels appear in this row, enabling efficient
// skipping of absent labels when iterating over the results.
void RandomObjectBbox::get_label_boundingboxes(std::vector<std::vector<std::vector<unsigned>>> &boxes,
                                          std::vector<std::pair<unsigned, unsigned>> &ranges,
                                          std::vector<unsigned> &hits,
                                          int *in,
                                          std::vector<int> origin,
                                          unsigned width) {
    for (auto &mask : hits) {
        mask = 0u;  // mark all labels as not found in this row
    }

    int ndim = origin.size();

    const unsigned nboxes = ranges.size();
    int background = -1;
    for (unsigned i = 0; i < width; i++) {
        if (in[i] != background) {
            // We make a "hole" in the label indices for the background.
            int skip_bg = (background >= 0 && in[i] >= background);
            unsigned idx = static_cast<unsigned>(in[i]) - skip_bg;
            // deliberate use of unsigned overflow to detect negative labels as out-of-range
            if (idx < nboxes) {
                if (!hit(hits, idx)) {
                    ranges[idx].first = i;
                }
                ranges[idx].second = i;
            }
        }
    }

    std::vector<unsigned> lo(4, 0);
    std::vector<unsigned> hi(4, 0);

    for (int i = 0; i < ndim; i++) {
        lo[i] = origin[i];
        hi[i] = origin[i] + 1;  // one past
    }
    const int d = 3;

    for (uint word = 0; word < hits.size(); word++) {
        unsigned mask = hits[word];
        unsigned i = 32 * word;
        while (mask) {
            if ((mask & 0xffu) == 0) {  // skip 8 labels if not set
                mask >>= 8;
                i += 8;
                continue;
            }
            if (mask & 1) {  // label found? mark it
                lo[d] = ranges[i].first + origin[d];
                hi[d] = (ranges[i].second + origin[d] + 1);  // one past the index found in this function
                if (boxes[i].empty()) {
                    // empty box - create a new one
                    boxes[i].push_back(lo);
                    boxes[i].push_back(hi);
                } else {
                    // expand existing
                    std::transform(boxes[i][0].begin(), boxes[i][0].end(), lo.begin(), boxes[i][0].begin(),
                                   [](const auto &val1, const auto &val2) {
                                       return val1 < val2 ? val1 : val2;
                                   });
                    std::transform(boxes[i][1].begin(), boxes[i][1].end(), hi.begin(), boxes[i][1].begin(),
                                   [](const auto &val1, const auto &val2) {
                                       return val1 > val2 ? val1 : val2;
                                   });
                }
            }
            mask >>= 1;
            i++;  // skip one label
        }
    }
}
