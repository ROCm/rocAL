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

#pragma once
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "pipeline/content_hash.h"
#include "pipeline/tensor.h"

struct CacheEntry {
    std::set<int> labels;
    std::unordered_map<int, std::vector<std::vector<std::vector<unsigned>>>> class_boxes;
    std::unordered_map<int, int> total_boxes;

    bool Get(std::vector<std::vector<std::vector<unsigned>>> &boxes, int label) const {
        auto it = class_boxes.find(label);
        if (it == class_boxes.end())
            return false;
        boxes = it->second;
        return true;
    }

    void Put(int label, const std::vector<std::vector<std::vector<unsigned>>> &boxes) {
        class_boxes[label] = boxes;
    }
};

class RandomObjectBbox {
   public:
    RandomObjectBbox(vx_context context, size_t user_batch_size, size_t cpu_num_threads);
    ~RandomObjectBbox();

    TensorList *init(Tensor *input, std::string output_format, int k_largest, float foreground_prob, bool cache_objects);
    void update();

    void *box1_buf() const { return _box1_buf; }
    void *box2_buf() const { return _box2_buf; }

   private:
    void findLabels(const u_int8_t *input, std::set<int> &labels, std::vector<int> roi_size, std::vector<size_t> max_size);
    void filterByLabel(const u_int8_t *input, std::vector<int> &output, std::vector<int> roi_size, std::vector<size_t> max_size, int label);
    void labelRow(const int *label_base, const int *in_row, int *out_row, unsigned length);
    int disjointGetGroup(const int &x) { return x; }
    int disjointSetGroup(int &x, int new_id);
    int disjointFind(int *items, int x);
    int disjointMerge(int *items, int x, int y);
    void mergeRow(int *label_base, const int *in1, const int *in2, int *out1, int *out2, unsigned n);
    int labelMergeFunc(const u_int8_t *input, int &selected_label, std::vector<int> &size, std::vector<size_t> &max_size, std::vector<int> &output_compact, std::mt19937 &rng, CacheEntry *cache_entry);
    bool hit(std::vector<unsigned> &hits, unsigned idx);
    void get_label_boundingboxes(std::vector<std::vector<std::vector<unsigned>>> &boxes, std::vector<std::pair<unsigned, unsigned>> ranges, std::vector<unsigned> hits, int *in, std::vector<int> origin, unsigned width);
    int pick_box(const std::vector<std::vector<std::vector<unsigned>>> &boxes, std::mt19937 &rng, int k_largest = -1);

    vx_context _context;
    size_t _user_batch_size;
    size_t _cpu_num_threads;
    Tensor *_label_tensor = nullptr;
    Tensor *_box1_tensor = nullptr;
    Tensor *_box2_tensor = nullptr;
    void *_box1_buf = nullptr;
    void *_box2_buf = nullptr;
    TensorList _tensor_list;
    std::string _output_format;
    int _k_largest = -1;
    float _foreground_prob = 1.0f;
    bool _cache_boxes = false;
    std::unordered_map<content_hash_t, CacheEntry> _boxes_cache;
};
