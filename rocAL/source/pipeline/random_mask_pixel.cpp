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

#include "pipeline/random_mask_pixel.h"

#include <algorithm>
#include <omp.h>

#include "parameters/parameter_factory.h"
#include "pipeline/log.h"

RandomMaskPixel::RandomMaskPixel(size_t user_batch_size, size_t cpu_num_threads)
    : _user_batch_size(user_batch_size), _cpu_num_threads(cpu_num_threads) {}

void RandomMaskPixel::set_config(bool is_foreground, int value, bool is_threshold) {
    _is_foreground = is_foreground;
    _value = value;
    _is_threshold = is_threshold;
}

void RandomMaskPixel::init_rngs() {
    const unsigned seed = ParameterFactory::instance()->get_seed();
    if (_seed == seed && _rngs.size() == _user_batch_size) {
        return;
    }
    _seed = seed;
    _rngs.resize(_user_batch_size);
    for (size_t i = 0; i < _user_batch_size; i++) {
        std::seed_seq seq_pixel{seed, 0x4D504958u, static_cast<unsigned>(i)};  // "MPIX"
        _rngs[i].seed(seq_pixel);
    }
}

int64_t RandomMaskPixel::find_pixel(const std::vector<int> &start, const std::vector<int> &foreground_count, int64_t val, int count) {
    if (val < 0 || val >= count || start.empty() || foreground_count.empty() || start.size() != foreground_count.size()) {
        return -1;
    }
    auto it = std::upper_bound(foreground_count.begin(), foreground_count.end(), val);
    if (it == foreground_count.begin()) {
        return -1;
    }
    size_t idx = static_cast<size_t>(it - foreground_count.begin() - 1);
    return start[idx] + (val - foreground_count[idx]);
}

// Select a random pixel per sample from the batch of segmentation masks.
// When _is_foreground is false, picks a uniformly random pixel.
// When _is_foreground is true, builds run-length encoded foreground spans
// (using either threshold or equality matching), then uniformly samples
// among foreground pixels. Falls back to a uniformly random pixel if no
// foreground pixels exist in the mask.
TensorList *RandomMaskPixel::run(rocalTensorList *input, TensorList &out_list) {
    init_rngs();
    _output_coords.clear();
    _output_coords.resize(_user_batch_size * 2);

    const int nthreads = static_cast<int>(std::max<size_t>(1, std::min(_cpu_num_threads, _user_batch_size)));

    if (!_is_foreground) {
#pragma omp parallel for num_threads(nthreads)
        for (unsigned i = 0; i < _user_batch_size; i++) {
            auto &rng = _rngs[i];
            const auto width = input->at(i)->dims().at(0);
            const auto height = input->at(i)->dims().at(1);
            if (width == 0 || height == 0) {
                _output_coords[i * 2] = 0;
                _output_coords[i * 2 + 1] = 0;
                continue;
            }
            auto row = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(height) - 1)(rng);
            auto col = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(width) - 1)(rng);
            _output_coords[i * 2] = static_cast<int>(row);
            _output_coords[i * 2 + 1] = static_cast<int>(col);
        }
    } else if (_is_threshold) {
#pragma omp parallel for num_threads(nthreads)
        for (unsigned i = 0; i < _user_batch_size; i++) {
            std::vector<int> start;
            std::vector<int> foreground_count;
            size_t id = 0;
            int count = 0;
            auto &rng = _rngs[i];
            int *mask_buffer = static_cast<int *>(input->at(i)->buffer());
            const auto width = input->at(i)->dims().at(0);
            const auto height = input->at(i)->dims().at(1);
            const size_t buffer_size = width * height;
            if (!mask_buffer || buffer_size == 0) {
                _output_coords[i * 2] = 0;
                _output_coords[i * 2 + 1] = 0;
                continue;
            }

            while (id < buffer_size) {
                if (mask_buffer[id] <= _value) {
                    id++;
                } else {
                    start.push_back(static_cast<int>(id++));
                    foreground_count.push_back(count++);
                    while (id < buffer_size && mask_buffer[id] > _value) {
                        id++;
                        count++;
                    }
                }
            }

            if (count != 0) {
                auto dist = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(count) - 1);
                auto flat_idx = find_pixel(start, foreground_count, dist(rng), count);
                if (flat_idx < 0) {
                    auto row = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(height) - 1)(rng);
                    auto col = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(width) - 1)(rng);
                    _output_coords[i * 2] = static_cast<int>(row);
                    _output_coords[i * 2 + 1] = static_cast<int>(col);
                    continue;
                }
                auto row = flat_idx / static_cast<int64_t>(width);
                auto col = flat_idx % static_cast<int64_t>(width);
                _output_coords[i * 2] = static_cast<int>(row);
                _output_coords[i * 2 + 1] = static_cast<int>(col);
            } else {
                auto row = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(height) - 1)(rng);
                auto col = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(width) - 1)(rng);
                _output_coords[i * 2] = static_cast<int>(row);
                _output_coords[i * 2 + 1] = static_cast<int>(col);
            }
        }
    } else {
#pragma omp parallel for num_threads(nthreads)
        for (unsigned i = 0; i < _user_batch_size; i++) {
            std::vector<int> start;
            std::vector<int> foreground_count;
            size_t id = 0;
            int count = 0;
            auto &rng = _rngs[i];
            int *mask_buffer = static_cast<int *>(input->at(i)->buffer());
            const auto width = input->at(i)->dims().at(0);
            const auto height = input->at(i)->dims().at(1);
            const size_t buffer_size = width * height;
            if (!mask_buffer || buffer_size == 0) {
                _output_coords[i * 2] = 0;
                _output_coords[i * 2 + 1] = 0;
                continue;
            }

            while (id < buffer_size) {
                if (mask_buffer[id] != _value) {
                    id++;
                } else {
                    start.push_back(static_cast<int>(id++));
                    foreground_count.push_back(count++);
                    while (id < buffer_size && mask_buffer[id] == _value) {
                        id++;
                        count++;
                    }
                }
            }

            if (count != 0) {
                auto dist = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(count) - 1);
                auto flat_idx = find_pixel(start, foreground_count, dist(rng), count);
                if (flat_idx < 0) {
                    auto row = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(height) - 1)(rng);
                    auto col = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(width) - 1)(rng);
                    _output_coords[i * 2] = static_cast<int>(row);
                    _output_coords[i * 2 + 1] = static_cast<int>(col);
                    continue;
                }
                auto row = flat_idx / static_cast<int64_t>(width);
                auto col = flat_idx % static_cast<int64_t>(width);
                _output_coords[i * 2] = static_cast<int>(row);
                _output_coords[i * 2 + 1] = static_cast<int>(col);
            } else {
                auto row = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(height) - 1)(rng);
                auto col = std::uniform_int_distribution<int64_t>(0, static_cast<int64_t>(width) - 1)(rng);
                _output_coords[i * 2] = static_cast<int>(row);
                _output_coords[i * 2 + 1] = static_cast<int>(col);
            }
        }
    }

    int *random_data_buffers = _output_coords.data();
    const std::vector<size_t> random_tensor_dims = {2};
    for (unsigned i = 0; i < _user_batch_size; i++) {
        out_list[i]->set_dims(random_tensor_dims);
        out_list[i]->set_mem_handle(static_cast<void *>(random_data_buffers));
        random_data_buffers += 2;
    }
    return &out_list;
}
